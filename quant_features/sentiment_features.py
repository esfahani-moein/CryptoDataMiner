"""
Sentiment Features Module

Market sentiment indicators from funding rates, open interest, and positioning data.
These features capture the sentiment and positioning of market participants.

Data Sources:
- Funding Rate: Cost of holding perpetual contracts
- Open Interest: Total open positions
- Long/Short Ratios: Trader positioning

All features use only past data to avoid look-ahead bias.
"""

from typing import List, Optional
import polars as pl
import numpy as np


# ============================================================================
# FUNDING RATE FEATURES
# ============================================================================

def add_funding_rate_features(
    df: pl.DataFrame,
    funding_df: Optional[pl.DataFrame] = None,
    windows: List[int] = [3, 8, 24]  # 8h funding = 3 per day
) -> pl.DataFrame:
    """
    Add funding rate derived features.
    
    Funding rate interpretation:
    - Positive: Longs pay shorts (bullish sentiment, crowded long)
    - Negative: Shorts pay longs (bearish sentiment, crowded short)
    - High absolute value: Strong directional bias
    
    Features:
    - funding_rate: Current funding rate
    - funding_ma_{w}: Moving average of funding
    - funding_cumsum_{w}: Cumulative funding paid
    - funding_zscore: Funding rate z-score
    - funding_direction: Sign of funding
    
    Args:
        df: Main DataFrame with OHLCV data
        funding_df: Funding rate DataFrame (optional, if already merged)
        windows: Lookback windows (in number of funding periods)
        
    Returns:
        DataFrame with funding features
    """
    result = df.clone()
    
    # Check if funding rate already in data
    if "last_funding_rate" not in result.columns:
        if funding_df is None:
            return result  # No funding data available
        # Would need to merge - handled by data_loader
        return result
    
    # Rename for clarity
    if "funding_rate" not in result.columns and "last_funding_rate" in result.columns:
        result = result.rename({"last_funding_rate": "funding_rate"})
    
    fr_col = "funding_rate" if "funding_rate" in result.columns else "last_funding_rate"
    
    # Forward fill funding rate (it's only updated every 8 hours)
    result = result.with_columns([
        pl.col(fr_col).forward_fill().alias(fr_col)
    ])
    
    for window in windows:
        # Rolling mean of funding rate
        result = result.with_columns([
            pl.col(fr_col).rolling_mean(window).alias(f"funding_ma_{window}")
        ])
        
        # Rolling std for z-score
        result = result.with_columns([
            pl.col(fr_col).rolling_std(window).alias(f"_funding_std_{window}")
        ])
        
        # Z-score
        result = result.with_columns([
            ((pl.col(fr_col) - pl.col(f"funding_ma_{window}")) 
             / (pl.col(f"_funding_std_{window}") + 1e-10))
            .alias(f"funding_zscore_{window}")
        ])
        
        # Cumulative funding (cost to hold position)
        result = result.with_columns([
            pl.col(fr_col).rolling_sum(window).alias(f"funding_cumsum_{window}")
        ])
        
        # Clean temp columns
        result = result.drop(f"_funding_std_{window}")
    
    # Funding direction and extremity
    result = result.with_columns([
        pl.col(fr_col).sign().alias("funding_direction"),
        pl.col(fr_col).abs().alias("funding_abs")
    ])
    
    # Funding rate percentile (using longer window)
    if len(windows) > 0:
        longest_window = max(windows) * 3
        result = result.with_columns([
            (pl.col(fr_col).rolling_rank(longest_window) 
             / longest_window * 100)
            .alias("funding_percentile")
        ])
    
    # Funding rate regime (extreme positive/negative/neutral)
    # Using 2 standard deviations as threshold
    if f"funding_ma_{windows[-1]}" in result.columns:
        result = result.with_columns([
            pl.when(pl.col(fr_col) > 0.0003).then(2)  # Very bullish
            .when(pl.col(fr_col) > 0.0001).then(1)   # Bullish
            .when(pl.col(fr_col) < -0.0003).then(-2) # Very bearish
            .when(pl.col(fr_col) < -0.0001).then(-1) # Bearish
            .otherwise(0)                             # Neutral
            .alias("funding_regime")
        ])
    
    return result


# ============================================================================
# OPEN INTEREST FEATURES
# ============================================================================

def add_open_interest_features(
    df: pl.DataFrame,
    windows: List[int] = [6, 24, 72]  # 5-min intervals: 6=30min, 24=2h, 72=6h
) -> pl.DataFrame:
    """
    Add open interest derived features.
    
    Open Interest interpretation:
    - Rising OI + Rising Price: New longs entering (bullish)
    - Rising OI + Falling Price: New shorts entering (bearish)
    - Falling OI + Rising Price: Short covering (less bullish)
    - Falling OI + Falling Price: Long liquidation (less bearish)
    
    Features:
    - oi_change: Change in open interest
    - oi_pct_change: Percentage change
    - oi_ma: Moving average
    - oi_regime: Combined OI/price regime signal
    
    Args:
        df: DataFrame with open interest columns
        windows: Lookback windows
        
    Returns:
        DataFrame with OI features
    """
    result = df.clone()
    
    # Check for OI columns (might be different names)
    oi_col = None
    if "sum_open_interest" in result.columns:
        oi_col = "sum_open_interest"
    elif "open_interest" in result.columns:
        oi_col = "open_interest"
    
    if oi_col is None:
        return result  # No OI data
    
    # Forward fill OI (in case of gaps)
    result = result.with_columns([
        pl.col(oi_col).forward_fill().alias("oi")
    ])
    
    # OI change
    result = result.with_columns([
        (pl.col("oi") - pl.col("oi").shift(1)).alias("oi_change"),
        ((pl.col("oi") - pl.col("oi").shift(1)) / (pl.col("oi").shift(1) + 1e-10) * 100)
        .alias("oi_pct_change")
    ])
    
    for window in windows:
        # Rolling mean of OI
        result = result.with_columns([
            pl.col("oi").rolling_mean(window).alias(f"oi_ma_{window}")
        ])
        
        # OI relative to MA
        result = result.with_columns([
            (pl.col("oi") / (pl.col(f"oi_ma_{window}") + 1e-10))
            .alias(f"oi_ratio_{window}")
        ])
        
        # OI change sum (momentum)
        result = result.with_columns([
            pl.col("oi_change").rolling_sum(window).alias(f"oi_momentum_{window}")
        ])
    
    # OI-Price regime (requires price data)
    if "close" in result.columns:
        # Price change
        if "ret_1" not in result.columns:
            result = result.with_columns([
                (pl.col("close").log() - pl.col("close").shift(1).log()).alias("_ret_1")
            ])
            ret_col = "_ret_1"
        else:
            ret_col = "ret_1"
        
        # Combined regime signal
        # +2: OI up, Price up (new longs)
        # +1: OI down, Price up (short covering)
        # -1: OI down, Price down (long liquidation)
        # -2: OI up, Price down (new shorts)
        result = result.with_columns([
            (pl.when((pl.col("oi_change") > 0) & (pl.col(ret_col) > 0)).then(2)
             .when((pl.col("oi_change") < 0) & (pl.col(ret_col) > 0)).then(1)
             .when((pl.col("oi_change") < 0) & (pl.col(ret_col) < 0)).then(-1)
             .when((pl.col("oi_change") > 0) & (pl.col(ret_col) < 0)).then(-2)
             .otherwise(0))
            .alias("oi_price_regime")
        ])
        
        # Clean temp column
        if "_ret_1" in result.columns:
            result = result.drop("_ret_1")
    
    # OI value (if available)
    if "sum_open_interest_value" in result.columns:
        result = result.with_columns([
            pl.col("sum_open_interest_value").forward_fill().alias("oi_value")
        ])
        
        result = result.with_columns([
            ((pl.col("oi_value") - pl.col("oi_value").shift(1)) 
             / (pl.col("oi_value").shift(1) + 1e-10) * 100)
            .alias("oi_value_pct_change")
        ])
    
    return result


# ============================================================================
# LONG/SHORT RATIO FEATURES
# ============================================================================

def add_long_short_ratio_features(
    df: pl.DataFrame,
    windows: List[int] = [6, 24, 72]
) -> pl.DataFrame:
    """
    Add long/short ratio features.
    
    Ratio interpretation:
    - >1: More longs than shorts (crowded long)
    - <1: More shorts than longs (crowded short)
    - Extreme values: Potential contrarian signal
    
    Available ratios:
    - count_toptrader_long_short_ratio: Top trader account ratio
    - sum_toptrader_long_short_ratio: Top trader position ratio
    - count_long_short_ratio: All account ratio
    - sum_taker_long_short_vol_ratio: Taker volume ratio
    
    Args:
        df: DataFrame with ratio columns
        windows: Lookback windows
        
    Returns:
        DataFrame with ratio features
    """
    result = df.clone()
    
    # Define ratio columns to process
    ratio_cols = [
        ("count_toptrader_long_short_ratio", "top_trader_count"),
        ("sum_toptrader_long_short_ratio", "top_trader_position"),
        ("count_long_short_ratio", "account_ratio"),
        ("sum_taker_long_short_vol_ratio", "taker_ratio")
    ]
    
    for original_col, short_name in ratio_cols:
        if original_col not in result.columns:
            continue
        
        # Forward fill (data comes at intervals)
        result = result.with_columns([
            pl.col(original_col).forward_fill().alias(short_name)
        ])
        
        # Log ratio (symmetric around 0)
        result = result.with_columns([
            pl.col(short_name).log().alias(f"{short_name}_log")
        ])
        
        for window in windows:
            # Moving average
            result = result.with_columns([
                pl.col(short_name).rolling_mean(window).alias(f"{short_name}_ma_{window}")
            ])
            
            # Deviation from mean
            result = result.with_columns([
                (pl.col(short_name) - pl.col(f"{short_name}_ma_{window}"))
                .alias(f"{short_name}_dev_{window}")
            ])
        
        # Change in ratio
        result = result.with_columns([
            (pl.col(short_name) - pl.col(short_name).shift(1)).alias(f"{short_name}_change")
        ])
        
        # Extreme detection (potential contrarian signals)
        # When ratio is very high, crowd is too bullish (contrarian bearish)
        result = result.with_columns([
            pl.when(pl.col(short_name) > 2.5).then(1)  # Extreme bullish (contrarian bearish)
            .when(pl.col(short_name) < 0.4).then(-1)  # Extreme bearish (contrarian bullish)
            .otherwise(0)
            .alias(f"{short_name}_extreme")
        ])
    
    # Composite sentiment score
    sentiment_cols = [f"{name}_log" for _, name in ratio_cols 
                      if f"{name}_log" in result.columns]
    
    if len(sentiment_cols) >= 2:
        # Average of log ratios (gives overall positioning)
        result = result.with_columns([
            pl.mean_horizontal(*[pl.col(c) for c in sentiment_cols])
            .alias("composite_sentiment")
        ])
        
        for window in windows:
            result = result.with_columns([
                pl.col("composite_sentiment").rolling_mean(window)
                .alias(f"composite_sentiment_ma_{window}")
            ])
    
    return result


# ============================================================================
# COMPOSITE FUNCTION
# ============================================================================

def add_all_sentiment_features(
    df: pl.DataFrame,
    windows: List[int] = [6, 24, 72]
) -> pl.DataFrame:
    """
    Add all sentiment-related features in one call.
    
    Args:
        df: DataFrame with OHLCV and sentiment data merged
        windows: Lookback windows
        
    Returns:
        DataFrame with all sentiment features
    """
    result = df.clone()
    
    # Funding rate features
    result = add_funding_rate_features(result, windows=[3, 8, 24])
    
    # Open interest features
    result = add_open_interest_features(result, windows)
    
    # Long/short ratio features
    result = add_long_short_ratio_features(result, windows)
    
    return result
