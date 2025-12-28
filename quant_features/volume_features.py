"""
Volume Features Module

Volume-based features and order flow analysis for quantitative trading.
Includes microstructure features, VWAP, and order flow imbalance.

Features Include:
- Volume metrics: absolute, relative, anomaly detection
- Order flow: buy/sell imbalance, tick rule
- VWAP: volume-weighted average price and deviations
- Trade intensity: count, size distribution

All features use only past data to avoid look-ahead bias.
"""

from typing import List, Optional
import polars as pl
import numpy as np


# ============================================================================
# VOLUME FEATURES
# ============================================================================

def add_volume_features(
    df: pl.DataFrame,
    windows: List[int] = [5, 15, 60, 240]
) -> pl.DataFrame:
    """
    Add volume-based features.
    
    Features for each window:
    - vol_ma_{w}: Moving average of volume
    - vol_ratio_{w}: Current volume vs MA
    - vol_std_{w}: Volume volatility
    - vol_zscore_{w}: Volume z-score for anomaly detection
    
    Args:
        df: DataFrame with OHLCV data (must have 'volume' column)
        windows: List of lookback windows
        
    Returns:
        DataFrame with volume features
    """
    result = df.clone()
    
    for window in windows:
        # Volume moving average
        result = result.with_columns([
            pl.col("volume").rolling_mean(window).alias(f"vol_ma_{window}")
        ])
        
        # Volume standard deviation
        result = result.with_columns([
            pl.col("volume").rolling_std(window).alias(f"vol_std_{window}")
        ])
        
        # Volume ratio (current vs MA)
        result = result.with_columns([
            (pl.col("volume") / (pl.col(f"vol_ma_{window}") + 1e-10))
            .alias(f"vol_ratio_{window}")
        ])
        
        # Volume z-score for anomaly detection
        result = result.with_columns([
            ((pl.col("volume") - pl.col(f"vol_ma_{window}")) 
             / (pl.col(f"vol_std_{window}") + 1e-10))
            .alias(f"vol_zscore_{window}")
        ])
    
    # Quote volume features (dollar volume)
    if "quote_volume" in df.columns:
        for window in windows:
            result = result.with_columns([
                pl.col("quote_volume").rolling_mean(window)
                .alias(f"quote_vol_ma_{window}")
            ])
            
            result = result.with_columns([
                (pl.col("quote_volume") / (pl.col(f"quote_vol_ma_{window}") + 1e-10))
                .alias(f"quote_vol_ratio_{window}")
            ])
    
    # Volume trend
    result = result.with_columns([
        # Volume change rate
        ((pl.col("volume") - pl.col("volume").shift(1)) 
         / (pl.col("volume").shift(1) + 1e-10))
        .alias("vol_change")
    ])
    
    # Price-volume correlation (rolling)
    if "ret_1" not in result.columns:
        result = result.with_columns([
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("ret_1")
        ])
    
    # On-Balance Volume (OBV) change
    obv_change = (
        pl.when(pl.col("close") > pl.col("close").shift(1)).then(pl.col("volume"))
        .when(pl.col("close") < pl.col("close").shift(1)).then(-pl.col("volume"))
        .otherwise(0)
    )
    
    result = result.with_columns([
        obv_change.cum_sum().alias("obv")
    ])
    
    # OBV momentum
    result = result.with_columns([
        (pl.col("obv") - pl.col("obv").shift(20)).alias("obv_momentum_20")
    ])
    
    return result


def add_trade_count_features(
    df: pl.DataFrame,
    windows: List[int] = [5, 15, 60]
) -> pl.DataFrame:
    """
    Add trade count (intensity) features.
    
    Features:
    - count_ma: Average trade count
    - count_ratio: Current vs average
    - avg_trade_size: Average size per trade
    
    Args:
        df: DataFrame with 'count' column (number of trades per bar)
        windows: Lookback windows
        
    Returns:
        DataFrame with trade count features
    """
    result = df.clone()
    
    if "count" not in df.columns:
        return result  # No trade count data available
    
    for window in windows:
        result = result.with_columns([
            pl.col("count").rolling_mean(window).alias(f"count_ma_{window}")
        ])
        
        result = result.with_columns([
            (pl.col("count") / (pl.col(f"count_ma_{window}") + 1e-10))
            .alias(f"count_ratio_{window}")
        ])
    
    # Average trade size
    result = result.with_columns([
        (pl.col("volume") / (pl.col("count") + 1e-10)).alias("avg_trade_size")
    ])
    
    # Trade size trend
    result = result.with_columns([
        pl.col("avg_trade_size").rolling_mean(20).alias("avg_trade_size_ma_20")
    ])
    
    result = result.with_columns([
        (pl.col("avg_trade_size") / (pl.col("avg_trade_size_ma_20") + 1e-10))
        .alias("avg_trade_size_ratio")
    ])
    
    return result


# ============================================================================
# ORDER FLOW IMBALANCE
# ============================================================================

def add_order_flow_imbalance(
    df: pl.DataFrame,
    windows: List[int] = [5, 15, 60]
) -> pl.DataFrame:
    """
    Add order flow imbalance features.
    
    Order flow imbalance measures the difference between buyer-initiated
    and seller-initiated trades.
    
    Features:
    - ofi: Order flow imbalance = taker_buy - taker_sell
    - ofi_ratio: Imbalance as ratio of total volume
    - ofi_ma: Rolling average of imbalance
    - ofi_cumsum: Cumulative imbalance
    
    Args:
        df: DataFrame with taker volume columns
        windows: Lookback windows
        
    Returns:
        DataFrame with order flow features
    """
    result = df.clone()
    
    # Calculate taker sell volume (total - taker buy)
    if "taker_buy_volume" in df.columns:
        result = result.with_columns([
            (pl.col("volume") - pl.col("taker_buy_volume")).alias("taker_sell_volume")
        ])
        
        # Order flow imbalance (signed)
        result = result.with_columns([
            (pl.col("taker_buy_volume") - pl.col("taker_sell_volume")).alias("ofi")
        ])
        
        # Imbalance ratio (normalized by total volume)
        result = result.with_columns([
            (pl.col("ofi") / (pl.col("volume") + 1e-10)).alias("ofi_ratio")
        ])
        
        # Dollar-weighted imbalance
        if "taker_buy_quote_volume" in df.columns:
            result = result.with_columns([
                (pl.col("quote_volume") - pl.col("taker_buy_quote_volume"))
                .alias("taker_sell_quote_volume")
            ])
            
            result = result.with_columns([
                (pl.col("taker_buy_quote_volume") - pl.col("taker_sell_quote_volume"))
                .alias("ofi_dollar")
            ])
        
        # Rolling statistics
        for window in windows:
            result = result.with_columns([
                pl.col("ofi").rolling_mean(window).alias(f"ofi_ma_{window}")
            ])
            
            result = result.with_columns([
                pl.col("ofi").rolling_sum(window).alias(f"ofi_cumsum_{window}")
            ])
            
            result = result.with_columns([
                pl.col("ofi_ratio").rolling_mean(window).alias(f"ofi_ratio_ma_{window}")
            ])
        
        # Buy pressure indicator
        result = result.with_columns([
            (pl.col("taker_buy_volume") / (pl.col("volume") + 1e-10))
            .alias("buy_pressure")
        ])
        
        # Buy pressure momentum
        result = result.with_columns([
            (pl.col("buy_pressure") - pl.col("buy_pressure").rolling_mean(20))
            .alias("buy_pressure_momentum")
        ])
    
    return result


def add_tick_imbalance(
    df: pl.DataFrame,
    windows: List[int] = [20, 50, 100]
) -> pl.DataFrame:
    """
    Add tick imbalance features from bar data.
    
    Tick imbalance uses the direction of price moves to estimate
    buy/sell pressure when direct trade data isn't available.
    
    Features:
    - tick_direction: +1 for up, -1 for down, 0 for unchanged
    - tick_imbalance: Cumulative sum of tick directions
    
    Args:
        df: DataFrame with price data
        windows: Lookback windows
        
    Returns:
        DataFrame with tick imbalance features
    """
    result = df.clone()
    
    # Tick direction from close prices
    result = result.with_columns([
        pl.when(pl.col("close") > pl.col("close").shift(1)).then(1)
        .when(pl.col("close") < pl.col("close").shift(1)).then(-1)
        .otherwise(0)
        .alias("tick_direction")
    ])
    
    for window in windows:
        # Rolling sum of tick directions
        result = result.with_columns([
            pl.col("tick_direction").rolling_sum(window)
            .alias(f"tick_imbalance_{window}")
        ])
        
        # Normalized by window size
        result = result.with_columns([
            (pl.col(f"tick_imbalance_{window}") / window)
            .alias(f"tick_imbalance_norm_{window}")
        ])
    
    return result


# ============================================================================
# VWAP FEATURES
# ============================================================================

def add_vwap_features(
    df: pl.DataFrame,
    windows: List[int] = [15, 60, 240]
) -> pl.DataFrame:
    """
    Add VWAP (Volume Weighted Average Price) features.
    
    VWAP = Σ(Price × Volume) / Σ(Volume)
    
    Features:
    - vwap_{w}: Rolling VWAP for window
    - vwap_dev_{w}: Price deviation from VWAP
    - vwap_dev_pct_{w}: Deviation as percentage
    
    Args:
        df: DataFrame with OHLCV data
        windows: Lookback windows
        
    Returns:
        DataFrame with VWAP features
    """
    result = df.clone()
    
    # Typical price (HLC average)
    result = result.with_columns([
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3)
        .alias("typical_price")
    ])
    
    # Price-volume product
    result = result.with_columns([
        (pl.col("typical_price") * pl.col("volume")).alias("pv_product")
    ])
    
    for window in windows:
        # Rolling VWAP
        result = result.with_columns([
            (pl.col("pv_product").rolling_sum(window) 
             / (pl.col("volume").rolling_sum(window) + 1e-10))
            .alias(f"vwap_{window}")
        ])
        
        # Price deviation from VWAP
        result = result.with_columns([
            (pl.col("close") - pl.col(f"vwap_{window}")).alias(f"vwap_dev_{window}")
        ])
        
        # Percentage deviation
        result = result.with_columns([
            (pl.col(f"vwap_dev_{window}") / pl.col(f"vwap_{window}") * 100)
            .alias(f"vwap_dev_pct_{window}")
        ])
    
    # Clean up intermediate columns
    result = result.drop(["typical_price", "pv_product"])
    
    return result


def add_volume_profile_features(
    df: pl.DataFrame,
    period: int = 20
) -> pl.DataFrame:
    """
    Add simplified volume profile features.
    
    Features:
    - high_vol_zone: Whether current price is in high volume area
    - vol_poc: Price level with highest volume (Point of Control)
    
    Note: This is a simplified version. Full volume profile would
    require tick data aggregation.
    
    Args:
        df: DataFrame with OHLCV data
        period: Lookback period for profile
        
    Returns:
        DataFrame with volume profile features
    """
    result = df.clone()
    
    # Estimate if trading at volume peak using volume-price correlation
    # High volume often clusters around certain price levels
    
    # Volume at current close relative to volume at highs/lows
    result = result.with_columns([
        pl.col("volume").rolling_mean(period).alias("_vol_ma")
    ])
    
    # Volume-weighted price range
    result = result.with_columns([
        ((pl.col("close") - pl.col("low").rolling_min(period))
         / (pl.col("high").rolling_max(period) - pl.col("low").rolling_min(period) + 1e-10))
        .alias("price_position_in_range")
    ])
    
    result = result.drop("_vol_ma")
    
    return result


# ============================================================================
# COMPOSITE FUNCTION
# ============================================================================

def add_all_volume_features(
    df: pl.DataFrame,
    windows: List[int] = [5, 15, 60, 240]
) -> pl.DataFrame:
    """
    Add all volume-based features in one call.
    
    Args:
        df: DataFrame with OHLCV data
        windows: Lookback windows
        
    Returns:
        DataFrame with all volume features
    """
    result = df.clone()
    
    # Base volume features
    result = add_volume_features(result, windows)
    
    # Trade count features (if available)
    result = add_trade_count_features(result, windows[:3])
    
    # Order flow imbalance
    result = add_order_flow_imbalance(result, windows[:3])
    
    # Tick imbalance (always available from price)
    result = add_tick_imbalance(result, windows)
    
    # VWAP features
    result = add_vwap_features(result, windows[:3])
    
    # Volume profile
    result = add_volume_profile_features(result)
    
    return result
