"""
Price Features Module

Technical indicators and price-based features for quantitative analysis.
All features are calculated using only past data to avoid look-ahead bias.

Features Include:
- Returns: Log returns, simple returns at various lags
- Volatility: Rolling std, Parkinson, Garman-Klass estimators
- Momentum: RSI, ROC, momentum oscillators
- Trend: Moving averages, MACD, crossovers
- Patterns: Support/resistance, higher highs, lower lows

All functions accept and return Polars DataFrames.
"""

from typing import List, Optional, Union
import polars as pl
import numpy as np


# ============================================================================
# RETURN FEATURES
# ============================================================================

def add_returns(
    df: pl.DataFrame,
    price_col: str = "close",
    periods: List[int] = [1, 5, 15, 60],
    log_returns: bool = True
) -> pl.DataFrame:
    """
    Add return features at various lookback periods.
    
    IMPORTANT: These are PAST returns (lookback), not forward returns.
    Forward returns for labels are calculated in the labeling module.
    
    Args:
        df: DataFrame with OHLCV data
        price_col: Column to calculate returns from
        periods: List of lookback periods in bars
        log_returns: If True, calculate log returns; else simple returns
        
    Returns:
        DataFrame with new columns: ret_{period} for each period
        
    Example:
        >>> df = add_returns(df, periods=[1, 5, 15])
        >>> # Adds: ret_1, ret_5, ret_15 (lookback returns)
    """
    result = df.clone()
    
    for period in periods:
        col_name = f"ret_{period}"
        
        if log_returns:
            # Log return: ln(P_t / P_{t-period})
            result = result.with_columns([
                (pl.col(price_col).log() - pl.col(price_col).shift(period).log())
                .alias(col_name)
            ])
        else:
            # Simple return: (P_t - P_{t-period}) / P_{t-period}
            result = result.with_columns([
                ((pl.col(price_col) - pl.col(price_col).shift(period)) 
                 / pl.col(price_col).shift(period))
                .alias(col_name)
            ])
    
    return result


def add_intrabar_returns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add intra-bar return features.
    
    Features:
    - open_to_close: Return from open to close within bar
    - high_to_low: Full range as percentage
    - open_to_high: Maximum upside from open
    - open_to_low: Maximum downside from open
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with intrabar return features
    """
    return df.with_columns([
        # Open to close (bar direction)
        ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("bar_return"),
        
        # High to low range
        ((pl.col("high") - pl.col("low")) / pl.col("open")).alias("bar_range"),
        
        # Open to high (max upside)
        ((pl.col("high") - pl.col("open")) / pl.col("open")).alias("bar_up_range"),
        
        # Open to low (max downside)
        ((pl.col("open") - pl.col("low")) / pl.col("open")).alias("bar_down_range"),
        
        # Upper shadow relative to body
        ((pl.col("high") - pl.col("close").clip(lower_bound=pl.col("open"))) 
         / (pl.col("high") - pl.col("low") + 1e-10)).alias("upper_shadow_ratio"),
        
        # Lower shadow relative to body
        ((pl.col("close").clip(upper_bound=pl.col("open")) - pl.col("low"))
         / (pl.col("high") - pl.col("low") + 1e-10)).alias("lower_shadow_ratio"),
    ])


# ============================================================================
# VOLATILITY FEATURES
# ============================================================================

def add_volatility_features(
    df: pl.DataFrame,
    windows: List[int] = [5, 15, 60, 240],
    price_col: str = "close"
) -> pl.DataFrame:
    """
    Add volatility estimators.
    
    Features for each window:
    - vol_std_{w}: Rolling standard deviation of returns
    - vol_parkinson_{w}: Parkinson estimator (uses H/L)
    - vol_garman_klass_{w}: Garman-Klass estimator (uses OHLC)
    - vol_ratio_{w}: Current volatility vs longer-term average
    
    Parkinson Estimator:
        σ² = (1/4ln(2)) * mean(ln(H/L)²)
        More efficient than close-to-close for continuous prices
        
    Garman-Klass Estimator:
        σ² = 0.5 * (ln(H/L))² - (2ln2 - 1) * (ln(C/O))²
        Most efficient for OHLC data
    
    Args:
        df: DataFrame with OHLCV data
        windows: List of lookback windows
        price_col: Price column for return-based volatility
        
    Returns:
        DataFrame with volatility features
    """
    result = df.clone()
    
    # First ensure we have returns
    if "ret_1" not in result.columns:
        result = add_returns(result, price_col, periods=[1])
    
    for window in windows:
        # Standard deviation of returns (realized volatility)
        result = result.with_columns([
            pl.col("ret_1").rolling_std(window).alias(f"vol_std_{window}")
        ])
        
        # Parkinson volatility estimator
        # Uses high-low range, more efficient than close-to-close
        result = result.with_columns([
            ((pl.col("high") / pl.col("low")).log().pow(2)
             .rolling_mean(window)
             / (4 * np.log(2)))
            .sqrt()
            .alias(f"vol_parkinson_{window}")
        ])
        
        # Garman-Klass volatility estimator
        # Uses OHLC, most efficient for daily/hourly data
        hl_term = 0.5 * (pl.col("high") / pl.col("low")).log().pow(2)
        co_term = (2 * np.log(2) - 1) * (pl.col("close") / pl.col("open")).log().pow(2)
        
        result = result.with_columns([
            (hl_term - co_term)
            .rolling_mean(window)
            .sqrt()
            .alias(f"vol_gk_{window}")
        ])
    
    # Volatility ratios (short-term vs long-term)
    if len(windows) >= 2:
        short_vol = f"vol_std_{windows[0]}"
        long_vol = f"vol_std_{windows[-1]}"
        result = result.with_columns([
            (pl.col(short_vol) / (pl.col(long_vol) + 1e-10)).alias("vol_ratio")
        ])
    
    return result


def add_atr(
    df: pl.DataFrame,
    windows: List[int] = [14, 21, 50]
) -> pl.DataFrame:
    """
    Add Average True Range (ATR) features.
    
    True Range = max(H-L, |H-C_prev|, |L-C_prev|)
    ATR = EMA or SMA of True Range
    
    Args:
        df: DataFrame with OHLCV data
        windows: List of ATR periods
        
    Returns:
        DataFrame with ATR columns
    """
    result = df.clone()
    
    # Calculate True Range
    prev_close = pl.col("close").shift(1)
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs()
    )
    
    result = result.with_columns([true_range.alias("true_range")])
    
    for window in windows:
        # Simple moving average ATR
        result = result.with_columns([
            pl.col("true_range").rolling_mean(window).alias(f"atr_{window}")
        ])
        
        # ATR as percentage of close (normalized)
        result = result.with_columns([
            (pl.col(f"atr_{window}") / pl.col("close") * 100).alias(f"atr_pct_{window}")
        ])
    
    return result


# ============================================================================
# MOMENTUM FEATURES
# ============================================================================

def add_momentum_features(
    df: pl.DataFrame,
    price_col: str = "close"
) -> pl.DataFrame:
    """
    Add momentum indicators.
    
    Features:
    - RSI: Relative Strength Index at various periods
    - ROC: Rate of Change
    - Momentum: Price difference
    - Williams %R: Normalized position in range
    - Stochastic: %K and %D oscillators
    
    Args:
        df: DataFrame with OHLCV data
        price_col: Price column to use
        
    Returns:
        DataFrame with momentum features
    """
    result = df.clone()
    
    # RSI at various periods
    for period in [7, 14, 21]:
        result = _add_rsi(result, period, price_col)
    
    # Rate of Change (ROC)
    for period in [5, 10, 20]:
        result = result.with_columns([
            ((pl.col(price_col) - pl.col(price_col).shift(period)) 
             / pl.col(price_col).shift(period) * 100)
            .alias(f"roc_{period}")
        ])
    
    # Williams %R
    for period in [14, 21]:
        result = result.with_columns([
            ((pl.col("high").rolling_max(period) - pl.col(price_col))
             / (pl.col("high").rolling_max(period) - pl.col("low").rolling_min(period) + 1e-10)
             * -100)
            .alias(f"williams_r_{period}")
        ])
    
    # Stochastic %K and %D
    for period in [14]:
        lowest_low = pl.col("low").rolling_min(period)
        highest_high = pl.col("high").rolling_max(period)
        
        result = result.with_columns([
            ((pl.col(price_col) - lowest_low) 
             / (highest_high - lowest_low + 1e-10) * 100)
            .alias(f"stoch_k_{period}")
        ])
        
        # %D is SMA of %K
        result = result.with_columns([
            pl.col(f"stoch_k_{period}").rolling_mean(3).alias(f"stoch_d_{period}")
        ])
    
    return result


def _add_rsi(
    df: pl.DataFrame,
    period: int,
    price_col: str = "close"
) -> pl.DataFrame:
    """
    Calculate RSI (Relative Strength Index).
    
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss (over period)
    
    Uses Wilder's smoothing method (EMA-like).
    """
    # Calculate price changes
    delta = pl.col(price_col) - pl.col(price_col).shift(1)
    
    # Separate gains and losses
    gains = delta.clip(lower_bound=0)
    losses = (-delta).clip(lower_bound=0)
    
    # Use SMA for initial RS, then EWM (simplified to rolling mean here for stability)
    avg_gain = gains.rolling_mean(period)
    avg_loss = losses.rolling_mean(period)
    
    # Calculate RSI
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return df.with_columns([rsi.alias(f"rsi_{period}")])


# ============================================================================
# TREND FEATURES
# ============================================================================

def add_trend_features(
    df: pl.DataFrame,
    price_col: str = "close"
) -> pl.DataFrame:
    """
    Add trend-following indicators.
    
    Features:
    - SMA/EMA at various periods
    - Price position relative to MAs
    - MACD and signal line
    - ADX (Average Directional Index)
    - Trend strength measures
    
    Args:
        df: DataFrame with OHLCV data
        price_col: Price column to use
        
    Returns:
        DataFrame with trend features
    """
    result = df.clone()
    
    # Moving Averages
    ma_periods = [5, 10, 20, 50, 100, 200]
    for period in ma_periods:
        # Simple Moving Average
        result = result.with_columns([
            pl.col(price_col).rolling_mean(period).alias(f"sma_{period}")
        ])
        
        # Exponential Moving Average
        result = result.with_columns([
            pl.col(price_col).ewm_mean(span=period).alias(f"ema_{period}")
        ])
    
    # Price position relative to MAs (normalized)
    for period in ma_periods:
        result = result.with_columns([
            ((pl.col(price_col) - pl.col(f"sma_{period}")) 
             / pl.col(f"sma_{period}") * 100)
            .alias(f"price_vs_sma_{period}")
        ])
    
    # MACD
    result = result.with_columns([
        (pl.col("ema_12") if "ema_12" in result.columns 
         else pl.col(price_col).ewm_mean(span=12)).alias("ema_12_temp"),
        (pl.col("ema_26") if "ema_26" in result.columns
         else pl.col(price_col).ewm_mean(span=26)).alias("ema_26_temp")
    ])
    
    result = result.with_columns([
        pl.col(price_col).ewm_mean(span=12).alias("_ema_12"),
        pl.col(price_col).ewm_mean(span=26).alias("_ema_26")
    ])
    
    result = result.with_columns([
        (pl.col("_ema_12") - pl.col("_ema_26")).alias("macd")
    ])
    
    result = result.with_columns([
        pl.col("macd").ewm_mean(span=9).alias("macd_signal")
    ])
    
    result = result.with_columns([
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist")
    ])
    
    # Clean up temp columns
    result = result.drop(["_ema_12", "_ema_26", "ema_12_temp", "ema_26_temp"])
    
    # MA Crossover signals
    result = result.with_columns([
        # Golden cross: short MA above long MA
        (pl.col("sma_20") > pl.col("sma_50")).cast(pl.Int8).alias("ma_cross_20_50"),
        (pl.col("sma_50") > pl.col("sma_200")).cast(pl.Int8).alias("ma_cross_50_200"),
    ])
    
    # Trend strength: slope of MA
    for period in [20, 50]:
        ma_col = f"sma_{period}"
        result = result.with_columns([
            ((pl.col(ma_col) - pl.col(ma_col).shift(5)) / pl.col(ma_col).shift(5) * 100)
            .alias(f"ma_slope_{period}")
        ])
    
    return result


def add_adx(
    df: pl.DataFrame,
    period: int = 14
) -> pl.DataFrame:
    """
    Add Average Directional Index (ADX) for trend strength.
    
    Components:
    - +DI: Positive Directional Indicator
    - -DI: Negative Directional Indicator
    - ADX: Smoothed average of DX
    
    Interpretation:
    - ADX > 25: Strong trend
    - ADX < 20: Weak/no trend
    - +DI > -DI: Uptrend
    - -DI > +DI: Downtrend
    """
    result = df.clone()
    
    # Calculate +DM and -DM
    high_diff = pl.col("high") - pl.col("high").shift(1)
    low_diff = pl.col("low").shift(1) - pl.col("low")
    
    # +DM: positive movement
    plus_dm = pl.when(high_diff > low_diff).then(high_diff.clip(lower_bound=0)).otherwise(0)
    
    # -DM: negative movement
    minus_dm = pl.when(low_diff > high_diff).then(low_diff.clip(lower_bound=0)).otherwise(0)
    
    # True Range for ATR
    if "true_range" not in result.columns:
        prev_close = pl.col("close").shift(1)
        tr = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - prev_close).abs(),
            (pl.col("low") - prev_close).abs()
        )
        result = result.with_columns([tr.alias("true_range")])
    
    # Smooth with rolling sum (Wilder's method approximated)
    result = result.with_columns([
        plus_dm.alias("_plus_dm"),
        minus_dm.alias("_minus_dm")
    ])
    
    result = result.with_columns([
        pl.col("_plus_dm").rolling_sum(period).alias("_plus_dm_sum"),
        pl.col("_minus_dm").rolling_sum(period).alias("_minus_dm_sum"),
        pl.col("true_range").rolling_sum(period).alias("_tr_sum")
    ])
    
    # Calculate +DI and -DI
    result = result.with_columns([
        (pl.col("_plus_dm_sum") / (pl.col("_tr_sum") + 1e-10) * 100).alias("plus_di"),
        (pl.col("_minus_dm_sum") / (pl.col("_tr_sum") + 1e-10) * 100).alias("minus_di")
    ])
    
    # Calculate DX
    result = result.with_columns([
        ((pl.col("plus_di") - pl.col("minus_di")).abs() 
         / (pl.col("plus_di") + pl.col("minus_di") + 1e-10) * 100)
        .alias("dx")
    ])
    
    # ADX is smoothed DX
    result = result.with_columns([
        pl.col("dx").rolling_mean(period).alias("adx")
    ])
    
    # Clean up temp columns
    temp_cols = ["_plus_dm", "_minus_dm", "_plus_dm_sum", "_minus_dm_sum", "_tr_sum"]
    result = result.drop([c for c in temp_cols if c in result.columns])
    
    return result


# ============================================================================
# PATTERN FEATURES
# ============================================================================

def add_price_patterns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add price pattern recognition features.
    
    Features:
    - Higher highs / Lower lows detection
    - Support/resistance proximity
    - Candlestick pattern signals
    - Price acceleration
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with pattern features
    """
    result = df.clone()
    
    # Higher highs and lower lows (for various lookbacks)
    for lookback in [5, 10, 20]:
        # Count of higher highs in last N bars
        result = result.with_columns([
            (pl.col("high") > pl.col("high").shift(1))
            .cast(pl.Int8)
            .rolling_sum(lookback)
            .alias(f"higher_highs_{lookback}")
        ])
        
        # Count of lower lows in last N bars
        result = result.with_columns([
            (pl.col("low") < pl.col("low").shift(1))
            .cast(pl.Int8)
            .rolling_sum(lookback)
            .alias(f"lower_lows_{lookback}")
        ])
    
    # Price relative to recent highs/lows
    for period in [20, 50]:
        result = result.with_columns([
            # Distance to recent high (potential resistance)
            ((pl.col("high").rolling_max(period) - pl.col("close"))
             / pl.col("close") * 100)
            .alias(f"dist_to_high_{period}"),
            
            # Distance to recent low (potential support)
            ((pl.col("close") - pl.col("low").rolling_min(period))
             / pl.col("close") * 100)
            .alias(f"dist_to_low_{period}")
        ])
    
    # Breakout signals
    result = result.with_columns([
        # New 20-period high
        (pl.col("close") >= pl.col("high").shift(1).rolling_max(20))
        .cast(pl.Int8)
        .alias("breakout_high_20"),
        
        # New 20-period low
        (pl.col("close") <= pl.col("low").shift(1).rolling_min(20))
        .cast(pl.Int8)
        .alias("breakout_low_20"),
    ])
    
    # Price acceleration (second derivative)
    if "ret_1" not in result.columns:
        result = add_returns(result, periods=[1])
    
    result = result.with_columns([
        (pl.col("ret_1") - pl.col("ret_1").shift(1)).alias("price_acceleration")
    ])
    
    # Candlestick body size
    result = result.with_columns([
        ((pl.col("close") - pl.col("open")).abs() 
         / (pl.col("high") - pl.col("low") + 1e-10))
        .alias("body_ratio")
    ])
    
    # Doji detection (small body)
    result = result.with_columns([
        (pl.col("body_ratio") < 0.1).cast(pl.Int8).alias("is_doji")
    ])
    
    return result


# ============================================================================
# COMPOSITE FEATURE FUNCTIONS
# ============================================================================

def add_all_price_features(
    df: pl.DataFrame,
    price_col: str = "close"
) -> pl.DataFrame:
    """
    Add all price-based features in one call.
    
    This is a convenience function that applies all price feature
    functions in the correct order.
    
    Args:
        df: DataFrame with OHLCV data
        price_col: Price column to use
        
    Returns:
        DataFrame with all price features added
    """
    result = df.clone()
    
    # Returns (foundation for other features)
    result = add_returns(result, price_col, periods=[1, 5, 15, 60, 240])
    result = add_intrabar_returns(result)
    
    # Volatility
    result = add_volatility_features(result, windows=[5, 15, 60, 240], price_col=price_col)
    result = add_atr(result, windows=[14, 21, 50])
    
    # Momentum
    result = add_momentum_features(result, price_col)
    
    # Trend
    result = add_trend_features(result, price_col)
    result = add_adx(result, period=14)
    
    # Patterns
    result = add_price_patterns(result)
    
    return result
