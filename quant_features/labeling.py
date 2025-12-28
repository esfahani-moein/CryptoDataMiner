"""
Labeling Module

Functions to create target labels for machine learning models.
Implements various labeling schemes used in quantitative finance.

Label Types:
1. Forward Returns: Simple future returns at various horizons
2. Triple Barrier: Stop-loss, take-profit, time-based exit
3. Trend Labels: Direction classification (up/down/sideways)
4. Volatility Labels: Regime classification (breakout/normal/squeeze)
5. Meta-Labels: Bet sizing and side prediction

CRITICAL: These functions intentionally use future data for labels.
The labels are what we're trying to predict. Look-ahead bias prevention
is handled during train/test split and feature engineering.
"""

from typing import List, Optional, Tuple, Union
import polars as pl
import numpy as np


# ============================================================================
# FORWARD RETURNS (TARGET VARIABLE)
# ============================================================================

def add_forward_returns(
    df: pl.DataFrame,
    periods: List[int] = [5, 15, 60, 240],
    price_col: str = "close",
    log_returns: bool = True
) -> pl.DataFrame:
    """
    Calculate forward returns as prediction targets.
    
    NOTE: These are FORWARD-LOOKING and should only be used as labels,
    not as features. Forward returns use FUTURE prices.
    
    Formula:
        Forward Return = P_{t+period} / P_t - 1  (simple)
        Forward Return = ln(P_{t+period} / P_t)  (log)
    
    Args:
        df: DataFrame with price data
        periods: List of forward periods (in bars)
        price_col: Price column to use
        log_returns: Use log returns if True
        
    Returns:
        DataFrame with forward return columns: fwd_ret_{period}
        
    Example:
        >>> df = add_forward_returns(df, periods=[5, 15, 60])
        >>> # Adds: fwd_ret_5, fwd_ret_15, fwd_ret_60
    """
    result = df.clone()
    
    for period in periods:
        col_name = f"fwd_ret_{period}"
        
        if log_returns:
            # Log forward return
            result = result.with_columns([
                (pl.col(price_col).shift(-period).log() - pl.col(price_col).log())
                .alias(col_name)
            ])
        else:
            # Simple forward return
            result = result.with_columns([
                ((pl.col(price_col).shift(-period) - pl.col(price_col)) 
                 / pl.col(price_col))
                .alias(col_name)
            ])
    
    return result


def add_forward_return_classes(
    df: pl.DataFrame,
    periods: List[int] = [5, 15, 60],
    threshold: float = 0.001,  # 0.1% = 10 bps
    price_col: str = "close"
) -> pl.DataFrame:
    """
    Create classification labels from forward returns.
    
    Classes:
    - 1 (Long): Forward return > threshold
    - 0 (Neutral): |Forward return| <= threshold
    - -1 (Short): Forward return < -threshold
    
    Args:
        df: DataFrame with price data
        periods: Forward periods to classify
        threshold: Return threshold for classification
        price_col: Price column to use
        
    Returns:
        DataFrame with classification columns: fwd_class_{period}
    """
    result = df.clone()
    
    # First ensure we have forward returns
    for period in periods:
        fwd_col = f"fwd_ret_{period}"
        class_col = f"fwd_class_{period}"
        
        if fwd_col not in result.columns:
            result = result.with_columns([
                (pl.col(price_col).shift(-period).log() - pl.col(price_col).log())
                .alias(fwd_col)
            ])
        
        # Create classes
        result = result.with_columns([
            pl.when(pl.col(fwd_col) > threshold).then(1)
            .when(pl.col(fwd_col) < -threshold).then(-1)
            .otherwise(0)
            .alias(class_col)
        ])
    
    return result


# ============================================================================
# TRIPLE BARRIER LABELS
# ============================================================================

def add_triple_barrier_labels(
    df: pl.DataFrame,
    max_holding_period: int = 60,  # Max bars to hold
    profit_taking: float = 0.01,   # 1% take profit
    stop_loss: float = 0.01,       # 1% stop loss
    price_col: str = "close",
    use_vertical_barrier: bool = True
) -> pl.DataFrame:
    """
    Implement Triple Barrier Labeling Method.
    
    Three barriers that can be hit:
    1. Upper (Take Profit): Price rises by profit_taking %
    2. Lower (Stop Loss): Price falls by stop_loss %
    3. Vertical (Time): Max holding period reached
    
    Labels:
    - 1: Upper barrier hit first (profitable long)
    - -1: Lower barrier hit first (unprofitable long / profitable short)
    - 0: Vertical barrier hit (no clear direction)
    
    Also returns:
    - barrier_touch_time: When barrier was touched
    - barrier_return: Actual return at barrier touch
    
    Args:
        df: DataFrame with OHLCV data
        max_holding_period: Maximum bars until vertical barrier
        profit_taking: Profit target as decimal (0.01 = 1%)
        stop_loss: Stop loss as decimal (0.01 = 1%)
        price_col: Price column for barrier calculation
        use_vertical_barrier: Whether to use time limit
        
    Returns:
        DataFrame with triple barrier labels
        
    Note:
        This is computationally expensive for large datasets.
        Consider using vectorized approximation for initial screening.
    """
    result = df.clone()
    n_rows = len(result)
    
    # Initialize arrays
    labels = np.zeros(n_rows, dtype=np.int8)
    touch_bars = np.zeros(n_rows, dtype=np.int32)
    returns_at_touch = np.zeros(n_rows, dtype=np.float64)
    
    # Get price array
    prices = result[price_col].to_numpy()
    
    # If we have high/low, use them for more accurate barrier detection
    if "high" in result.columns and "low" in result.columns:
        highs = result["high"].to_numpy()
        lows = result["low"].to_numpy()
        use_hl = True
    else:
        use_hl = False
    
    # Process each bar
    for i in range(n_rows):
        entry_price = prices[i]
        upper_barrier = entry_price * (1 + profit_taking)
        lower_barrier = entry_price * (1 - stop_loss)
        
        # Search forward for barrier touch
        for j in range(1, min(max_holding_period + 1, n_rows - i)):
            idx = i + j
            
            if use_hl:
                # Check if high touched upper barrier
                upper_touched = highs[idx] >= upper_barrier
                # Check if low touched lower barrier
                lower_touched = lows[idx] <= lower_barrier
            else:
                upper_touched = prices[idx] >= upper_barrier
                lower_touched = prices[idx] <= lower_barrier
            
            if upper_touched and lower_touched:
                # Both touched in same bar - use close to determine
                if prices[idx] >= entry_price:
                    labels[i] = 1
                else:
                    labels[i] = -1
                touch_bars[i] = j
                returns_at_touch[i] = (prices[idx] - entry_price) / entry_price
                break
            elif upper_touched:
                labels[i] = 1
                touch_bars[i] = j
                returns_at_touch[i] = profit_taking
                break
            elif lower_touched:
                labels[i] = -1
                touch_bars[i] = j
                returns_at_touch[i] = -stop_loss
                break
        else:
            # Vertical barrier hit (time ran out)
            if use_vertical_barrier:
                labels[i] = 0
                touch_bars[i] = max_holding_period
                if i + max_holding_period < n_rows:
                    returns_at_touch[i] = (prices[i + max_holding_period] - entry_price) / entry_price
                else:
                    returns_at_touch[i] = np.nan
    
    # Add to DataFrame
    result = result.with_columns([
        pl.Series("tb_label", labels),
        pl.Series("tb_touch_bars", touch_bars),
        pl.Series("tb_return", returns_at_touch)
    ])
    
    return result


def add_triple_barrier_vectorized(
    df: pl.DataFrame,
    max_holding_period: int = 60,
    profit_taking: float = 0.01,
    stop_loss: float = 0.01,
    price_col: str = "close"
) -> pl.DataFrame:
    """
    Vectorized approximation of triple barrier labels.
    
    This is faster than the iterative version but less accurate
    because it only checks at specific horizons.
    
    Args:
        df: DataFrame with price data
        max_holding_period: Maximum holding period
        profit_taking: Take profit level
        stop_loss: Stop loss level
        price_col: Price column
        
    Returns:
        DataFrame with approximate triple barrier labels
    """
    result = df.clone()
    
    # Calculate forward max and min over holding period
    result = result.with_columns([
        # Maximum high reached in next N bars
        pl.col("high").shift(-1).rolling_max(max_holding_period)
        .alias("_fwd_max_high"),
        
        # Minimum low reached in next N bars
        pl.col("low").shift(-1).rolling_min(max_holding_period)
        .alias("_fwd_min_low"),
        
        # Return at vertical barrier
        (pl.col(price_col).shift(-max_holding_period) / pl.col(price_col) - 1)
        .alias("_fwd_ret_vertical")
    ])
    
    # Calculate barrier levels
    result = result.with_columns([
        (pl.col(price_col) * (1 + profit_taking)).alias("_upper_barrier"),
        (pl.col(price_col) * (1 - stop_loss)).alias("_lower_barrier"),
    ])
    
    # Check which barrier was hit
    result = result.with_columns([
        (pl.col("_fwd_max_high") >= pl.col("_upper_barrier")).alias("_hit_upper"),
        (pl.col("_fwd_min_low") <= pl.col("_lower_barrier")).alias("_hit_lower"),
    ])
    
    # Assign labels (simplified - doesn't capture exact order of barrier hits)
    result = result.with_columns([
        pl.when(pl.col("_hit_upper") & ~pl.col("_hit_lower")).then(1)
        .when(pl.col("_hit_lower") & ~pl.col("_hit_upper")).then(-1)
        .when(pl.col("_hit_upper") & pl.col("_hit_lower")).then(
            # Both hit - use final return direction
            pl.when(pl.col("_fwd_ret_vertical") > 0).then(1).otherwise(-1)
        )
        .otherwise(0)  # Neither hit - vertical barrier
        .alias("tb_label_approx")
    ])
    
    # Clean up temp columns
    temp_cols = ["_fwd_max_high", "_fwd_min_low", "_fwd_ret_vertical",
                 "_upper_barrier", "_lower_barrier", "_hit_upper", "_hit_lower"]
    result = result.drop(temp_cols)
    
    return result


# ============================================================================
# TREND LABELS
# ============================================================================

def add_trend_labels(
    df: pl.DataFrame,
    window: int = 20,
    up_threshold: float = 0.02,    # 2% for uptrend
    down_threshold: float = -0.02  # -2% for downtrend
) -> pl.DataFrame:
    """
    Classify price trend direction.
    
    Uses forward-looking price change over window to determine trend.
    
    Labels:
    - 1: Uptrend (price rises by > up_threshold)
    - -1: Downtrend (price falls by > |down_threshold|)
    - 0: Sideways (price change within thresholds)
    
    Args:
        df: DataFrame with price data
        window: Forward window for trend determination
        up_threshold: Minimum return for uptrend
        down_threshold: Maximum return for downtrend (negative)
        
    Returns:
        DataFrame with trend_label column
    """
    result = df.clone()
    
    # Calculate forward return over window
    fwd_ret_col = f"_fwd_ret_{window}"
    result = result.with_columns([
        (pl.col("close").shift(-window) / pl.col("close") - 1).alias(fwd_ret_col)
    ])
    
    # Classify trend
    result = result.with_columns([
        pl.when(pl.col(fwd_ret_col) > up_threshold).then(1)
        .when(pl.col(fwd_ret_col) < down_threshold).then(-1)
        .otherwise(0)
        .alias("trend_label")
    ])
    
    # Also add trend strength
    result = result.with_columns([
        pl.col(fwd_ret_col).abs().alias("trend_strength")
    ])
    
    # Clean up temp column
    result = result.drop(fwd_ret_col)
    
    return result


def add_regime_labels(
    df: pl.DataFrame,
    vol_window: int = 20,
    vol_lookback: int = 100,
    high_vol_percentile: float = 0.75,
    low_vol_percentile: float = 0.25
) -> pl.DataFrame:
    """
    Add volatility regime labels.
    
    Labels based on where current volatility stands relative to history:
    - 2: Breakout (very high volatility)
    - 1: Elevated (high volatility)
    - 0: Normal
    - -1: Low volatility
    - -2: Squeeze (very low volatility)
    
    Args:
        df: DataFrame with OHLCV data
        vol_window: Window for current volatility calculation
        vol_lookback: Lookback for percentile calculation
        high_vol_percentile: Threshold for high vol (default 75th)
        low_vol_percentile: Threshold for low vol (default 25th)
        
    Returns:
        DataFrame with regime_label column
    """
    result = df.clone()
    
    # Calculate realized volatility (standard deviation of returns)
    if "ret_1" not in result.columns:
        result = result.with_columns([
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("_ret_1")
        ])
        ret_col = "_ret_1"
    else:
        ret_col = "ret_1"
    
    result = result.with_columns([
        pl.col(ret_col).rolling_std(vol_window).alias("_current_vol")
    ])
    
    # Calculate rolling percentile rank
    result = result.with_columns([
        (pl.col("_current_vol").rolling_rank(vol_lookback) / vol_lookback)
        .alias("_vol_percentile")
    ])
    
    # Classify regime
    result = result.with_columns([
        pl.when(pl.col("_vol_percentile") >= 0.9).then(2)  # Breakout
        .when(pl.col("_vol_percentile") >= high_vol_percentile).then(1)  # Elevated
        .when(pl.col("_vol_percentile") <= 0.1).then(-2)  # Squeeze
        .when(pl.col("_vol_percentile") <= low_vol_percentile).then(-1)  # Low
        .otherwise(0)  # Normal
        .alias("regime_label")
    ])
    
    # Also keep volatility percentile as a feature
    result = result.rename({"_vol_percentile": "vol_percentile"})
    
    # Clean up
    temp_cols = ["_ret_1", "_current_vol"] if "_ret_1" in result.columns else ["_current_vol"]
    result = result.drop([c for c in temp_cols if c in result.columns])
    
    return result


# ============================================================================
# VOLATILITY LABELS
# ============================================================================

def add_volatility_labels(
    df: pl.DataFrame,
    forward_window: int = 20,
    lookback_window: int = 100
) -> pl.DataFrame:
    """
    Label periods by forward volatility (what volatility will be).
    
    This labels based on FUTURE volatility, useful for volatility
    prediction models.
    
    Labels:
    - 2: High volatility ahead (> 75th percentile)
    - 1: Elevated volatility (50-75th percentile)
    - 0: Normal volatility (25-50th percentile)
    - -1: Low volatility (< 25th percentile)
    
    Args:
        df: DataFrame with price data
        forward_window: Window for future volatility
        lookback_window: Lookback for percentile calculation
        
    Returns:
        DataFrame with volatility_label column
    """
    result = df.clone()
    
    # Calculate returns
    if "ret_1" not in result.columns:
        result = result.with_columns([
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("_ret_1")
        ])
        ret_col = "_ret_1"
    else:
        ret_col = "ret_1"
    
    # Forward volatility (using shift to look ahead)
    result = result.with_columns([
        pl.col(ret_col).shift(-1).rolling_std(forward_window).alias("_fwd_vol")
    ])
    
    # Historical volatility for comparison
    result = result.with_columns([
        pl.col(ret_col).rolling_std(lookback_window).alias("_hist_vol")
    ])
    
    # Ratio of forward to historical
    result = result.with_columns([
        (pl.col("_fwd_vol") / (pl.col("_hist_vol") + 1e-10)).alias("_vol_ratio")
    ])
    
    # Percentile of forward volatility
    result = result.with_columns([
        (pl.col("_fwd_vol").rolling_rank(lookback_window) / lookback_window)
        .alias("_fwd_vol_percentile")
    ])
    
    # Classify
    result = result.with_columns([
        pl.when(pl.col("_fwd_vol_percentile") >= 0.75).then(2)
        .when(pl.col("_fwd_vol_percentile") >= 0.50).then(1)
        .when(pl.col("_fwd_vol_percentile") >= 0.25).then(0)
        .otherwise(-1)
        .alias("volatility_label")
    ])
    
    # Keep forward vol as target
    result = result.rename({"_fwd_vol": "fwd_volatility"})
    
    # Clean up
    temp_cols = ["_ret_1", "_hist_vol", "_vol_ratio", "_fwd_vol_percentile"]
    result = result.drop([c for c in temp_cols if c in result.columns])
    
    return result


# ============================================================================
# META-LABELS
# ============================================================================

def add_meta_labels(
    df: pl.DataFrame,
    base_label_col: str = "fwd_class_60",
    confidence_threshold: float = 0.6
) -> pl.DataFrame:
    """
    Create meta-labels for bet sizing.
    
    Meta-labeling approach (from AFML):
    1. First, have a base model that predicts direction (side)
    2. Meta-label predicts whether the base prediction is correct
    3. Use meta-label confidence for bet sizing
    
    For this implementation, we use triple barrier labels as "truth"
    and create meta-labels showing prediction quality.
    
    Labels:
    - 1: Correct prediction (profitable trade)
    - 0: Incorrect prediction (unprofitable trade)
    
    Args:
        df: DataFrame with base labels and triple barrier labels
        base_label_col: Column with base model predictions
        confidence_threshold: Minimum confidence for meta-label
        
    Returns:
        DataFrame with meta_label column
    """
    result = df.clone()
    
    if base_label_col not in result.columns or "tb_label" not in result.columns:
        # Can't create meta-labels without base labels
        return result
    
    # Meta-label: was the base prediction correct?
    result = result.with_columns([
        # Correct if signs match and tb_label != 0
        (
            (pl.col(base_label_col) * pl.col("tb_label") > 0)  # Same sign
            | ((pl.col(base_label_col) == 0) & (pl.col("tb_label") == 0))  # Both neutral
        ).cast(pl.Int8)
        .alias("meta_label")
    ])
    
    # Bet size based on historical accuracy
    # Rolling accuracy of base model
    result = result.with_columns([
        pl.col("meta_label").rolling_mean(100).alias("rolling_accuracy")
    ])
    
    # Suggested bet size (higher accuracy = larger bet)
    result = result.with_columns([
        pl.when(pl.col("rolling_accuracy") > confidence_threshold)
        .then(pl.col("rolling_accuracy"))
        .otherwise(0.0)
        .alias("bet_size")
    ])
    
    return result


def add_sample_weights(
    df: pl.DataFrame,
    return_col: str = "tb_return",
    decay_factor: float = 0.99,
    uniqueness_window: int = 20
) -> pl.DataFrame:
    """
    Calculate sample weights for training.
    
    Sample weights based on:
    1. Return magnitude: Larger moves are more important
    2. Time decay: Recent samples more relevant
    3. Uniqueness: Reduce weight of overlapping samples
    
    Args:
        df: DataFrame with return data
        return_col: Column with returns
        decay_factor: Time decay factor (0.99 = slow decay)
        uniqueness_window: Window for uniqueness calculation
        
    Returns:
        DataFrame with sample_weight column
    """
    result = df.clone()
    n_rows = len(result)
    
    # Return-based weight (larger absolute returns = higher weight)
    if return_col in result.columns:
        result = result.with_columns([
            (pl.col(return_col).abs() + 0.1).alias("_ret_weight")
        ])
    else:
        result = result.with_columns([
            pl.lit(1.0).alias("_ret_weight")
        ])
    
    # Time decay weight (more recent = higher weight)
    time_weights = np.power(decay_factor, np.arange(n_rows)[::-1])
    result = result.with_columns([
        pl.Series("_time_weight", time_weights)
    ])
    
    # Uniqueness weight (based on average uniqueness of features)
    # Simplified: use inverse of sample overlap
    result = result.with_columns([
        pl.lit(1.0 / uniqueness_window).alias("_unique_weight")
    ])
    
    # Combine weights
    result = result.with_columns([
        (pl.col("_ret_weight") * pl.col("_time_weight") * pl.col("_unique_weight"))
        .alias("sample_weight")
    ])
    
    # Normalize weights
    total_weight = result["sample_weight"].sum()
    result = result.with_columns([
        (pl.col("sample_weight") / total_weight * n_rows).alias("sample_weight")
    ])
    
    # Clean up
    result = result.drop(["_ret_weight", "_time_weight", "_unique_weight"])
    
    return result


# ============================================================================
# COMPOSITE FUNCTION
# ============================================================================

def add_all_labels(
    df: pl.DataFrame,
    forward_periods: List[int] = [5, 15, 60, 240],
    tb_max_period: int = 60,
    tb_profit: float = 0.01,
    tb_stop: float = 0.01
) -> pl.DataFrame:
    """
    Add all label types in one call.
    
    Args:
        df: DataFrame with OHLCV data
        forward_periods: Periods for forward returns
        tb_max_period: Max period for triple barrier
        tb_profit: Take profit level for triple barrier
        tb_stop: Stop loss level for triple barrier
        
    Returns:
        DataFrame with all labels added
    """
    result = df.clone()
    
    # Forward returns and classes
    result = add_forward_returns(result, forward_periods)
    result = add_forward_return_classes(result, forward_periods[:3])
    
    # Triple barrier (vectorized for speed)
    result = add_triple_barrier_vectorized(result, tb_max_period, tb_profit, tb_stop)
    
    # Trend labels
    result = add_trend_labels(result)
    
    # Regime labels (current regime)
    result = add_regime_labels(result)
    
    # Volatility labels (forward-looking)
    result = add_volatility_labels(result)
    
    # Meta-labels (if base predictions available)
    result = add_meta_labels(result, "fwd_class_60")
    
    # Sample weights
    result = add_sample_weights(result)
    
    return result
