"""
High-Performance Dollar Bars Aggregator

This module implements Dollar Bars aggregation with two approaches:
1. Fixed Threshold: Constant dollar volume threshold
2. Dynamic Threshold: Adaptive threshold based on EMA of daily volume

The implementation focuses on:
- High accuracy with proper residual handling
- Maximum performance using Polars
- Maintaining Unix millisecond timestamps
- Avoiding unnecessary datetime conversions
"""

from typing import Optional, Tuple, Dict
import polars as pl
import numpy as np


# Time interval constants in milliseconds (matching existing module pattern)
class TimeInterval:
    """Time intervals in milliseconds for various timeframes"""
    SECOND = 1_000          # 1 second
    MINUTE = 60_000         # 1 minute
    HOUR = 3_600_000        # 1 hour
    DAY = 86_400_000        # 1 day
    WEEK = 604_800_000      # 1 week


def _calculate_dollar_value(trades_df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate dollar value for each trade.
    
    Dollar value is already available in the 'quote_quantity' column,
    which represents price * quantity. However, we'll ensure it's 
    correctly calculated if needed.
    
    Args:
        trades_df: DataFrame with 'price', 'quantity', and 'quote_quantity' columns
        
    Returns:
        DataFrame with 'dollar_value' column added
        
    Note:
        Uses quote_quantity directly for efficiency, but can recalculate
        if needed: dollar_value = price * quantity
    """
    return trades_df.with_columns([
        pl.col("quote_quantity").alias("dollar_value")
    ])


def _calculate_time_bin(time_ms: int, interval_ms: int) -> int:
    """
    Calculate the time bin (bucket) for a timestamp.
    
    Args:
        time_ms: Unix timestamp in milliseconds
        interval_ms: Interval size in milliseconds
        
    Returns:
        Start timestamp of the bin
    """
    return (time_ms // interval_ms) * interval_ms


def aggregate_trades_to_dollar_bars_fixed(
    trades_df: pl.DataFrame,
    threshold: float,
    ensure_sorted: bool = True
) -> pl.DataFrame:
    """
    Aggregate trades into Dollar Bars using a fixed threshold.
    
    This function implements the core Dollar Bars logic:
    1. Calculate dollar value for each trade (price * volume)
    2. Accumulate dollar values until threshold is reached
    3. Create a new bar when threshold is exceeded
    4. Carry forward the residual (overshoot) to the next bar
    
    The Formula:
        θ_t = Σ(price_i × volume_i)  from last bar to current trade
        New bar triggered when: θ_t ≥ T (threshold)
        Residual carried forward: θ_new = θ_t - T
    
    Args:
        trades_df: DataFrame with columns:
                   - time: Unix timestamp in milliseconds
                   - price: Trade price
                   - quantity: Base asset quantity
                   - quote_quantity: Quote asset quantity (price * quantity)
                   - is_buyer_maker: Boolean indicating if buyer is maker
        threshold: Dollar volume threshold (e.g., 10_000_000 for $10M)
        ensure_sorted: Whether to sort by time (default True for accuracy)
        
    Returns:
        DataFrame with Dollar Bars in OHLCV format:
        - bar_id: Sequential bar identifier
        - open_time: Timestamp of first trade in bar
        - close_time: Timestamp of last trade in bar
        - open: First trade price in bar
        - high: Highest trade price in bar
        - low: Lowest trade price in bar
        - close: Last trade price in bar
        - volume: Total base asset volume
        - quote_volume: Total quote asset volume (should equal threshold approximately)
        - taker_buy_base_volume: Volume from buyer-initiated trades
        - taker_buy_quote_volume: Quote volume from buyer-initiated trades
        - num_trades: Number of trades in bar
        - bar_dollar_volume: Actual dollar volume in bar (including residual effects)
        
    Performance:
        For ~148M trades: Expected runtime < 5 seconds on modern hardware
        
    Accuracy:
        Residual handling ensures no dollar volume is lost between bars.
        Each bar's starting cumulative sum includes the previous bar's overshoot.
    """
    # Ensure data is sorted by time for correct sequential processing
    if ensure_sorted:
        trades_df = trades_df.sort("time")
    
    # Calculate dollar value if not already present
    if "dollar_value" not in trades_df.columns:
        trades_df = _calculate_dollar_value(trades_df)
    
    # Convert to numpy for efficient iteration (Polars to_numpy is fast)
    # We need: time, price, quantity, quote_quantity, is_buyer_maker, dollar_value
    times = trades_df["time"].to_numpy()
    prices = trades_df["price"].to_numpy()
    quantities = trades_df["quantity"].to_numpy()
    quote_quantities = trades_df["quote_quantity"].to_numpy()
    is_buyer_maker = trades_df["is_buyer_maker"].to_numpy()
    dollar_values = trades_df["dollar_value"].to_numpy()
    
    n_trades = len(times)
    
    # Pre-allocate lists for bar data (more efficient than appending)
    # Estimate number of bars to pre-allocate (conservative estimate)
    estimated_bars = int(np.sum(dollar_values) / threshold) + 1
    
    bar_ids = []
    open_times = []
    close_times = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    quote_volumes = []
    taker_buy_base_volumes = []
    taker_buy_quote_volumes = []
    num_trades_list = []
    bar_dollar_volumes = []
    
    # Initialize accumulation state
    cumsum = 0.0  # Current cumulative dollar value (θ_t)
    bar_id = 0
    bar_start_idx = 0
    
    # Variables to track current bar statistics
    bar_open_time = times[0]
    bar_close_time = times[0]
    bar_open_price = prices[0]
    bar_high_price = prices[0]
    bar_low_price = prices[0]
    bar_close_price = prices[0]
    bar_volume = 0.0
    bar_quote_volume = 0.0
    bar_taker_buy_base = 0.0
    bar_taker_buy_quote = 0.0
    bar_num_trades = 0
    bar_dollar_volume = 0.0
    
    # Iterate through all trades
    for i in range(n_trades):
        # Add current trade to accumulation
        cumsum += dollar_values[i]
        
        # Update bar statistics
        if bar_num_trades == 0:
            # First trade in bar
            bar_open_time = times[i]
            bar_open_price = prices[i]
            bar_high_price = prices[i]
            bar_low_price = prices[i]
        else:
            # Update high and low
            if prices[i] > bar_high_price:
                bar_high_price = prices[i]
            if prices[i] < bar_low_price:
                bar_low_price = prices[i]
        
        bar_close_time = times[i]
        bar_close_price = prices[i]
        bar_volume += quantities[i]
        bar_quote_volume += quote_quantities[i]
        bar_dollar_volume += dollar_values[i]
        bar_num_trades += 1
        
        # Update taker buy volumes (if buyer is NOT maker, it's a taker buy)
        if not is_buyer_maker[i]:
            bar_taker_buy_base += quantities[i]
            bar_taker_buy_quote += quote_quantities[i]
        
        # Check if threshold is reached
        if cumsum >= threshold:
            # Save current bar
            bar_ids.append(bar_id)
            open_times.append(bar_open_time)
            close_times.append(bar_close_time)
            opens.append(bar_open_price)
            highs.append(bar_high_price)
            lows.append(bar_low_price)
            closes.append(bar_close_price)
            volumes.append(bar_volume)
            quote_volumes.append(bar_quote_volume)
            taker_buy_base_volumes.append(bar_taker_buy_base)
            taker_buy_quote_volumes.append(bar_taker_buy_quote)
            num_trades_list.append(bar_num_trades)
            bar_dollar_volumes.append(bar_dollar_volume)
            
            # CRITICAL: Handle residual (overshoot)
            # The cumsum may exceed threshold, carry forward the excess
            cumsum -= threshold
            
            # Reset bar statistics for next bar
            bar_id += 1
            bar_num_trades = 0
            bar_volume = 0.0
            bar_quote_volume = 0.0
            bar_taker_buy_base = 0.0
            bar_taker_buy_quote = 0.0
            bar_dollar_volume = 0.0
    
    # Handle remaining trades in incomplete final bar
    if bar_num_trades > 0:
        bar_ids.append(bar_id)
        open_times.append(bar_open_time)
        close_times.append(bar_close_time)
        opens.append(bar_open_price)
        highs.append(bar_high_price)
        lows.append(bar_low_price)
        closes.append(bar_close_price)
        volumes.append(bar_volume)
        quote_volumes.append(bar_quote_volume)
        taker_buy_base_volumes.append(bar_taker_buy_base)
        taker_buy_quote_volumes.append(bar_taker_buy_quote)
        num_trades_list.append(bar_num_trades)
        bar_dollar_volumes.append(bar_dollar_volume)
    
    # Create result DataFrame using Polars
    result_df = pl.DataFrame({
        "bar_id": bar_ids,
        "open_time": open_times,
        "close_time": close_times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "quote_volume": quote_volumes,
        "taker_buy_base_volume": taker_buy_base_volumes,
        "taker_buy_quote_volume": taker_buy_quote_volumes,
        "num_trades": num_trades_list,
        "bar_dollar_volume": bar_dollar_volumes,
    })
    
    return result_df


def calculate_daily_dollar_volume(
    trades_df: pl.DataFrame,
    day_interval_ms: int = TimeInterval.DAY
) -> pl.DataFrame:
    """
    Calculate total dollar volume for each day.
    
    This function aggregates trades into daily buckets and sums
    the total dollar volume for each day.
    
    Args:
        trades_df: DataFrame with 'time' and 'quote_quantity' columns
        day_interval_ms: Interval for a day in milliseconds (default 86400000)
        
    Returns:
        DataFrame with:
        - day_start: Start timestamp of the day
        - daily_dollar_volume: Total dollar volume for that day
        
    Note:
        Uses floor division to assign each trade to its day bin.
    """
    # Calculate day bin for each trade
    daily_volume = trades_df.select([
        ((pl.col("time") // day_interval_ms) * day_interval_ms).alias("day_start"),
        pl.col("quote_quantity").alias("dollar_value")
    ]).group_by("day_start").agg([
        pl.sum("dollar_value").alias("daily_dollar_volume")
    ]).sort("day_start")
    
    return daily_volume


def calculate_ema_daily_volume(
    daily_volumes: pl.DataFrame,
    span: int = 20,
    min_periods: Optional[int] = None
) -> pl.DataFrame:
    """
    Calculate Exponential Moving Average (EMA) of daily dollar volume.
    
    EMA gives more weight to recent observations and is commonly used
    for adaptive thresholds in financial analysis.
    
    Formula:
        EMA_t = α × V_t + (1 - α) × EMA_{t-1}
        where α = 2 / (span + 1)
    
    Args:
        daily_volumes: DataFrame with 'day_start' and 'daily_dollar_volume' columns
        span: Number of days for EMA calculation (default 20)
        min_periods: Minimum periods required for EMA (default = span)
        
    Returns:
        DataFrame with additional column:
        - ema_daily_volume: EMA of daily dollar volume
        
    Note:
        For first few days where we don't have enough data, the EMA
        will use available data (min_periods can adjust this behavior).
    """
    if min_periods is None:
        min_periods = span
    
    # Calculate EMA using Polars' efficient rolling operations
    # Note: Polars EMA is calculated with adjust=True by default
    result = daily_volumes.with_columns([
        pl.col("daily_dollar_volume")
          .ewm_mean(span=span, min_periods=min_periods)
          .alias("ema_daily_volume")
    ])
    
    return result


def calculate_dynamic_threshold(
    ema_daily_volume: float,
    target_bars_per_day: int
) -> float:
    """
    Calculate dynamic threshold based on EMA of daily volume.
    
    Formula:
        T = EMA(Daily Dollar Volume) / K
        where K is the target number of bars per day
    
    Args:
        ema_daily_volume: EMA of daily dollar volume
        target_bars_per_day: Desired number of bars per day (K)
        
    Returns:
        Threshold value for Dollar Bars
        
    Example:
        If average daily volume is $500M and we want 50 bars/day:
        threshold = $500M / 50 = $10M per bar
    """
    if target_bars_per_day <= 0:
        raise ValueError("target_bars_per_day must be positive")
    
    return ema_daily_volume / target_bars_per_day


def aggregate_trades_to_dollar_bars_dynamic(
    trades_df: pl.DataFrame,
    target_bars_per_day: int,
    ema_span: int = 20,
    initial_threshold: Optional[float] = None,
    recalculation_interval_ms: int = TimeInterval.DAY,
    ensure_sorted: bool = True
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Aggregate trades into Dollar Bars using dynamic adaptive threshold.
    
    This function implements adaptive Dollar Bars where the threshold
    adjusts based on recent market activity using EMA of daily volume.
    
    Process:
    1. Calculate daily dollar volumes from trades
    2. Compute EMA of daily volumes
    3. Update threshold periodically (e.g., daily) based on: T = EMA / K
    4. Apply threshold to create Dollar Bars with residual handling
    
    Args:
        trades_df: DataFrame with trade data (same schema as fixed threshold)
        target_bars_per_day: Desired number of bars per day (K)
        ema_span: Number of days for EMA calculation (default 20)
        initial_threshold: Optional initial threshold for first day
                          (if None, uses simple average from data)
        recalculation_interval_ms: How often to recalculate threshold
                                   (default: daily)
        ensure_sorted: Whether to sort by time (default True)
        
    Returns:
        Tuple of (bars_df, threshold_history_df):
        - bars_df: Dollar Bars with OHLCV format (same as fixed threshold)
                  plus 'threshold_used' column
        - threshold_history_df: DataFrame with threshold updates
                               (day_start, threshold)
    
    Performance:
        Slightly slower than fixed threshold due to threshold calculations,
        but still efficient for large datasets.
        
    Note:
        The threshold adapts to market conditions, creating more bars
        during high-volume periods and fewer during low-volume periods.
    """
    # Ensure data is sorted by time
    if ensure_sorted:
        trades_df = trades_df.sort("time")
    
    # Calculate dollar value if not present
    if "dollar_value" not in trades_df.columns:
        trades_df = _calculate_dollar_value(trades_df)
    
    # Step 1: Calculate daily dollar volumes
    daily_volumes = calculate_daily_dollar_volume(
        trades_df, 
        day_interval_ms=recalculation_interval_ms
    )
    
    # Step 2: Calculate EMA of daily volumes
    daily_volumes_with_ema = calculate_ema_daily_volume(
        daily_volumes,
        span=ema_span,
        min_periods=min(ema_span, len(daily_volumes))
    )
    
    # Step 3: Calculate threshold for each day
    thresholds = daily_volumes_with_ema.with_columns([
        (pl.col("ema_daily_volume") / target_bars_per_day).alias("threshold")
    ]).select(["day_start", "threshold", "ema_daily_volume"])
    
    # Handle initial threshold if EMA is null for first few days
    if initial_threshold is None:
        # Use average of first available thresholds as initial
        first_valid_threshold = thresholds.filter(
            pl.col("threshold").is_not_null()
        )["threshold"].first()
        
        if first_valid_threshold is None:
            # Fallback: calculate simple average from all data
            total_dollar_volume = trades_df["dollar_value"].sum()
            time_range_days = (trades_df["time"].max() - trades_df["time"].min()) / TimeInterval.DAY
            avg_daily_volume = total_dollar_volume / max(1, time_range_days)
            initial_threshold = avg_daily_volume / target_bars_per_day
        else:
            initial_threshold = first_valid_threshold
    
    # Fill null thresholds with initial threshold
    thresholds = thresholds.with_columns([
        pl.col("threshold").fill_null(initial_threshold)
    ])
    
    # Step 4: Apply dynamic thresholds to create bars
    # Convert trades and thresholds to numpy for efficient processing
    times = trades_df["time"].to_numpy()
    prices = trades_df["price"].to_numpy()
    quantities = trades_df["quantity"].to_numpy()
    quote_quantities = trades_df["quote_quantity"].to_numpy()
    is_buyer_maker = trades_df["is_buyer_maker"].to_numpy()
    dollar_values = trades_df["dollar_value"].to_numpy()
    
    # Create a lookup for thresholds by day
    threshold_dict = {
        row[0]: row[1] 
        for row in thresholds.select(["day_start", "threshold"]).iter_rows()
    }
    
    n_trades = len(times)
    
    # Pre-allocate lists for bar data
    bar_ids = []
    open_times = []
    close_times = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    quote_volumes = []
    taker_buy_base_volumes = []
    taker_buy_quote_volumes = []
    num_trades_list = []
    bar_dollar_volumes = []
    thresholds_used = []
    
    # Initialize state
    cumsum = 0.0
    bar_id = 0
    current_threshold = initial_threshold
    
    # Bar statistics
    bar_open_time = times[0]
    bar_close_time = times[0]
    bar_open_price = prices[0]
    bar_high_price = prices[0]
    bar_low_price = prices[0]
    bar_close_price = prices[0]
    bar_volume = 0.0
    bar_quote_volume = 0.0
    bar_taker_buy_base = 0.0
    bar_taker_buy_quote = 0.0
    bar_num_trades = 0
    bar_dollar_volume = 0.0
    
    # Track last day bin to detect day changes
    last_day_bin = _calculate_time_bin(times[0], recalculation_interval_ms)
    
    # Iterate through all trades
    for i in range(n_trades):
        # Check if we've entered a new day and update threshold
        current_day_bin = _calculate_time_bin(times[i], recalculation_interval_ms)
        if current_day_bin != last_day_bin:
            # Update threshold for new day
            if current_day_bin in threshold_dict:
                current_threshold = threshold_dict[current_day_bin]
            last_day_bin = current_day_bin
        
        # Add current trade to accumulation
        cumsum += dollar_values[i]
        
        # Update bar statistics
        if bar_num_trades == 0:
            bar_open_time = times[i]
            bar_open_price = prices[i]
            bar_high_price = prices[i]
            bar_low_price = prices[i]
        else:
            if prices[i] > bar_high_price:
                bar_high_price = prices[i]
            if prices[i] < bar_low_price:
                bar_low_price = prices[i]
        
        bar_close_time = times[i]
        bar_close_price = prices[i]
        bar_volume += quantities[i]
        bar_quote_volume += quote_quantities[i]
        bar_dollar_volume += dollar_values[i]
        bar_num_trades += 1
        
        if not is_buyer_maker[i]:
            bar_taker_buy_base += quantities[i]
            bar_taker_buy_quote += quote_quantities[i]
        
        # Check if threshold is reached
        if cumsum >= current_threshold:
            # Save current bar
            bar_ids.append(bar_id)
            open_times.append(bar_open_time)
            close_times.append(bar_close_time)
            opens.append(bar_open_price)
            highs.append(bar_high_price)
            lows.append(bar_low_price)
            closes.append(bar_close_price)
            volumes.append(bar_volume)
            quote_volumes.append(bar_quote_volume)
            taker_buy_base_volumes.append(bar_taker_buy_base)
            taker_buy_quote_volumes.append(bar_taker_buy_quote)
            num_trades_list.append(bar_num_trades)
            bar_dollar_volumes.append(bar_dollar_volume)
            thresholds_used.append(current_threshold)
            
            # Handle residual
            cumsum -= current_threshold
            
            # Reset for next bar
            bar_id += 1
            bar_num_trades = 0
            bar_volume = 0.0
            bar_quote_volume = 0.0
            bar_taker_buy_base = 0.0
            bar_taker_buy_quote = 0.0
            bar_dollar_volume = 0.0
    
    # Handle remaining trades
    if bar_num_trades > 0:
        bar_ids.append(bar_id)
        open_times.append(bar_open_time)
        close_times.append(bar_close_time)
        opens.append(bar_open_price)
        highs.append(bar_high_price)
        lows.append(bar_low_price)
        closes.append(bar_close_price)
        volumes.append(bar_volume)
        quote_volumes.append(bar_quote_volume)
        taker_buy_base_volumes.append(bar_taker_buy_base)
        taker_buy_quote_volumes.append(bar_taker_buy_quote)
        num_trades_list.append(bar_num_trades)
        bar_dollar_volumes.append(bar_dollar_volume)
        thresholds_used.append(current_threshold)
    
    # Create result DataFrames
    bars_df = pl.DataFrame({
        "bar_id": bar_ids,
        "open_time": open_times,
        "close_time": close_times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "quote_volume": quote_volumes,
        "taker_buy_base_volume": taker_buy_base_volumes,
        "taker_buy_quote_volume": taker_buy_quote_volumes,
        "num_trades": num_trades_list,
        "bar_dollar_volume": bar_dollar_volumes,
        "threshold_used": thresholds_used,
    })
    
    return bars_df, thresholds
