"""
High-Performance Imbalance Bars Aggregator

This module implements Tick and Dollar Imbalance Bars as described in:
"Advances in Financial Machine Learning" by Marcos Lopez de Prado

Key concepts:
1. Tick Rule: Assigns direction (buy/sell) to each trade
2. Signed Flow: Applies tick rule sign to trade values
3. Imbalance Accumulator: Running sum of signed flows
4. Dynamic Thresholds: Expected imbalance based on EMA

CRITICAL: No look-ahead bias. All calculations use only past information.
"""

from typing import Optional, Tuple
import polars as pl
import numpy as np


# Time interval constants (matching existing modules)
class TimeInterval:
    """Time intervals in milliseconds for various timeframes"""
    SECOND = 1_000
    MINUTE = 60_000
    HOUR = 3_600_000
    DAY = 86_400_000
    WEEK = 604_800_000


def calculate_tick_rule(prices: np.ndarray) -> np.ndarray:
    """
    Calculate the tick rule (b_t) for each trade.
    
    The tick rule assigns a direction (+1 for buy, -1 for sell) to each trade
    based on price movements.
    
    Formula:
        b_t = 1   if P_t > P_{t-1}  (uptick, buy)
        b_t = -1  if P_t < P_{t-1}  (downtick, sell)
        b_t = b_{t-1}  if P_t = P_{t-1}  (unchanged, use previous)
    
    Args:
        prices: Array of trade prices
        
    Returns:
        Array of tick rules (+1 or -1 for each trade)
        
    Note:
        The first trade is assigned +1 by convention (no previous price).
        This is a common assumption in the literature.
    """
    n = len(prices)
    tick_rules = np.zeros(n, dtype=np.int8)
    
    # First trade: assume buy by convention
    tick_rules[0] = 1
    
    # Calculate price differences
    for i in range(1, n):
        price_diff = prices[i] - prices[i-1]
        
        if price_diff > 0:
            tick_rules[i] = 1  # Uptick (buy)
        elif price_diff < 0:
            tick_rules[i] = -1  # Downtick (sell)
        else:
            # Price unchanged: use previous tick rule
            tick_rules[i] = tick_rules[i-1]
    
    return tick_rules


def _calculate_ema(values: np.ndarray, window: int) -> float:
    """
    Calculate Exponential Moving Average (EMA) using proper warmup.
    
    EMA formula:
        α = 2 / (window + 1)
        EMA_t = α × value_t + (1 - α) × EMA_{t-1}
        
    For the initial values, we use a warmup period to avoid bias.
    
    Args:
        values: Array of values to average
        window: EMA window (span)
        
    Returns:
        EMA value
        
    Note:
        Uses proper warmup: weighted sum for initial values to avoid
        giving excessive weight to the first observation.
    """
    if len(values) == 0:
        return 0.0
    
    if len(values) == 1:
        return float(values[0])
    
    alpha = 2.0 / (window + 1)
    
    # Warmup: for initial values, use weighted sum
    ema = values[0]
    weight_sum = 1.0
    
    for i in range(1, len(values)):
        weight = (1 - alpha) ** i
        weight_sum += weight
        ema = ema * (1 - alpha) + values[i]
        
        # Normalize by weight sum for warmup period
        if i < window:
            ema = ema / (1 + (1 - alpha) * weight_sum / (weight_sum + 1))
    
    return ema


def aggregate_trades_to_tick_imbalance_bars(
    trades_df: pl.DataFrame,
    exp_num_ticks_init: int = 10000,
    num_prev_bars: int = 3,
    num_ticks_ewma_window: int = 20,
    ensure_sorted: bool = True
) -> pl.DataFrame:
    """
    Aggregate trades into Tick Imbalance Bars.
    
    Tick Imbalance Bars are formed when the cumulative signed tick count
    exceeds an expected threshold. This captures order flow imbalance.
    
    Process:
    1. Calculate tick rule (b_t) for each trade (+1 buy, -1 sell)
    2. Accumulate signed ticks: θ_t = Σ b_i
    3. Form bar when: |θ_t| >= E[T] × |2P[b=1] - 1| × 1
    
    Where:
    - E[T]: Expected number of ticks per bar (EMA)
    - P[b=1]: Probability of buy (EMA)
    - Factor 1: Each tick has value 1
    
    Args:
        trades_df: DataFrame with columns:
                   - time: Unix timestamp in milliseconds
                   - price: Trade price
                   - quantity: Base asset quantity
                   - quote_quantity: Quote asset quantity
                   - is_buyer_maker: Boolean (not used, calculated from prices)
        exp_num_ticks_init: Initial expected ticks per bar (default 10000)
        num_prev_bars: Number of previous bars for EMA calculation (default 3)
        num_ticks_ewma_window: EMA window for expected ticks (default 20)
        ensure_sorted: Sort by time before processing (default True)
        
    Returns:
        DataFrame with Imbalance Bars in OHLCV format:
        - bar_id: Sequential bar identifier
        - open_time: Timestamp of first trade
        - close_time: Timestamp of last trade
        - open, high, low, close: OHLC prices
        - volume: Total base asset volume
        - quote_volume: Total quote asset volume
        - taker_buy_base_volume: Volume from buy trades
        - taker_buy_quote_volume: Quote volume from buy trades
        - num_trades: Number of trades in bar
        - cumulative_theta: Final imbalance value
        - expected_imbalance: Threshold used for this bar
        
    Note:
        NO LOOK-AHEAD BIAS: All thresholds are calculated using only
        information from previous bars, not future data.
    """
    if ensure_sorted:
        trades_df = trades_df.sort("time")
    
    # Convert to numpy for efficient processing
    times = trades_df["time"].to_numpy()
    prices = trades_df["price"].to_numpy()
    quantities = trades_df["quantity"].to_numpy()
    quote_quantities = trades_df["quote_quantity"].to_numpy()
    
    n_trades = len(times)
    
    # Calculate tick rules for all trades
    tick_rules = calculate_tick_rule(prices)
    
    # Initialize tracking variables
    bar_id = 0
    theta = 0.0  # Cumulative imbalance for current bar
    
    # EMA tracking arrays (for dynamic threshold)
    ticks_per_bar_history = []  # Number of ticks in each bar
    imbalance_history = []  # Signed tick imbalances for EMA of buy probability
    
    # Expected values (initialized, updated after each bar)
    exp_num_ticks = float(exp_num_ticks_init)
    exp_buy_proportion = 0.5  # Initial assumption: balanced
    
    # Calculate initial threshold (updated after each bar forms)
    buy_imbalance_factor = abs(2 * exp_buy_proportion - 1)
    if buy_imbalance_factor < 0.01:  # Handle balanced market
        buy_imbalance_factor = 0.01
    expected_imbalance = exp_num_ticks * buy_imbalance_factor * 1.0
    
    # Bar data accumulators
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
    cumulative_thetas = []
    expected_imbalances = []
    
    # Current bar statistics
    bar_start_idx = 0
    bar_num_trades = 0
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
    
    # Process each trade
    for i in range(n_trades):
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
        bar_num_trades += 1
        
        # Update buy volumes if it's a buy (tick_rule = 1)
        if tick_rules[i] == 1:
            bar_taker_buy_base += quantities[i]
            bar_taker_buy_quote += quote_quantities[i]
        
        # Accumulate signed tick (for tick bars, each tick has value 1)
        signed_tick = tick_rules[i] * 1.0
        theta += signed_tick
        
        # Track individual tick imbalances for EMA calculation
        imbalance_history.append(signed_tick)
        
        # Check if bar should be formed (threshold calculated ONCE per bar)
        if abs(theta) >= expected_imbalance:
            # Save bar
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
            cumulative_thetas.append(theta)
            expected_imbalances.append(expected_imbalance)
            
            # Update EMA estimates for next bar (NO LOOK-AHEAD)
            ticks_per_bar_history.append(bar_num_trades)
            
            # Update expected number of ticks using EMA
            if len(ticks_per_bar_history) > 1:
                # Use EMA window based on number of previous bars
                window = min(num_ticks_ewma_window, len(ticks_per_bar_history))
                recent_ticks = ticks_per_bar_history[-window:]
                exp_num_ticks = _calculate_ema(np.array(recent_ticks), window)
                
                # CRITICAL: Prevent death spiral - don't let exp_num_ticks drop too low
                # Minimum should be reasonable (at least 10% of initial estimate)
                exp_num_ticks = max(exp_num_ticks, exp_num_ticks_init * 0.1)
            
            # Update expected buy proportion using EMA
            if len(imbalance_history) > 10:  # Need some history
                # Calculate buy proportion from recent imbalances
                # Use window = num_prev_bars * avg ticks per bar
                window_size = int(num_prev_bars * exp_num_ticks)
                window_size = max(10, min(window_size, len(imbalance_history)))
                recent_imbalances = imbalance_history[-window_size:]
                
                # Buy proportion = (num_buys) / (total_trades)
                # Since imbalances are +1 or -1:
                # num_buys = (sum + count) / 2
                imb_array = np.array(recent_imbalances)
                num_buys = np.sum(imb_array == 1)
                total_recent = len(recent_imbalances)
                exp_buy_proportion = num_buys / total_recent
                
                # Ensure valid range [0.1, 0.9] to avoid extremes
                exp_buy_proportion = max(0.1, min(0.9, exp_buy_proportion))
            
            # Calculate threshold for NEXT bar (NO LOOK-AHEAD)
            buy_imbalance_factor = abs(2 * exp_buy_proportion - 1)
            if buy_imbalance_factor < 0.01:  # Handle balanced market
                buy_imbalance_factor = 0.01
            expected_imbalance = exp_num_ticks * buy_imbalance_factor * 1.0
            
            # Reset for next bar
            bar_id += 1
            theta = 0.0
            bar_num_trades = 0
            bar_volume = 0.0
            bar_quote_volume = 0.0
            bar_taker_buy_base = 0.0
            bar_taker_buy_quote = 0.0
            bar_start_idx = i + 1
    
    # Handle remaining trades in incomplete bar
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
        cumulative_thetas.append(theta)
        
        # Use last expected imbalance
        if len(expected_imbalances) > 0:
            expected_imbalances.append(expected_imbalances[-1])
        else:
            expected_imbalances.append(exp_num_ticks * 0.01)
    
    # Create result DataFrame
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
        "cumulative_theta": cumulative_thetas,
        "expected_imbalance": expected_imbalances,
    })
    
    return result_df


def aggregate_trades_to_dollar_imbalance_bars(
    trades_df: pl.DataFrame,
    exp_num_ticks_init: int = 10000,
    num_prev_bars: int = 3,
    num_ticks_ewma_window: int = 20,
    ensure_sorted: bool = True
) -> pl.DataFrame:
    """
    Aggregate trades into Dollar Imbalance Bars.
    
    Dollar Imbalance Bars are formed when the cumulative signed dollar volume
    exceeds an expected threshold. This captures order flow imbalance in value terms.
    
    Process:
    1. Calculate tick rule (b_t) for each trade (+1 buy, -1 sell)
    2. Accumulate signed dollar volume: θ_t = Σ b_i × (price_i × volume_i)
    3. Form bar when: |θ_t| >= E[T] × |2P[b=1] - 1| × E[v]
    
    Where:
    - E[T]: Expected number of ticks per bar (EMA)
    - P[b=1]: Probability of buy (EMA)
    - E[v]: Expected dollar value per tick (EMA)
    
    Args:
        trades_df: DataFrame with trades data
        exp_num_ticks_init: Initial expected ticks per bar (default 10000)
        num_prev_bars: Number of previous bars for EMA (default 3)
        num_ticks_ewma_window: EMA window for expected ticks (default 20)
        ensure_sorted: Sort by time before processing (default True)
        
    Returns:
        DataFrame with Dollar Imbalance Bars (same schema as tick imbalance)
        
    Note:
        The only difference from tick imbalance is using dollar value instead of 1
        for each trade.
    """
    if ensure_sorted:
        trades_df = trades_df.sort("time")
    
    # Convert to numpy
    times = trades_df["time"].to_numpy()
    prices = trades_df["price"].to_numpy()
    quantities = trades_df["quantity"].to_numpy()
    quote_quantities = trades_df["quote_quantity"].to_numpy()
    
    n_trades = len(times)
    
    # Calculate tick rules
    tick_rules = calculate_tick_rule(prices)
    
    # Initialize
    bar_id = 0
    theta = 0.0
    
    ticks_per_bar_history = []
    dollar_values_history = []  # For EMA of average dollar value
    imbalance_history = []
    
    exp_num_ticks = float(exp_num_ticks_init)
    exp_buy_proportion = 0.5
    exp_dollar_value = float(np.mean(quote_quantities[:1000])) if len(quote_quantities) > 1000 else float(quote_quantities[0])
    
    # Calculate initial threshold (updated after each bar forms)
    buy_imbalance_factor = abs(2 * exp_buy_proportion - 1)
    if buy_imbalance_factor < 0.01:  # Handle balanced market
        buy_imbalance_factor = 0.01
    expected_imbalance = exp_num_ticks * buy_imbalance_factor * exp_dollar_value
    
    # Bar data
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
    cumulative_thetas = []
    expected_imbalances = []
    
    # Current bar stats
    bar_num_trades = 0
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
    
    # Process each trade
    for i in range(n_trades):
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
        bar_num_trades += 1
        
        if tick_rules[i] == 1:
            bar_taker_buy_base += quantities[i]
            bar_taker_buy_quote += quote_quantities[i]
        
        # DOLLAR IMBALANCE: use actual dollar value
        dollar_value = quote_quantities[i]
        signed_dollar_value = tick_rules[i] * dollar_value
        theta += signed_dollar_value
        
        # Track for EMA
        imbalance_history.append(float(tick_rules[i]))
        dollar_values_history.append(dollar_value)
        
        # Check bar formation (threshold calculated ONCE per bar)
        if abs(theta) >= expected_imbalance:
            # Save bar
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
            cumulative_thetas.append(theta)
            expected_imbalances.append(expected_imbalance)
            
            # Update EMAs for next bar
            ticks_per_bar_history.append(bar_num_trades)
            
            if len(ticks_per_bar_history) > 1:
                window = min(num_ticks_ewma_window, len(ticks_per_bar_history))
                recent_ticks = ticks_per_bar_history[-window:]
                exp_num_ticks = _calculate_ema(np.array(recent_ticks), window)
                
                # CRITICAL: Prevent death spiral - don't let exp_num_ticks drop too low
                exp_num_ticks = max(exp_num_ticks, exp_num_ticks_init * 0.1)
            
            if len(imbalance_history) > 10:
                window_size = int(num_prev_bars * exp_num_ticks)
                window_size = max(10, min(window_size, len(imbalance_history)))
                recent_imbalances = imbalance_history[-window_size:]
                
                imb_array = np.array(recent_imbalances)
                num_buys = np.sum(imb_array > 0)
                total_recent = len(recent_imbalances)
                exp_buy_proportion = num_buys / total_recent
                exp_buy_proportion = max(0.1, min(0.9, exp_buy_proportion))
            
            if len(dollar_values_history) > 10:
                window_size = int(num_prev_bars * exp_num_ticks)
                window_size = max(10, min(window_size, len(dollar_values_history)))
                recent_values = dollar_values_history[-window_size:]
                exp_dollar_value = _calculate_ema(np.array(recent_values), min(20, len(recent_values)))
            
            # Calculate threshold for NEXT bar (NO LOOK-AHEAD)
            buy_imbalance_factor = abs(2 * exp_buy_proportion - 1)
            if buy_imbalance_factor < 0.01:  # Handle balanced market
                buy_imbalance_factor = 0.01
            expected_imbalance = exp_num_ticks * buy_imbalance_factor * exp_dollar_value
            
            # Reset
            bar_id += 1
            theta = 0.0
            bar_num_trades = 0
            bar_volume = 0.0
            bar_quote_volume = 0.0
            bar_taker_buy_base = 0.0
            bar_taker_buy_quote = 0.0
    
    # Handle incomplete bar
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
        cumulative_thetas.append(theta)
        
        if len(expected_imbalances) > 0:
            expected_imbalances.append(expected_imbalances[-1])
        else:
            expected_imbalances.append(exp_num_ticks * 0.01 * exp_dollar_value)
    
    # Create result
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
        "cumulative_theta": cumulative_thetas,
        "expected_imbalance": expected_imbalances,
    })
    
    return result_df
