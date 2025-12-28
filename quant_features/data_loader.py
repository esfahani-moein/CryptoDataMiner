"""
Data Loader Module

Functions to load and merge multiple data sources for feature engineering.
Handles path discovery, file loading, and timestamp normalization.

Data Sources:
- trades: High-frequency trade data
- metrics: Open interest, long/short ratios
- funding_rate: Perpetual funding rates
- book_depth: Order book depth snapshots
- klines: OHLCV data (mark, index, premium)

All functions use Polars for efficient data processing.
"""

from pathlib import Path
from typing import List, Optional, Union, Dict
import polars as pl


# ============================================================================
# PATH DISCOVERY FUNCTIONS
# ============================================================================

def discover_data_files(
    base_path: Union[str, Path],
    symbol: str,
    data_type: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int
) -> List[Path]:
    """
    Discover parquet files for a date range.
    
    Directory structure expected:
    {base_path}/dataset_{symbol}/{year}_{month:02d}/{data_type}/
    
    Args:
        base_path: Base directory containing dataset folders
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        data_type: Type of data (trades, metrics, fundingRate, etc.)
        start_year: Start year
        start_month: Start month (1-12)
        end_year: End year
        end_month: End month (1-12)
        
    Returns:
        List of Path objects to parquet files, sorted by date
        
    Example:
        >>> files = discover_data_files('dataset', 'BTCUSDT', 'trades', 2025, 11, 2025, 12)
    """
    base_path = Path(base_path)
    files = []
    
    current_year = start_year
    current_month = start_month
    
    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        folder = base_path / f"dataset_{symbol}" / f"{current_year}_{current_month:02d}" / data_type
        
        if folder.exists():
            # Find parquet files in the folder
            parquet_files = sorted(folder.glob("*.parquet"))
            files.extend(parquet_files)
        
        # Move to next month
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    
    return files


# ============================================================================
# LOADING FUNCTIONS
# ============================================================================

def load_trades(
    base_path: Union[str, Path],
    symbol: str = "BTCUSDT",
    start_year: int = 2025,
    start_month: int = 11,
    end_year: int = 2025,
    end_month: int = 11
) -> pl.DataFrame:
    """
    Load trades data from parquet files.
    
    Schema:
    - id: Trade ID (Int64)
    - price: Trade price (Float64)
    - qty: Base asset quantity (Float64)
    - quote_qty: Quote asset quantity (Float64)
    - time: Unix timestamp in milliseconds (Int64)
    - is_buyer_maker: True if buyer was maker (Boolean)
    
    Args:
        base_path: Base directory containing dataset folders
        symbol: Trading pair symbol
        start_year, start_month: Start of date range
        end_year, end_month: End of date range
        
    Returns:
        Polars DataFrame with trades data, sorted by time
    """
    files = discover_data_files(base_path, symbol, "trades", 
                                start_year, start_month, end_year, end_month)
    
    if not files:
        raise FileNotFoundError(f"No trades files found for {symbol}")
    
    # Load and concatenate all files
    dfs = [pl.read_parquet(f) for f in files]
    df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]
    
    # Ensure sorted by time
    return df.sort("time")


def load_metrics(
    base_path: Union[str, Path],
    symbol: str = "BTCUSDT",
    start_year: int = 2025,
    start_month: int = 11,
    end_year: int = 2025,
    end_month: int = 11
) -> pl.DataFrame:
    """
    Load metrics data (open interest, long/short ratios).
    
    Schema:
    - create_time: Timestamp string (needs parsing)
    - symbol: Trading pair
    - sum_open_interest: Total open interest (Float64)
    - sum_open_interest_value: OI in quote currency (Float64)
    - count_toptrader_long_short_ratio: Top trader ratio (Float64)
    - sum_toptrader_long_short_ratio: Top trader position ratio (Float64)
    - count_long_short_ratio: Account long/short ratio (Float64)
    - sum_taker_long_short_vol_ratio: Taker buy/sell ratio (Float64)
    
    Returns:
        DataFrame with metrics, timestamp converted to milliseconds
    """
    files = discover_data_files(base_path, symbol, "metrics",
                                start_year, start_month, end_year, end_month)
    
    if not files:
        raise FileNotFoundError(f"No metrics files found for {symbol}")
    
    dfs = [pl.read_parquet(f) for f in files]
    df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]
    
    # Convert create_time string to datetime then to milliseconds for consistency
    df = df.with_columns([
        pl.col("create_time")
        .str.to_datetime("%Y-%m-%d %H:%M:%S")
        .dt.epoch(time_unit="ms")
        .alias("time")
    ]).drop("create_time")
    
    return df.sort("time")


def load_funding_rate(
    base_path: Union[str, Path],
    symbol: str = "BTCUSDT",
    start_year: int = 2025,
    start_month: int = 11,
    end_year: int = 2025,
    end_month: int = 11
) -> pl.DataFrame:
    """
    Load funding rate data.
    
    Schema:
    - calc_time: Calculation timestamp in ms (Int64)
    - funding_interval_hours: Hours between funding (Int64)
    - last_funding_rate: Funding rate (Float64)
    
    Returns:
        DataFrame with funding rates, renamed for consistency
    """
    files = discover_data_files(base_path, symbol, "fundingRate",
                                start_year, start_month, end_year, end_month)
    
    if not files:
        raise FileNotFoundError(f"No funding rate files found for {symbol}")
    
    dfs = [pl.read_parquet(f) for f in files]
    df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]
    
    # Rename for consistency
    df = df.rename({"calc_time": "time"})
    
    return df.sort("time")


def load_book_depth(
    base_path: Union[str, Path],
    symbol: str = "BTCUSDT",
    start_year: int = 2025,
    start_month: int = 11,
    end_year: int = 2025,
    end_month: int = 11
) -> pl.DataFrame:
    """
    Load order book depth data.
    
    Schema:
    - timestamp: Timestamp string (needs parsing)
    - percentage: Price level percentage (-5 to +5)
    - depth: Quantity at this level (Float64)
    - notional: Notional value at this level (Float64)
    
    Percentage values:
    - Negative: Bid side (e.g., -5 = 5% below mid)
    - Positive: Ask side (e.g., +5 = 5% above mid)
    
    Returns:
        DataFrame with book depth, timestamp in milliseconds
    """
    files = discover_data_files(base_path, symbol, "bookDepth",
                                start_year, start_month, end_year, end_month)
    
    if not files:
        raise FileNotFoundError(f"No book depth files found for {symbol}")
    
    dfs = [pl.read_parquet(f) for f in files]
    df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]
    
    # Convert timestamp to milliseconds
    df = df.with_columns([
        pl.col("timestamp")
        .str.to_datetime("%Y-%m-%d %H:%M:%S")
        .dt.epoch(time_unit="ms")
        .alias("time")
    ]).drop("timestamp")
    
    return df.sort("time")


def load_klines(
    base_path: Union[str, Path],
    symbol: str = "BTCUSDT",
    kline_type: str = "markPriceKlines",  # or indexPriceKlines, premiumIndexKlines
    start_year: int = 2025,
    start_month: int = 11,
    end_year: int = 2025,
    end_month: int = 11
) -> pl.DataFrame:
    """
    Load klines data (OHLCV format).
    
    Available types:
    - markPriceKlines: Mark price used for liquidations
    - indexPriceKlines: Index price (spot average)
    - premiumIndexKlines: Premium/discount to index
    
    Schema (standard OHLCV):
    - open_time: Bar start time in ms (Int64)
    - open, high, low, close: Prices (Float64)
    - volume: Not meaningful for mark/index (always 0)
    - close_time: Bar end time in ms (Int64)
    - quote_volume: Quote volume (Float64)
    - count: Trade count (Int64)
    - taker_buy_volume, taker_buy_quote_volume: Taker metrics
    - ignore: Always 0
    
    Returns:
        DataFrame with klines data, sorted by time
    """
    files = discover_data_files(base_path, symbol, kline_type,
                                start_year, start_month, end_year, end_month)
    
    if not files:
        raise FileNotFoundError(f"No {kline_type} files found for {symbol}")
    
    dfs = [pl.read_parquet(f) for f in files]
    df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]
    
    return df.sort("open_time")


def load_all_data(
    base_path: Union[str, Path],
    symbol: str = "BTCUSDT",
    start_year: int = 2025,
    start_month: int = 11,
    end_year: int = 2025,
    end_month: int = 11
) -> Dict[str, pl.DataFrame]:
    """
    Load all available data types into a dictionary.
    
    Args:
        base_path: Base directory containing dataset folders
        symbol: Trading pair symbol
        start_year, start_month: Start of date range
        end_year, end_month: End of date range
        
    Returns:
        Dictionary with keys:
        - 'trades': Trade data
        - 'metrics': Open interest and ratios
        - 'funding': Funding rate data
        - 'book_depth': Order book depth
        - 'mark_klines': Mark price OHLCV
        - 'index_klines': Index price OHLCV
        - 'premium_klines': Premium index OHLCV
    """
    data = {}
    
    # Load each data type with error handling
    try:
        data['trades'] = load_trades(base_path, symbol, start_year, start_month, 
                                     end_year, end_month)
    except FileNotFoundError:
        data['trades'] = None
        
    try:
        data['metrics'] = load_metrics(base_path, symbol, start_year, start_month,
                                       end_year, end_month)
    except FileNotFoundError:
        data['metrics'] = None
        
    try:
        data['funding'] = load_funding_rate(base_path, symbol, start_year, start_month,
                                            end_year, end_month)
    except FileNotFoundError:
        data['funding'] = None
        
    try:
        data['book_depth'] = load_book_depth(base_path, symbol, start_year, start_month,
                                             end_year, end_month)
    except FileNotFoundError:
        data['book_depth'] = None
        
    try:
        data['mark_klines'] = load_klines(base_path, symbol, "markPriceKlines",
                                          start_year, start_month, end_year, end_month)
    except FileNotFoundError:
        data['mark_klines'] = None
        
    try:
        data['index_klines'] = load_klines(base_path, symbol, "indexPriceKlines",
                                           start_year, start_month, end_year, end_month)
    except FileNotFoundError:
        data['index_klines'] = None
        
    try:
        data['premium_klines'] = load_klines(base_path, symbol, "premiumIndexKlines",
                                             start_year, start_month, end_year, end_month)
    except FileNotFoundError:
        data['premium_klines'] = None
    
    return data


# ============================================================================
# MERGE FUNCTIONS
# ============================================================================

def merge_features_to_ohlcv(
    ohlcv_df: pl.DataFrame,
    features_df: pl.DataFrame,
    ohlcv_time_col: str = "open_time",
    features_time_col: str = "time",
    method: str = "asof"
) -> pl.DataFrame:
    """
    Merge features onto OHLCV bars using point-in-time joins.
    
    CRITICAL: Uses asof join to avoid look-ahead bias.
    Each bar only gets features available at its open_time.
    
    Args:
        ohlcv_df: Base OHLCV DataFrame with timestamp column
        features_df: Features DataFrame with timestamp column
        ohlcv_time_col: Name of timestamp column in OHLCV
        features_time_col: Name of timestamp column in features
        method: Join method ('asof' for point-in-time, 'exact' for exact match)
        
    Returns:
        OHLCV DataFrame with features merged in
        
    Note:
        For 'asof' join: feature at time T is matched to OHLCV bars
        where bar_time >= T. This ensures no look-ahead.
    """
    # Ensure both DataFrames are sorted by time
    ohlcv_sorted = ohlcv_df.sort(ohlcv_time_col)
    features_sorted = features_df.sort(features_time_col)
    
    if method == "asof":
        # As-of join: get the most recent feature available at each bar's time
        # strategy="backward" means we look back for the most recent value
        result = ohlcv_sorted.join_asof(
            features_sorted,
            left_on=ohlcv_time_col,
            right_on=features_time_col,
            strategy="backward"
        )
    elif method == "exact":
        # Exact match only
        result = ohlcv_sorted.join(
            features_sorted,
            left_on=ohlcv_time_col,
            right_on=features_time_col,
            how="left"
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'asof' or 'exact'")
    
    return result


def aggregate_trades_to_ohlcv(
    trades_df: pl.DataFrame,
    interval_ms: int = 60_000  # Default 1 minute
) -> pl.DataFrame:
    """
    Aggregate trades into OHLCV format.
    
    This is a convenience wrapper for creating OHLCV bars from trades.
    Uses the trades_aggregator module for actual aggregation.
    
    Args:
        trades_df: Trades DataFrame with columns:
            - time: Unix timestamp in ms
            - price: Trade price
            - qty: Base quantity
            - quote_qty: Quote quantity
            - is_buyer_maker: Boolean
        interval_ms: Bar interval in milliseconds
        
    Returns:
        OHLCV DataFrame with standard columns
    """
    # Import here to avoid circular imports
    from trades_aggregation.trades_aggregator import aggregate_trades
    
    # Rename columns to match expected format
    trades_renamed = trades_df.rename({
        "qty": "quantity",
        "quote_qty": "quote_quantity"
    })
    
    return aggregate_trades(trades_renamed, interval_ms)
