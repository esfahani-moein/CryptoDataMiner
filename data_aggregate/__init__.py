"""
High-accuracy OHLCV data aggregation module for cryptocurrency trading data.

This module provides production-grade aggregation of OHLCV (Open, High, Low, Close, Volume)
data from any timeframe to any higher timeframe, with automatic timestamp format detection
and exchange-agnostic column handling.

Main Features:
    - Automatic timestamp format detection (ms, μs, seconds, datetime)
    - Exchange-agnostic (works with any data source)
    - High-accuracy aggregation (>99.999%)
    - Flexible column mapping
    - Comprehensive validation utilities

Example:
    >>> from data_aggregate import OHLCVAggregator
    >>> import polars as pl
    >>> 
    >>> # Load your data
    >>> df = pl.read_parquet("BTCUSDT-1s-2024-11.parquet")
    >>> 
    >>> # Create aggregator
    >>> aggregator = OHLCVAggregator(source_interval_seconds=1)
    >>> 
    >>> # Aggregate to 5 minutes
    >>> df_5m = aggregator.aggregate(df, target_interval_seconds=300)
    >>> 
    >>> # Aggregate to 1 hour
    >>> df_1h = aggregator.aggregate(df, target_interval_seconds=3600)
"""

from data_aggregate.aggregator import OHLCVAggregator
from data_aggregate.validators import ValidationResult, compare_dataframes
from data_aggregate.timestamp_utils import (
    TimestampFormat, 
    detect_timestamp_format,
    fix_binance_timestamp_display
)
from data_aggregate.config import DEFAULT_OHLCV_COLUMNS, STANDARD_INTERVALS

__version__ = "1.0.0"

__all__ = [
    "OHLCVAggregator",
    "ValidationResult",
    "compare_dataframes",
    "TimestampFormat",
    "detect_timestamp_format",
    "fix_binance_timestamp_display",
    "DEFAULT_OHLCV_COLUMNS",
    "STANDARD_INTERVALS",
]
