"""
High-accuracy OHLCV data aggregation module for cryptocurrency trading data.

This module provides production-grade aggregation of OHLCV (Open, High, Low, Close, Volume)
data from any timeframe to any higher timeframe, with automatic timestamp format detection
and exchange-agnostic column handling.

Main Features:
    - Automatic source interval detection (no manual specification needed)
    - Automatic timestamp format detection and fixing (including Binance quirks)
    - Exchange-agnostic (works with any data source)
    - High-accuracy aggregation (>99.999%)
    - Functional API (no class instantiation required)
    - Comprehensive validation utilities

Recommended Usage (New Functional API):
    >>> from data_aggregate import aggregate_ohlcv, STANDARD_INTERVALS
    >>> import polars as pl
    >>> 
    >>> # Load your data (any format, any exchange)
    >>> df = pl.read_parquet("BTCUSDT-1s-2024-11.parquet")
    >>> 
    >>> # Aggregate to multiple timeframes at once (auto-detects everything)
    >>> intervals = {k: STANDARD_INTERVALS[k] for k in ['5m', '1h', '4h', '1d']}
    >>> result = aggregate_ohlcv(df, intervals)
    >>> df_5m = result['5m']
    >>> df_1h = result['1h']
    >>> 
    >>> # Or aggregate to a single timeframe
    >>> df_5m = aggregate_ohlcv(df, 300)  # 300 seconds = 5 minutes
    >>> df_1h = aggregate_ohlcv(df, '1h')  # Using interval name

Legacy Usage (Class-based API - still supported):
    >>> from data_aggregate import OHLCVAggregator
    >>> 
    >>> # Create aggregator (requires manual interval specification)
    >>> aggregator = OHLCVAggregator(source_interval_seconds=1)
    >>> df_5m = aggregator.aggregate(df, target_interval_seconds=300)
"""

# Main functional API (recommended)
from data_aggregate.aggregator import aggregate_ohlcv, detect_source_interval

# Legacy class-based API (still supported)
from data_aggregate.aggregator import OHLCVAggregator

# Validation utilities
from data_aggregate.validators import ValidationResult, compare_dataframes

# Timestamp utilities (for advanced use)
from data_aggregate.timestamp_utils import (
    TimestampFormat, 
    detect_timestamp_format,
    fix_binance_timestamp_display,
    normalize_timestamp_to_datetime,
    restore_original_timestamp_format
)

# Configuration
from data_aggregate.config import DEFAULT_OHLCV_COLUMNS, STANDARD_INTERVALS

__version__ = "2.0.0"

__all__ = [
    # Main functional API
    "aggregate_ohlcv",
    "detect_source_interval",
    
    # Legacy class-based API
    "OHLCVAggregator",
    
    # Validation
    "ValidationResult",
    "compare_dataframes",
    
    # Timestamp utilities
    "TimestampFormat",
    "detect_timestamp_format",
    "fix_binance_timestamp_display",
    "normalize_timestamp_to_datetime",
    "restore_original_timestamp_format",
    
    # Configuration
    "DEFAULT_OHLCV_COLUMNS",
    "STANDARD_INTERVALS",
]
