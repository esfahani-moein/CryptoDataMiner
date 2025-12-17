"""
Configuration constants and default settings for data aggregation.
"""

from typing import Dict, List

# Default OHLCV column names (standard across most exchanges)
DEFAULT_OHLCV_COLUMNS = {
    'timestamp': 'open_time',
    'close_timestamp': 'close_time',
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume': 'volume',
}

# Optional columns that may be present in exchange data
OPTIONAL_COLUMNS = [
    'quote_asset_volume',
    'number_of_trades',
    'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume',
    'ignore',
]

# Standard interval definitions in seconds
STANDARD_INTERVALS: Dict[str, int] = {
    '1s': 1,
    '5s': 5,
    '15s': 15,
    '30s': 30,
    '1m': 60,
    '3m': 180,
    '5m': 300,
    '15m': 900,
    '30m': 1800,
    '1h': 3600,
    '2h': 7200,
    '4h': 14400,
    '6h': 21600,
    '8h': 28800,
    '12h': 43200,
    '1d': 86400,
    '3d': 259200,
    '1w': 604800,
}

# Default validation tolerance for floating-point comparisons
DEFAULT_TOLERANCE = 1e-10

# Timestamp format detection thresholds
MILLISECOND_THRESHOLD = 1e12  # Unix timestamps > 1e12 are likely milliseconds
MICROSECOND_THRESHOLD = 1e13  # Binance quirk: datetime[μs] with ms values (~1.7e12 for 2024)
