"""
Trades Aggregation Module

High-performance module for aggregating cryptocurrency trades data 
into OHLCV (Open, High, Low, Close, Volume) format at various timeframes.
"""

from .trades_time_aggregate import (
    aggregate_trades_to_ohlcv,
    TimeInterval
)

__all__ = [
    'aggregate_trades_to_ohlcv',
    'TimeInterval'
]
