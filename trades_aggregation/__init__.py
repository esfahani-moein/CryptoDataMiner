"""
Trades Aggregation Module

High-performance module for aggregating cryptocurrency trades data 
into OHLCV (Open, High, Low, Close, Volume) format at various timeframes.
"""

from .trades_aggregator import (
    aggregate_trades_to_ohlcv,
    aggregate_trades_from_file,
    validate_aggregation_accuracy,
    print_validation_results,
    TimeInterval
)

__all__ = [
    'aggregate_trades_to_ohlcv',
    'aggregate_trades_from_file',
    'validate_aggregation_accuracy',
    'print_validation_results',
    'TimeInterval'
]
