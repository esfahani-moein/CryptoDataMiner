"""
Dollar Bars Aggregation Module

This module provides high-performance functions to aggregate cryptocurrency trades
into Dollar Bars - a volume-based sampling method that creates bars based on 
dollar volume traded rather than time.

Key Features:
- Fixed threshold Dollar Bars
- Dynamic threshold Dollar Bars with EMA adaptation
- Pure Polars implementation for maximum speed
- Proper residual handling for high accuracy
- Unix millisecond timestamp support

All timestamps remain in Unix milliseconds for efficiency.
"""

from .dollar_bars_aggregator import (
    aggregate_trades_to_dollar_bars_fixed,
    aggregate_trades_to_dollar_bars_dynamic,
    calculate_daily_dollar_volume,
    calculate_ema_daily_volume,
    calculate_dynamic_threshold,
    TimeInterval,
)

__all__ = [
    'aggregate_trades_to_dollar_bars_fixed',
    'aggregate_trades_to_dollar_bars_dynamic',
    'calculate_daily_dollar_volume',
    'calculate_ema_daily_volume',
    'calculate_dynamic_threshold',
    'TimeInterval',
]
