"""
Imbalance Bars Aggregation Module

This module provides high-performance functions to aggregate cryptocurrency trades
into Imbalance Bars - advanced volume-based sampling methods that detect order flow
imbalance in the market.

Key Features:
- Tick Imbalance Bars: Bars based on trade count imbalance
- Dollar Imbalance Bars: Bars based on dollar volume imbalance
- Dynamic thresholds using EMA
- Proper handling of look-ahead bias
- Pure Polars implementation for maximum speed

All timestamps remain in Unix milliseconds for efficiency.
"""

from .imbalance_bars_aggregator import (
    aggregate_trades_to_tick_imbalance_bars,
    aggregate_trades_to_dollar_imbalance_bars,
    calculate_tick_rule,
    TimeInterval,
)

__all__ = [
    'aggregate_trades_to_tick_imbalance_bars',
    'aggregate_trades_to_dollar_imbalance_bars',
    'calculate_tick_rule',
    'TimeInterval',
]
