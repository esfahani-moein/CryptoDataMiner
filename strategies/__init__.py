"""
Strategies Package
==================

Collection of quantitative trading strategies for cryptocurrency analysis.
Each strategy explores different combinations of features, timeframes, and ML models.
"""

from .strategy_base import (
    StrategyBase,
    StrategyResult,
    remove_correlated_features,
    save_results,
)

__all__ = [
    'StrategyBase',
    'StrategyResult',
    'remove_correlated_features',
    'save_results',
]
