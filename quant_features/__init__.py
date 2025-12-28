"""
Quant Features Module

A comprehensive feature extraction library for cryptocurrency data analysis.
Designed for quantitative research and machine learning model development.

Key Modules:
- data_loader: Load and merge multiple data sources
- price_features: Technical indicators and price-based features
- volume_features: Volume, order flow, and microstructure features
- sentiment_features: Market sentiment from funding rates, OI, and ratios
- orderbook_features: Order book depth and liquidity features
- labeling: Forward returns, triple barrier, and classification labels

All functions are designed to avoid look-ahead bias by using only past data.
"""

from quant_features.data_loader import (
    load_trades,
    load_metrics,
    load_funding_rate,
    load_book_depth,
    load_klines,
    load_all_data,
    merge_features_to_ohlcv
)

from quant_features.price_features import (
    add_returns,
    add_volatility_features,
    add_momentum_features,
    add_trend_features,
    add_price_patterns,
    add_all_price_features
)

from quant_features.volume_features import (
    add_volume_features,
    add_order_flow_imbalance,
    add_vwap_features,
    add_all_volume_features
)

from quant_features.sentiment_features import (
    add_funding_rate_features,
    add_open_interest_features,
    add_long_short_ratio_features,
    add_all_sentiment_features
)

from quant_features.orderbook_features import (
    add_depth_imbalance,
    add_liquidity_features,
    add_all_orderbook_features
)

from quant_features.labeling import (
    add_forward_returns,
    add_triple_barrier_labels,
    add_trend_labels,
    add_volatility_labels,
    add_meta_labels,
    add_all_labels
)

from quant_features.model_pipeline import (
    prepare_feature_dataset,
    time_series_split,
    purged_kfold_split,
    get_feature_columns,
    remove_highly_correlated,
    prepare_xy,
    train_xgboost_model,
    train_lightgbm_model,
    evaluate_classification,
    evaluate_regression,
    evaluate_trading_metrics,
    get_feature_importance,
    run_full_pipeline
)

__all__ = [
    # Data Loading
    'load_trades',
    'load_metrics',
    'load_funding_rate',
    'load_book_depth',
    'load_klines',
    'load_all_data',
    'merge_features_to_ohlcv',
    
    # Price Features
    'add_returns',
    'add_volatility_features',
    'add_momentum_features',
    'add_trend_features',
    'add_price_patterns',
    'add_all_price_features',
    
    # Volume Features
    'add_volume_features',
    'add_order_flow_imbalance',
    'add_vwap_features',
    'add_all_volume_features',
    
    # Sentiment Features
    'add_funding_rate_features',
    'add_open_interest_features',
    'add_long_short_ratio_features',
    'add_all_sentiment_features',
    
    # Orderbook Features
    'add_depth_imbalance',
    'add_liquidity_features',
    'add_all_orderbook_features',
    
    # Labeling
    'add_forward_returns',
    'add_triple_barrier_labels',
    'add_trend_labels',
    'add_volatility_labels',
    'add_meta_labels',
    'add_all_labels',
    
    # Model Pipeline
    'prepare_feature_dataset',
    'time_series_split',
    'purged_kfold_split',
    'get_feature_columns',
    'remove_highly_correlated',
    'prepare_xy',
    'train_xgboost_model',
    'train_lightgbm_model',
    'evaluate_classification',
    'evaluate_regression',
    'evaluate_trading_metrics',
    'get_feature_importance',
    'run_full_pipeline',
]
