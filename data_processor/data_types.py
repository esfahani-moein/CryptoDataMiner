

import polars as pl
import warnings

# DEPRECATION WARNING
warnings.warn(
    "data_processor/data_types.py is deprecated. "
    "Use data_fetcher.binance_config.get_data_type_schema() instead.",
    DeprecationWarning,
    stacklevel=2
)

def get_data_config(data_type: str) -> dict:
    """
    Get configuration for a specific data type.
    
    Args:
        data_type: Type of data (e.g., 'klines', 'trades')
    
    Returns:
        Dict with 'columns', 'dtypes', 'timestamp_cols', 'time_unit'
    """
    if data_type == 'klines_v1':
        return {
            'columns': [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ],
            'dtypes': {
                'open_time': pl.Int64,
                'open': pl.Float64,
                'high': pl.Float64,
                'low': pl.Float64,
                'close': pl.Float64,
                'volume': pl.Float64,
                'close_time': pl.Int64,
                'quote_asset_volume': pl.Float64,
                'number_of_trades': pl.Int64,
                'taker_buy_base_asset_volume': pl.Float64,
                'taker_buy_quote_asset_volume': pl.Float64,
                'ignore': pl.Int64
            },
            'timestamp_cols': ['open_time', 'close_time'],
            'time_unit': 'ms'  # Milliseconds for timestamp
        }
    if data_type == 'klines':
        return {
            'columns': [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ],
            'dtypes': {
                'open_time': pl.Int64,
                'open': pl.Float64,
                'high': pl.Float64,
                'low': pl.Float64,
                'close': pl.Float64,
                'volume': pl.Float64,
                'close_time': pl.Int64,
                'quote_asset_volume': pl.Float64,
                'number_of_trades': pl.Int64,
                'taker_buy_base_asset_volume': pl.Float64,
                'taker_buy_quote_asset_volume': pl.Float64,
                'ignore': pl.Int64
            }
        }
    elif data_type == 'trades_v1':
        return {
            'columns': [
                'trade_id', 'price', 'quantity', 'quote_quantity',
                'time', 'is_buyer_maker'
            ],
            'dtypes': {
                'trade_id': pl.Int64,
                'price': pl.Float64,
                'quantity': pl.Float64,
                'quote_quantity': pl.Float64,
                'time': pl.Int64,
                'is_buyer_maker': pl.Boolean
            },
            'timestamp_cols': ['time'],
            'time_unit': 'ms'  # Milliseconds for timestamp
        }
    elif data_type == 'trades':
        return {
            'columns': [
                'trade_id', 'price', 'quantity', 'quote_quantity',
                'time', 'is_buyer_maker'
            ],
            'dtypes': {
                'trade_id': pl.Int64,
                'price': pl.Float64,
                'quantity': pl.Float64,
                'quote_quantity': pl.Float64,
                'time': pl.Int64,
                'is_buyer_maker': pl.Boolean
            }
        }
    elif data_type == 'aggTrades':
        return {
            'columns': [
                'agg_trade_id', 'price', 'quantity', 'first_trade_id',
                'last_trade_id', 'timestamp', 'is_buyer_maker'
            ],
            'dtypes': {
                'agg_trade_id': pl.Int64,
                'price': pl.Float64,
                'quantity': pl.Float64,
                'first_trade_id': pl.Int64,
                'last_trade_id': pl.Int64,
                'timestamp': pl.Int64,
                'is_buyer_maker': pl.Boolean
            },
            'timestamp_cols': ['timestamp'],
            'time_unit': 'ms'  # Milliseconds for timestamp
        }
    elif data_type == 'bookTicker':
        return {
            'columns': [
                'update_id', 'best_bid_price', 'best_bid_qty',
                'best_ask_price', 'best_ask_qty', 'transaction_time', 'event_time'
            ],
            'dtypes': {
                'update_id': pl.Int64,
                'best_bid_price': pl.Float64,
                'best_bid_qty': pl.Float64,
                'best_ask_price': pl.Float64,
                'best_ask_qty': pl.Float64,
                'transaction_time': pl.Int64,
                'event_time': pl.Int64
            },
            'timestamp_cols': ['transaction_time', 'event_time'],
            'time_unit': 'ms'  # Milliseconds for timestamp
        }
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Supported types: 'klines', 'trades', 'aggTrades', 'bookTicker'")