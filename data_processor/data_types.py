

import polars as pl

def get_data_config(data_type: str) -> dict:
    """
    Get configuration for a specific data type.
    
    Args:
        data_type: Type of data (e.g., 'klines')
    
    Returns:
        Dict with 'columns', 'dtypes', 'timestamp_cols', 'time_unit'
    """
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
            },
            'timestamp_cols': ['open_time', 'close_time'],
            'time_unit': 'us'  # Microseconds for SPOT data from 2025
        }
    # Add other data types here, e.g., 'trades': {...}
    else:
        raise ValueError(f"Unknown data_type: {data_type}")