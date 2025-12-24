"""
Binance Data Repository Configuration
Defines supported data types, markets, and URL patterns
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class Market(Enum):
    """Supported market types"""
    SPOT = "spot"
    FUTURES_UM = "futures/um"  # USD-M Futures
    FUTURES_CM = "futures/cm"  # COIN-M Futures


class Frequency(Enum):
    """Data frequency types"""
    DAILY = "daily"
    MONTHLY = "monthly"


class DataType(Enum):
    """Supported data types"""
    KLINES = "klines"
    TRADES = "trades"
    AGG_TRADES = "aggTrades"
    BOOK_TICKER = "bookTicker"
    BOOK_DEPTH = "bookDepth"
    METRICS = "metrics"
    PREMIUM_INDEX_KLINES = "premiumIndexKlines"
    MARK_PRICE_KLINES = "markPriceKlines"
    INDEX_PRICE_KLINES = "indexPriceKlines"
    LIQUIDATION_SNAPSHOT = "liquidationSnapshot"


@dataclass
class DataConfig:
    """Configuration for a specific data download request"""
    symbol: str
    data_type: DataType
    market: Market
    start_date: str  # YYYY-MM-DD format
    end_date: str  # YYYY-MM-DD format
    interval: Optional[str] = None  # Required for klines (1s, 1m, 1h, 1d, etc.)
    
    def __post_init__(self):
        """Validate configuration"""
        # Convert string enums to Enum if needed
        if isinstance(self.data_type, str):
            self.data_type = DataType(self.data_type)
        if isinstance(self.market, str):
            self.market = Market(self.market)
        
        # Validate interval for klines
        if self.data_type in [DataType.KLINES, DataType.PREMIUM_INDEX_KLINES, 
                              DataType.MARK_PRICE_KLINES, DataType.INDEX_PRICE_KLINES]:
            if not self.interval:
                raise ValueError(f"{self.data_type.value} requires an interval (e.g., '1m', '1h', '1d')")


class BinanceDataRepository:
    """Central configuration for Binance data repository"""
    
    BASE_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
    
    # Data types that require interval parameter
    INTERVAL_REQUIRED = {
        DataType.KLINES,
        DataType.PREMIUM_INDEX_KLINES,
        DataType.MARK_PRICE_KLINES,
        DataType.INDEX_PRICE_KLINES
    }
    
    # Data types available for each market
    MARKET_DATA_TYPES = {
        Market.SPOT: {
            DataType.KLINES,
            DataType.TRADES,
            DataType.AGG_TRADES,
            DataType.BOOK_TICKER,
            DataType.BOOK_DEPTH
        },
        Market.FUTURES_UM: {
            DataType.KLINES,
            DataType.TRADES,
            DataType.AGG_TRADES,
            DataType.BOOK_TICKER,
            DataType.BOOK_DEPTH,
            DataType.METRICS,
            DataType.PREMIUM_INDEX_KLINES,
            DataType.MARK_PRICE_KLINES,
            DataType.INDEX_PRICE_KLINES,
            DataType.LIQUIDATION_SNAPSHOT
        },
        Market.FUTURES_CM: {
            DataType.KLINES,
            DataType.TRADES,
            DataType.AGG_TRADES,
            DataType.BOOK_TICKER,
            DataType.BOOK_DEPTH,
            DataType.METRICS,
            DataType.PREMIUM_INDEX_KLINES,
            DataType.MARK_PRICE_KLINES,
            DataType.INDEX_PRICE_KLINES,
            DataType.LIQUIDATION_SNAPSHOT
        }
    }
    
    @classmethod
    def build_prefix(cls, config: DataConfig, frequency: Frequency) -> str:
        """
        Build S3 prefix path based on configuration
        
        Args:
            config: DataConfig instance
            frequency: Daily or monthly frequency
            
        Returns:
            S3 prefix path
            
        Examples:
            spot: data/spot/monthly/klines/BTCUSDT/1m/
            futures: data/futures/um/monthly/trades/BTCUSDT/
        """
        parts = ["data", config.market.value, frequency.value, config.data_type.value, config.symbol]
        
        # Add interval if required for this data type
        if config.data_type in cls.INTERVAL_REQUIRED:
            parts.append(config.interval)
        
        return "/".join(parts) + "/"
    
    @classmethod
    def validate_config(cls, config: DataConfig) -> None:
        """
        Validate that data type is available for the specified market
        
        Raises:
            ValueError: If configuration is invalid
        """
        if config.data_type not in cls.MARKET_DATA_TYPES.get(config.market, set()):
            raise ValueError(
                f"{config.data_type.value} is not available for {config.market.value} market. "
                f"Available types: {[dt.value for dt in cls.MARKET_DATA_TYPES[config.market]]}"
            )


def get_data_type_schema(data_type: DataType) -> Dict:
    """
    Get schema configuration for data processing
    
    Args:
        data_type: Type of data
        
    Returns:
        Dict with columns, dtypes, timestamp_cols, time_unit
    """
    import polars as pl
    
    schemas = {
        DataType.KLINES: {
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
            'time_unit': 'ms'
        },
        DataType.TRADES: {
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
            'time_unit': 'ms'
        },
        DataType.AGG_TRADES: {
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
            'time_unit': 'ms'
        },
        DataType.BOOK_TICKER: {
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
            'time_unit': 'ms'
        }
    }
    
    if data_type not in schemas:
        raise ValueError(f"Schema not defined for {data_type.value}")
    
    return schemas[data_type]
