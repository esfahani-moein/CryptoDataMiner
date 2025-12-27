
"""
Data Fetcher Module
Supports downloading data from Binance repositories with smart monthly/daily selection with priority to monthly data.
"""

# Enhanced API (recommended)
from .binance_config import (
    DataConfig,
    Market,
    DataType,
    Frequency,
    BinanceDataRepository,
    get_data_type_schema
)
from .data_info_loader_v2 import (
    fetch_files_for_config,
    fetch_and_combine_smart,
    determine_optimal_frequency
)
from .data_downloader_v2 import (
    download_files_df,
    batch_download_multiple,
    DownloadOptimizer
)

__all__ = [
    # Enhanced API
    'DataConfig',
    'Market',
    'DataType',
    'Frequency',
    'BinanceDataRepository',
    'get_data_type_schema',
    'fetch_files_for_config',
    'fetch_and_combine_smart',
    'determine_optimal_frequency',
    'download_files_df',
    'batch_download_multiple',
    'DownloadOptimizer'
]