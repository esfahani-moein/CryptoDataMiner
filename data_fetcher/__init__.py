
"""
Data Fetcher Module
Supports downloading data from Binance repositories with smart monthly/daily selection with priorithy to monthly data.
"""

# Backward compatibility - old API
from .data_info_loader import fetch_available_files, filter_by_date_range
from .data_downloader import download_files_df

# enhanced API
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
    # Old API (backward compatibility)
    'fetch_available_files',
    'filter_by_date_range',
    
    
    # New API
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