"""
Data Processing Pipeline for SLURM
Optimized for batch processing of Binance historical data
"""

import os
import sys
from pathlib import Path
import data_fetcher
from src import *
import polars as pl
from data_fetcher import download_files_df
from data_processor import *
import asyncio

async def main():

    symbol = "BTCUSDT"
    interval = "1s"
    start_date = "2024-08-01"
    end_date = "2024-12-31"
    data_type = "spot"
    frequency = "monthly" # or "monthly" or "daily"

    num_cpu = os.cpu_count()
    print(f"Number of CPU cores available: {num_cpu}")
    num_workers = 5
    num_concurrent_downloads = 5

    # directories
    base_dir = Path.cwd()
    dataset_dir = base_dir / f'dataset_{symbol}_{interval}'
    zip_dir = dataset_dir / f'files_zip_{symbol}_{interval}'
    parquet_dir = dataset_dir / f'processed_parquet_{symbol}_{interval}'
    
    # Create directories
    dataset_dir.mkdir(exist_ok=True)
    zip_dir.mkdir(exist_ok=True)
    parquet_dir.mkdir(exist_ok=True)


    base_url = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
    prefix = f"data/{data_type}/{frequency}/klines/{symbol}/{interval}/"
    files_df, start_date_fetch, end_date_fetch = data_fetcher.fetch_available_files(base_url, prefix, symbol, interval)

    files_df_filtered = data_fetcher.filter_by_date_range(files_df, str(start_date_fetch), str(end_date_fetch))

    # Set File Info
    file_name = f"info_binance_{data_type}_{frequency}_{symbol}_{interval}_files"
    save_df_to_parquet(files_df, dataset_dir, file_name)

    # Download Files
    await download_files_df(files_df_filtered, zip_dir, max_concurrent=num_concurrent_downloads)

    # Process Downloaded Files
    dfs = process_downloaded_files(zip_dir, files_df_filtered, data_type='klines', save_folder=parquet_dir, max_workers=num_workers)


if __name__ == "__main__":
    asyncio.run(main())