"""
Binance Futures Bulk Data Downloader (Using Existing Modules)
==============================================================
Production script that integrates existing data_fetcher and data_processor modules.

Features:
- Uses existing modules for consistency
- Structured partitioning: dataset/dataset_{SYMBOL}/{YEAR}_{MONTH}/{data_type}/
- Supports all 7 data types from Binance futures
- Handles monthly and daily data automatically
- Unix timestamp integers (Int64)
- Float64 for monetary values

Usage:
    python bulk_download_integrated.py --symbol BTCUSDT --start-date 2025-11-01 --end-date 2025-11-30
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List
import logging

# Import existing modules
from data_fetcher.binance_config import DataConfig, DataType, Market
from data_fetcher.data_downloader_v2 import download_files_df
from data_processor.bulk_processor import (
    process_zip_to_parquet, process_csv_to_parquet,
    combine_daily_to_monthly, get_output_path,
    get_data_frequency, get_file_extension
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration for each data type
DATA_TYPE_CONFIG = {
    'trades': {'name': 'trades', 'interval': None},
    'fundingRate': {'name': 'fundingRate', 'interval': None},
    'metrics': {'name': 'metrics', 'interval': None},
    'bookDepth': {'name': 'bookDepth', 'interval': None},
    'premiumIndexKlines': {'name': 'premiumIndexKlines', 'interval': '1m'},
    'markPriceKlines': {'name': 'markPriceKlines', 'interval': '1m'},
    'indexPriceKlines': {'name': 'indexPriceKlines', 'interval': '1m'},
}


def generate_date_list(start_date: str, end_date: str, frequency: str) -> List[str]:
    """
    Generate list of dates based on frequency
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        frequency: 'monthly' or 'daily'
        
    Returns:
        List of date strings in appropriate format
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    dates = []
    
    if frequency == 'monthly':
        current = start.replace(day=1)
        while current <= end:
            dates.append(current.strftime('%Y-%m'))
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
    else:  # daily
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
    
    return dates


def build_file_url(symbol: str, data_type_name: str, date_str: str, interval: str = None) -> tuple:
    """
    Build Binance data URL and filename
    
    Returns:
        Tuple of (url, filename)
    """
    base_url = "https://data.binance.vision"
    market = "futures/um"
    frequency = get_data_frequency(data_type_name)
    file_ext = get_file_extension(data_type_name)
    
    # Build path components
    if interval:
        path = f"data/{market}/{frequency}/{data_type_name}/{symbol}/{interval}"
    else:
        path = f"data/{market}/{frequency}/{data_type_name}/{symbol}"
    
    # Build filename
    if frequency == 'monthly':
        if interval:
            filename = f"{symbol}-{interval}-{date_str}.{file_ext}"
        else:
            filename = f"{symbol}-{data_type_name}-{date_str}.{file_ext}"
    else:  # daily
        if interval:
            filename = f"{symbol}-{interval}-{date_str}.{file_ext}"
        else:
            filename = f"{symbol}-{data_type_name}-{date_str}.{file_ext}"
    
    url = f"{base_url}/{path}/{filename}"
    return url, filename


async def download_and_process_datatype(
    symbol: str,
    data_type_name: str,
    start_date: str,
    end_date: str,
    base_dir: Path,
    max_concurrent: int = 5
) -> dict:
    """
    Download and process all files for a specific data type using existing modules
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing {data_type_name} for {symbol}")
    logger.info(f"{'='*70}")
    
    config = DATA_TYPE_CONFIG.get(data_type_name)
    if not config:
        logger.error(f"Unknown data type: {data_type_name}")
        return {'data_type': data_type_name, 'status': 'failed'}
    
    frequency = get_data_frequency(data_type_name)
    file_ext = get_file_extension(data_type_name)
    
    logger.info(f"Frequency: {frequency}")
    
    # Generate dates
    dates = generate_date_list(start_date, end_date, frequency)
    logger.info(f"Processing {len(dates)} {frequency} periods")
    
    # Create temp directory
    temp_dir = base_dir / 'temp_downloads' / data_type_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Build list of files to download
    import polars as pl
    download_info = []
    
    for date_str in dates:
        url, filename = build_file_url(symbol, data_type_name, date_str, config['interval'])
        filepath = temp_dir / filename
        
        # Skip if already exists
        if filepath.exists():
            logger.info(f"⊙ Skipping {filename} (already exists)")
            continue
        
        download_info.append({
            'url': url,
            'filename': filename,
            'size_bytes': 0,  # Unknown until download
            'date': date_str
        })
    
    # Download files using existing module
    if download_info:
        df = pl.DataFrame(download_info)
        logger.info(f"Downloading {len(df)} files...")
        
        try:
            await download_files_df(df, str(temp_dir), max_concurrent=max_concurrent)
        except Exception as e:
            logger.error(f"Download error: {e}")
    
    # Process files
    logger.info(f"\nProcessing downloaded files...")
    
    if frequency == 'monthly':
        # Process each monthly file directly
        for date_str in dates:
            _, filename = build_file_url(symbol, data_type_name, date_str, config['interval'])
            downloaded_path = temp_dir / filename
            
            if not downloaded_path.exists():
                continue
            
            # Create output path with new structure
            output_path = get_output_path(symbol, data_type_name, date_str, str(base_dir), filename)
            
            # Skip if already processed
            if output_path.exists():
                logger.info(f"⊙ Skipping {output_path.name} (already processed)")
                downloaded_path.unlink(missing_ok=True)
                continue
            
            # Process based on file type
            if file_ext == 'zip':
                success = process_zip_to_parquet(
                    downloaded_path, output_path, data_type_name, symbol, temp_dir
                )
            else:  # csv
                success = process_csv_to_parquet(
                    downloaded_path, output_path, data_type_name, symbol
                )
            
            if success:
                downloaded_path.unlink(missing_ok=True)
    
    else:  # daily - need to combine by month
        # Group daily files by month
        monthly_groups = {}
        for date_str in dates:
            year_month = date_str[:7]  # YYYY-MM
            if year_month not in monthly_groups:
                monthly_groups[year_month] = []
            
            _, filename = build_file_url(symbol, data_type_name, date_str, config['interval'])
            downloaded_path = temp_dir / filename
            
            if downloaded_path.exists():
                monthly_groups[year_month].append((date_str, downloaded_path))
        
        # Process each month
        for year_month, daily_items in monthly_groups.items():
            logger.info(f"\nCombining {len(daily_items)} daily files for {year_month}...")
            
            # Create output path for combined monthly file
            year_month_formatted = year_month.replace('-', '_')
            output_dir = base_dir / f"dataset_{symbol}" / year_month_formatted / data_type_name
            output_filename = f"{symbol}-{data_type_name}-{year_month}.parquet"
            output_path = output_dir / output_filename
            
            # Skip if already processed
            if output_path.exists():
                logger.info(f"⊙ Skipping {year_month} (already processed)")
                for _, daily_path in daily_items:
                    daily_path.unlink(missing_ok=True)
                continue
            
            # Process daily files to temp parquet files
            temp_parquet_files = []
            for date_str, downloaded_path in daily_items:
                temp_parquet = temp_dir / f"{downloaded_path.stem}.parquet"
                
                if file_ext == 'zip':
                    success = process_zip_to_parquet(
                        downloaded_path, temp_parquet, data_type_name, symbol, temp_dir
                    )
                else:  # csv
                    success = process_csv_to_parquet(
                        downloaded_path, temp_parquet, data_type_name, symbol
                    )
                
                if success:
                    temp_parquet_files.append(temp_parquet)
                
                downloaded_path.unlink(missing_ok=True)
            
            # Combine daily parquet files into monthly
            if temp_parquet_files:
                combine_daily_to_monthly(temp_parquet_files, output_path, data_type_name)
    
    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    logger.info(f"✓ Completed processing {data_type_name}")
    
    return {'data_type': data_type_name, 'status': 'completed'}


async def bulk_download(
    symbol: str,
    start_date: str,
    end_date: str,
    data_types: List[str],
    output_dir: str = 'dataset',
    max_concurrent: int = 5
):
    """
    Main function to download and process all specified data types
    """
    logger.info(f"\n{'#'*70}")
    logger.info(f"BINANCE FUTURES BULK DOWNLOADER (INTEGRATED)")
    logger.info(f"{'#'*70}")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Data types: {', '.join(data_types)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'#'*70}\n")
    
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each data type sequentially
    results = []
    for data_type_name in data_types:
        if data_type_name not in DATA_TYPE_CONFIG:
            logger.error(f"Unknown data type: {data_type_name}")
            continue
        
        result = await download_and_process_datatype(
            symbol, data_type_name, start_date, end_date, base_dir, max_concurrent
        )
        results.append(result)
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"BULK DOWNLOAD COMPLETED")
    logger.info(f"{'#'*70}")
    for result in results:
        logger.info(f"  {result['data_type']}: {result['status']}")
    logger.info(f"{'#'*70}\n")


def main():
#     """Command-line interface"""
#     parser = argparse.ArgumentParser(
#         description='Binance Futures Bulk Downloader (Integrated with existing modules)',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Download all data types for November 2025
#   python bulk_download_integrated.py --symbol BTCUSDT --start-date 2025-11-01 --end-date 2025-11-30
  
#   # Download specific data types
#   python bulk_download_integrated.py --symbol BTCUSDT --start-date 2025-11-01 --end-date 2025-11-30 \\
#     --data-types trades fundingRate
#         """
#     )
    
    # parser.add_argument('--symbol', type=str, required=True,
    #                     help='Trading symbol (e.g., BTCUSDT)')
    # parser.add_argument('--start-date', type=str, required=True,
    #                     help='Start date in YYYY-MM-DD format')
    # parser.add_argument('--end-date', type=str, required=True,
    #                     help='End date in YYYY-MM-DD format')
    # parser.add_argument('--data-types', nargs='+',
    #                     default=['trades', 'metrics', 'fundingRate', 'bookDepth',
    #                             'premiumIndexKlines', 'markPriceKlines', 'indexPriceKlines'],
    #                     choices=list(DATA_TYPE_CONFIG.keys()),
    #                     help='Data types to download (default: all)')
    # parser.add_argument('--output-dir', type=str, default='dataset',
    #                     help='Output directory (default: dataset)')
    # parser.add_argument('--max-concurrent', type=int, default=5,
    #                     help='Maximum concurrent downloads (default: 5)')
    
    # args = parser.parse_args()
    
    # Run the async main function
    # asyncio.run(bulk_download(
    #     symbol=args.symbol,
    #     start_date=args.start_date,
    #     end_date=args.end_date,
    #     data_types=args.data_types,
    #     output_dir=args.output_dir,
    #     max_concurrent=args.max_concurrent
    # ))

    # data_types=["trades", "metrics", "fundingRate", "bookDepth",
    #                 "premiumIndexKlines", "markPriceKlines", "indexPriceKlines"],

    asyncio.run(bulk_download(
        symbol="BTCUSDT",
        start_date="2025-10-01",
        end_date="2025-10-30",
        data_types=["metrics", "fundingRate", "bookDepth",
                    "premiumIndexKlines", "markPriceKlines", "indexPriceKlines"],
        output_dir="dataset",
        max_concurrent=7
    ))


if __name__ == '__main__':
    main()
