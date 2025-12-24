"""
Enhanced Data Fetcher Examples
Demonstrates new features: futures data, multiple symbols, smart date selection, batch downloads
"""

import os
import asyncio
from pathlib import Path
import polars as pl

# Import new enhanced API
from data_fetcher import *

# =============================================================================
# EXAMPLE 1: Download Futures Trades Data
# =============================================================================
async def example_1_futures_trades():
    """Download futures trades data for BTCUSDT"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Futures Trades Data")
    print("="*70)
    
    # Configure download
    config = DataConfig(
        symbol="BTCUSDT",
        data_type=DataType.TRADES,
        market=Market.FUTURES_UM,
        start_date="2024-01-01",
        end_date="2024-01-31"
    )
    
    # Fetch file list with smart monthly/daily selection
    files_df = fetch_and_combine_smart(config)
    
    if files_df.is_empty():
        print("No files found!")
        return
    
    # Download files
    download_folder = Path("dataset/futures_trades_BTCUSDT")
    stats = await download_files_df(
        files_df,
        str(download_folder),
        max_concurrent=5,
        max_retries=3
    )
    
    print(f"\nDownload stats: {stats}")


# =============================================================================
# EXAMPLE 2: Download Futures BookTicker Data
# =============================================================================
async def example_2_futures_bookticker():
    """Download futures bookTicker data"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Futures BookTicker Data")
    print("="*70)
    
    config = DataConfig(
        symbol="ETHUSDT",
        data_type=DataType.BOOK_TICKER,
        market=Market.FUTURES_UM,
        start_date="2024-06-01",
        end_date="2024-06-30"
    )
    
    files_df = fetch_and_combine_smart(config)
    
    if files_df.is_empty():
        print("No files found!")
        return
    
    download_folder = Path("dataset/futures_bookticker_ETHUSDT")
    stats = await download_files_df(files_df, str(download_folder), max_concurrent=5)
    
    print(f"\nDownload stats: {stats}")


# =============================================================================
# EXAMPLE 3: Smart Monthly/Daily Selection
# =============================================================================
async def example_3_smart_date_selection():
    """
    Demonstrate smart monthly/daily selection
    Date range: 2024-05-03 to 2024-11-20
    - May 3-31: Daily
    - June-October: Monthly
    - November 1-20: Daily
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Smart Monthly/Daily Selection")
    print("="*70)
    
    config = DataConfig(
        symbol="SOLUSDT",
        data_type=DataType.KLINES,
        market=Market.SPOT,
        interval="1h",
        start_date="2024-05-03",
        end_date="2024-11-20"
    )
    
    # This will automatically use monthly for full months, daily for partial months
    files_df = fetch_and_combine_smart(config)
    
    if files_df.is_empty():
        print("No files found!")
        return
    
    # Show breakdown by frequency
    print("\nFiles by frequency:")
    print(files_df.group_by('frequency').agg([
        pl.count('filename').alias('count'),
        (pl.col('size_bytes').sum() / (1024**2)).alias('size_mb')
    ]))
    
    download_folder = Path("dataset/spot_klines_SOLUSDT_1h")
    stats = await download_files_df(files_df, str(download_folder), max_concurrent=5)
    
    print(f"\nDownload stats: {stats}")


# =============================================================================
# EXAMPLE 4: Multiple Downloads (Batch Processing)
# =============================================================================
async def example_4_batch_download():
    """
    Download multiple symbols and data types concurrently
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Download Multiple Configurations")
    print("="*70)
    
    # Define multiple download configurations
    configs = [
        # BTCUSDT klines
        DataConfig(
            symbol="BTCUSDT",
            data_type=DataType.KLINES,
            market=Market.SPOT,
            interval="1m",
            start_date="2024-08-01",
            end_date="2024-08-31"
        ),
        # SOLUSDT trades
        DataConfig(
            symbol="SOLUSDT",
            data_type=DataType.TRADES,
            market=Market.SPOT,
            start_date="2024-08-01",
            end_date="2024-08-15"
        ),
        # ETHUSDT futures klines
        DataConfig(
            symbol="ETHUSDT",
            data_type=DataType.KLINES,
            market=Market.FUTURES_UM,
            interval="5m",
            start_date="2024-08-01",
            end_date="2024-08-31"
        )
    ]
    
    # Fetch file lists for all configs
    configs_with_dfs = []
    for config in configs:
        print(f"\nFetching files for {config.symbol} - {config.data_type.value}...")
        files_df = fetch_and_combine_smart(config)
        if not files_df.is_empty():
            configs_with_dfs.append((config, files_df))
    
    # Batch download all configurations
    base_folder = "dataset/batch_download"
    results = await batch_download_multiple(
        configs_with_dfs,
        base_folder,
        max_concurrent_per_config=3,
        max_concurrent_configs=2
    )
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH DOWNLOAD SUMMARY")
    print("="*70)
    for (config, _), result in zip(configs_with_dfs, results):
        print(f"\n{config.symbol} - {config.data_type.value} ({config.market.value}):")
        print(f"  Files: {result['successful']}/{result['total_files']}")
        print(f"  Speed: {result['avg_speed_mbps']:.2f} MB/s")
        print(f"  Time: {result['elapsed_time']:.2f}s")


# =============================================================================
# EXAMPLE 5: All Futures Data Types for One Symbol
# =============================================================================
async def example_5_all_futures_data_types():
    """Download multiple data types for same symbol"""
    print("\n" + "="*70)
    print("EXAMPLE 5: All Futures Data Types for BTCUSDT")
    print("="*70)
    
    symbol = "BTCUSDT"
    start_date = "2024-12-01"
    end_date = "2024-12-31"
    
    # Define all futures data types to download
    data_types_configs = [
        (DataType.KLINES, "1m"),       # Klines need interval
        (DataType.TRADES, None),       # Trades don't need interval
        (DataType.AGG_TRADES, None),
        (DataType.BOOK_TICKER, None),
    ]
    
    configs_with_dfs = []
    
    for data_type, interval in data_types_configs:
        config = DataConfig(
            symbol=symbol,
            data_type=data_type,
            market=Market.FUTURES_UM,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        print(f"\nFetching {data_type.value}...")
        files_df = fetch_and_combine_smart(config)
        
        if not files_df.is_empty():
            configs_with_dfs.append((config, files_df))
    
    # Batch download
    base_folder = f"dataset/{symbol}_futures_complete"
    results = await batch_download_multiple(
        configs_with_dfs,
        base_folder,
        max_concurrent_per_config=3,
        max_concurrent_configs=2
    )


# =============================================================================
# EXAMPLE 6: Custom Download with Progress Tracking
# =============================================================================
async def example_6_custom_optimization():
    """Demonstrate custom optimization settings"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Custom Optimization Settings")
    print("="*70)
    
    config = DataConfig(
        symbol="BTCUSDT",
        data_type=DataType.KLINES,
        market=Market.SPOT,
        interval="1s",
        start_date="2024-08-01",
        end_date="2024-08-31"
    )
    
    files_df = fetch_and_combine_smart(config)
    
    if files_df.is_empty():
        print("No files found!")
        return
    
    # Custom optimization: larger chunks, more concurrent downloads
    download_folder = Path("dataset/btc_1s_optimized")
    stats = await download_files_df(
        files_df,
        str(download_folder),
        max_concurrent=10,           # More concurrent downloads
        chunk_size=5 * 1024 * 1024,  # 5MB chunks (larger for big files)
        max_retries=5                # More retries for reliability
    )
    
    print(f"\nDownload stats: {stats}")


# =============================================================================
# Main Function
# =============================================================================
async def main():
    """Run examples"""
    
    print("\n" + "#"*70)
    print("# Enhanced Data Fetcher Examples")
    print("#"*70)
    
    # Uncomment the examples you want to run:
    
    # Example 1: Futures trades
    # await example_1_futures_trades()
    
    # Example 2: Futures bookTicker
    # await example_2_futures_bookticker()
    
    # Example 3: Smart date selection
    # await example_3_smart_date_selection()
    
    # Example 4: Batch download multiple configs
    await example_4_batch_download()
    
    # Example 5: All data types for one symbol
    # await example_5_all_futures_data_types()
    
    # Example 6: Custom optimization
    # await example_6_custom_optimization()
    
    print("\n" + "#"*70)
    print("# Examples Complete!")
    print("#"*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
