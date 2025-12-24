"""
Data Processing Pipeline
Downloads and processes Binance futures data (klines and trades)
Uses the new enhanced data fetcher with smart monthly/daily selection
"""

import os
from pathlib import Path
import polars as pl
import asyncio
from data_fetcher import (
    DataConfig,
    DataType,
    Market,
    fetch_and_combine_smart,
    batch_download_multiple
)
from data_processor import process_downloaded_files


async def main():
    """
    Download and process data:
    """
    
    # Configuration
    symbol = "BTCUSDT"
    start_date = "2025-11-01"
    end_date = "2025-11-30"
    klines_interval = "1m"
    
    num_cpu = os.cpu_count()
    print(f"\n{'='*70}")
    print(f"Number of CPU cores available: {num_cpu}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*70}\n")
    
    num_workers = min(10, num_cpu) if num_cpu else 5
    num_concurrent_downloads = 10

    # Setup directories
    base_dir = Path.cwd()
    dataset_dir = base_dir / 'dataset'/ f'dataset_{symbol}'
    
    # Create main dataset directory
    dataset_dir.mkdir(exist_ok=True, parents=True)
    
    # ========================================================================
    # STEP 1: Configure Downloads (Klines + Trades)
    # ========================================================================
    print("\n" + "="*70)
    print("Configuring Downloads")
    print("="*70)
    
    configs = [
        
        DataConfig(
            symbol=symbol,
            data_type=DataType.KLINES,
            market=Market.FUTURES_UM,
            interval=klines_interval,
            start_date=start_date,
            end_date=end_date
        ),

        DataConfig(
            symbol=symbol,
            data_type=DataType.KLINES,
            market=Market.SPOT,
            interval=klines_interval,
            start_date=start_date,
            end_date=end_date
        ),
        
        DataConfig(
            symbol=symbol,
            data_type=DataType.TRADES,
            market=Market.FUTURES_UM,
            start_date=start_date,
            end_date=end_date
        )
    ]
    
    print(f"Configured {len(configs)} download tasks:")
    for config in configs:
        if config.interval:
            print(f"  - {config.symbol} {config.data_type.value} ({config.interval})")
        else:
            print(f"  - {config.symbol} {config.data_type.value}")
    
    # ========================================================================
    # STEP 2: Fetch File Lists (Smart Monthly/Daily Selection)
    # ========================================================================
    print("\n" + "="*70)
    print("Fetching File Lists")
    print("="*70)
    
    configs_with_dfs = []
    for config in configs:
        print(f"\nFetching files for {config.symbol} {config.data_type.value}...")
        files_df = fetch_and_combine_smart(config)
        
        if not files_df.is_empty():
            configs_with_dfs.append((config, files_df))
            print(f"Found {len(files_df)} files for {config.symbol} {config.data_type.value}")
        else:
            print(f"No files found for {config.symbol} {config.data_type.value}")
    
    if not configs_with_dfs:
        print("\n[!] No files to download. Exiting.")
        return
    
    # ========================================================================
    # STEP 3: Check Existing Files and Download Missing Ones
    # ========================================================================
    print("\n" + "="*70)
    print("Checking Existing Files and Downloading Missing Ones")
    print("="*70)
    
    # Filter out already downloaded files
    configs_with_filtered_dfs = []
    for config, files_df in configs_with_dfs:
        # Determine download directory (replace slashes to prevent subfolders)
        market_name = config.market.value.replace('/', '_')
        folder_name = f"{config.symbol}_{config.data_type.value}_{market_name}"
        if config.interval:
            folder_name += f"_{config.interval}"
        
        download_dir = dataset_dir / folder_name
        
        # Check which files already exist with correct size
        if download_dir.exists():
            existing_files = set()
            files_to_download = []
            
            for row in files_df.iter_rows(named=True):
                filename = row['filename']
                expected_size = row['size_bytes']
                file_path = download_dir / filename
                
                if file_path.exists():
                    actual_size = file_path.stat().st_size
                    if actual_size == expected_size:
                        existing_files.add(filename)
                    else:
                        files_to_download.append(row)
                else:
                    files_to_download.append(row)
            
            if existing_files:
                print(f"\n[+] {config.symbol} {config.data_type.value}:")
                print(f"    Already downloaded: {len(existing_files)} files")
                print(f"    Need to download: {len(files_to_download)} files")
            
            if files_to_download:
                # Create filtered DataFrame with only files to download
                filtered_df = pl.DataFrame(files_to_download)
                configs_with_filtered_dfs.append((config, filtered_df))
            else:
                print(f"    [+] All files already exist, skipping download")
        else:
            # Directory doesn't exist, download all files
            configs_with_filtered_dfs.append((config, files_df))
    
    # Download missing files
    if configs_with_filtered_dfs:
        results = await batch_download_multiple(
            configs_with_filtered_dfs,
            str(dataset_dir),
            max_concurrent_per_config=num_concurrent_downloads,
            max_concurrent_configs=2  # Download both types concurrently
        )
    else:
        print("\n[+] All files already downloaded, skipping download step")
        results = []
    
    # Print download summary
    if results:
        print("\n" + "="*70)
        print("DOWNLOAD SUMMARY")
        print("="*70)
        total_files = 0
        total_successful = 0
        for (config, _), result in zip(configs_with_filtered_dfs, results):
            data_type_name = config.data_type.value
            if config.interval:
                data_type_name += f" ({config.interval})"
            
            print(f"\n{config.symbol} - {data_type_name}:")
            print(f"  Files: {result['successful']}/{result['total_files']}")
            print(f"  Speed: {result['avg_speed_mbps']:.2f} MB/s")
            print(f"  Time: {result['elapsed_time']:.2f}s")
            
            total_files += result['total_files']
            total_successful += result['successful']
        
        print(f"\n{'='*70}")
        print(f"TOTAL: {total_successful}/{total_files} files downloaded successfully")
        print(f"{'='*70}\n")
    
    # ========================================================================
    # Processing Files to Parquet
    # ========================================================================
    print("\n" + "="*70)
    print("Processing Files to Parquet")
    print("="*70)
    
    for config, files_df in configs_with_dfs:
        # Determine directory for this config
        market_name = config.market.value.replace('/', '_')
        folder_name = f"{config.symbol}_{config.data_type.value}_{market_name}"
        if config.interval:
            folder_name += f"_{config.interval}"
        
        zip_dir = dataset_dir / folder_name
        parquet_dir = zip_dir  # Save parquet files in same directory as zip files
        
        parquet_dir.mkdir(exist_ok=True)
        
        # Check if zip directory exists and has files
        if not zip_dir.exists():
            print(f"[!] Directory not found: {zip_dir}")
            continue
        
        zip_files = list(zip_dir.glob("*.zip"))
        if not zip_files:
            print(f"[!] No ZIP files found in: {zip_dir}")
            continue
        
        print(f"\nProcessing {config.symbol} {config.data_type.value}...")
        print(f"  Source: {zip_dir}")
        print(f"  Destination: {parquet_dir}")
        print(f"  Files to process: {len(zip_files)}")
        
        try:
            # Process files to parquet
            result = process_downloaded_files(
                local_folder=str(zip_dir),
                files_df=files_df,
                data_type=config.data_type.value,
                save_folder=str(parquet_dir),
                max_workers=num_workers
            )
            
            if result:
                print(f"[+] Processed {result['successful']}/{result['total']} files successfully")
                if result['failed'] > 0:
                    print(f"[!] {result['failed']} files failed")
                print(f"    Output: {parquet_dir}")
            
        except Exception as e:
            print(f"[X] Error processing {config.data_type.value}: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "#"*70)
    print("# PIPELINE COMPLETE!")
    print("#"*70)
    print(f"\nAll data saved in: {dataset_dir}")
    print("\n" + "#"*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())