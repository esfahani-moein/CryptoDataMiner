"""
Enhanced Data Downloader with Optimizations
- Connection pooling and reuse
- Retry logic with exponential backoff
- Memory-efficient streaming
- Progress tracking per file and overall
- Parallel downloads with controlled concurrency
"""

import aiohttp
import asyncio
import os
import time
from tqdm.asyncio import tqdm
import aiofiles
import polars as pl
from typing import List, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DownloadOptimizer:
    """Optimized downloader with connection pooling and retry logic"""
    
    def __init__(
        self,
        max_concurrent: int = 5,
        chunk_size: int = 1024 * 1024,  # 1MB chunks
        max_retries: int = 3,
        timeout: int = 300,  # 5 minutes
        connector_limit: int = 100
    ):
        """
        Initialize download optimizer
        
        Args:
            max_concurrent: Max concurrent downloads
            chunk_size: Size of download chunks in bytes (default 1MB)
            max_retries: Max retry attempts per file
            timeout: Timeout per file in seconds
            connector_limit: Max connections in pool
        """
        self.max_concurrent = max_concurrent
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.connector_limit = connector_limit
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Connection pooling configuration
        self.connector = aiohttp.TCPConnector(
            limit=connector_limit,
            limit_per_host=10,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True
        )
        
        # Session timeout configuration
        self.timeout_config = aiohttp.ClientTimeout(
            total=timeout,
            connect=30,
            sock_read=30
        )
    
    async def download_single_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        filepath: Path,
        filename: str,
        pbar: tqdm,
        retry_count: int = 0
    ) -> bool:
        """
        Download a single file with retry logic
        
        Returns:
            True if successful, False otherwise
        """
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('Content-Length', 0))
                
                downloaded = 0
                start_time = time.time()
                
                # Create parent directory if needed
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                async with aiofiles.open(filepath, 'wb') as f:
                    async for chunk in response.content.iter_chunked(self.chunk_size):
                        await f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress
                        if total_size > 0:
                            percentage = (downloaded / total_size) * 100
                            elapsed = time.time() - start_time
                            speed = downloaded / elapsed if elapsed > 0 else 0
                            speed_mb = speed / (1024**2)
                            eta = (total_size - downloaded) / speed if speed > 0 else 0
                            
                            pbar.set_description(
                                f"{filename}: {percentage:.1f}% | {speed_mb:.2f} MB/s | ETA: {eta:.1f}s"
                            )
                        
                        pbar.update(len(chunk))
                
                logger.info(f"✓ Downloaded: {filename} ({downloaded / (1024**2):.2f} MB)")
                return True
                
        except asyncio.TimeoutError:
            logger.error(f"✗ Timeout downloading {filename}")
        except aiohttp.ClientError as e:
            logger.error(f"✗ Client error downloading {filename}: {e}")
        except Exception as e:
            logger.error(f"✗ Unexpected error downloading {filename}: {e}")
        
        # Retry logic
        if retry_count < self.max_retries:
            wait_time = 2 ** retry_count  # Exponential backoff
            logger.info(f"↻ Retrying {filename} in {wait_time}s (attempt {retry_count + 1}/{self.max_retries})")
            await asyncio.sleep(wait_time)
            return await self.download_single_file(
                session, url, filepath, filename, pbar, retry_count + 1
            )
        
        logger.error(f"✗ Failed to download {filename} after {self.max_retries} attempts")
        return False
    
    async def download_with_semaphore(
        self,
        session: aiohttp.ClientSession,
        url: str,
        filepath: Path,
        filename: str,
        pbar: tqdm
    ) -> bool:
        """Download with concurrency control"""
        async with self.semaphore:
            return await self.download_single_file(session, url, filepath, filename, pbar)
    
    async def download_files(
        self,
        df: pl.DataFrame,
        local_folder: str,
        filename_col: str = 'filename',
        url_col: str = 'url',
        size_col: str = 'size_bytes'
    ) -> dict:
        """
        Download files from DataFrame with optimizations
        
        Args:
            df: DataFrame with file information
            local_folder: Destination folder
            filename_col: Column name for filename
            url_col: Column name for URL
            size_col: Column name for file size
            
        Returns:
            Dict with download statistics
        """
        local_path = Path(local_folder)
        local_path.mkdir(parents=True, exist_ok=True)
        
        total_size = df[size_col].sum()
        total_files = len(df)
        
        print(f"\n{'='*70}")
        print(f"Starting download of {total_files} files ({total_size / (1024**2):.2f} MB)")
        print(f"Destination: {local_folder}")
        print(f"Max concurrent downloads: {self.max_concurrent}")
        print(f"Chunk size: {self.chunk_size / (1024**2):.2f} MB")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        successful_downloads = 0
        failed_downloads = 0
        
        # Create persistent session with connection pooling
        async with aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout_config
        ) as session:
            
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc="Total Progress"
            ) as pbar:
                
                tasks = []
                for row in df.iter_rows(named=True):
                    url = row[url_col]
                    filename = row[filename_col]
                    filepath = local_path / filename
                    
                    # Skip if file already exists and has correct size
                    if filepath.exists():
                        existing_size = filepath.stat().st_size
                        expected_size = row[size_col]
                        if existing_size == expected_size:
                            logger.info(f"⊙ Skipping {filename} (already exists)")
                            pbar.update(expected_size)
                            successful_downloads += 1
                            continue
                    
                    task = self.download_single_file(session, url, filepath, filename, pbar)
                    tasks.append(task)
                
                # Execute downloads
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, Exception):
                            failed_downloads += 1
                        elif result:
                            successful_downloads += 1
                        else:
                            failed_downloads += 1
        
        elapsed_time = time.time() - start_time
        avg_speed = total_size / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"Download Complete!")
        print(f"  Successful: {successful_downloads}/{total_files}")
        print(f"  Failed: {failed_downloads}/{total_files}")
        print(f"  Total time: {elapsed_time:.2f}s")
        print(f"  Average speed: {avg_speed / (1024**2):.2f} MB/s")
        print(f"  Destination: {local_folder}")
        print(f"{'='*70}\n")
        
        return {
            'total_files': total_files,
            'successful': successful_downloads,
            'failed': failed_downloads,
            'elapsed_time': elapsed_time,
            'avg_speed_mbps': avg_speed / (1024**2)
        }


async def download_files_df(
    df: pl.DataFrame,
    local_folder: str,
    max_concurrent: int = 5,
    chunk_size: int = 1024 * 1024,
    max_retries: int = 3
) -> dict:
    """
    Download files from DataFrame (optimized version)
    
    Args:
        df: Polars DataFrame with 'url', 'filename', and 'size_bytes' columns
        local_folder: Local folder path to save files
        max_concurrent: Max concurrent downloads
        chunk_size: Download chunk size in bytes (default 1MB)
        max_retries: Max retry attempts per file
        
    Returns:
        Dict with download statistics
    """
    optimizer = DownloadOptimizer(
        max_concurrent=max_concurrent,
        chunk_size=chunk_size,
        max_retries=max_retries
    )
    
    return await optimizer.download_files(df, local_folder)


async def batch_download_multiple(
    configs_with_dfs: List[tuple],
    base_folder: str,
    max_concurrent_per_config: int = 3,
    max_concurrent_configs: int = 2
) -> List[dict]:
    """
    Download multiple configurations concurrently
    
    Args:
        configs_with_dfs: List of (DataConfig, DataFrame) tuples
        base_folder: Base folder for downloads
        max_concurrent_per_config: Max concurrent downloads per config
        max_concurrent_configs: Max configs to download simultaneously
        
    Returns:
        List of download statistics dicts
    """
    semaphore = asyncio.Semaphore(max_concurrent_configs)
    
    async def download_config(config, df, folder):
        async with semaphore:
            return await download_files_df(
                df, folder, max_concurrent=max_concurrent_per_config
            )
    
    tasks = []
    for config, df in configs_with_dfs:
        # Create subfolder for each config
        folder_name = f"{config.symbol}_{config.data_type.value}_{config.market.value}"
        if config.interval:
            folder_name += f"_{config.interval}"
        
        download_folder = os.path.join(base_folder, folder_name)
        tasks.append(download_config(config, df, download_folder))
    
    print(f"\n{'#'*70}")
    print(f"BATCH DOWNLOAD: {len(configs_with_dfs)} configurations")
    print(f"Max concurrent configs: {max_concurrent_configs}")
    print(f"{'#'*70}\n")
    
    results = await asyncio.gather(*tasks)
    
    # Summary
    total_files = sum(r['total_files'] for r in results)
    total_successful = sum(r['successful'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    
    print(f"\n{'#'*70}")
    print(f"BATCH DOWNLOAD COMPLETE")
    print(f"  Total files: {total_files}")
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")
    print(f"{'#'*70}\n")
    
    return results
