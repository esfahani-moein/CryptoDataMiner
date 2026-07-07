"""
Async Data Downloader with connection pooling, retry logic, and streaming.
"""

import asyncio
import logging
import time
from pathlib import Path

import aiofiles
import aiohttp
import polars as pl
from tqdm.asyncio import tqdm

from .binance_config import DataConfig

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1 MB


class DownloadOptimizer:
    """Downloader with connection pooling and exponential-backoff retry."""

    def __init__(
        self,
        max_concurrent: int = 5,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_retries: int = 3,
        timeout: int = 300,
        connector_limit: int = 100,
    ):
        self.max_concurrent = max_concurrent
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)

        self.connector = aiohttp.TCPConnector(
            limit=connector_limit,
            limit_per_host=10,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True,
        )
        self.timeout_config = aiohttp.ClientTimeout(
            total=timeout, connect=30, sock_read=30,
        )

    async def download_single_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        filepath: Path,
        filename: str,
        pbar: tqdm,
        retry_count: int = 0,
    ) -> bool:
        """Download a single file with retry. Returns True on success."""
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0

                filepath.parent.mkdir(parents=True, exist_ok=True)

                async with aiofiles.open(filepath, 'wb') as f:
                    async for chunk in response.content.iter_chunked(self.chunk_size):
                        await f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            elapsed = time.time()
                            speed = downloaded / elapsed if elapsed > 0 else 0
                            eta = (total_size - downloaded) / speed if speed > 0 else 0
                            pbar.set_description(
                                f"{filename}: {downloaded / total_size * 100:.1f}%"
                                f" | {speed / 1048576:.2f} MB/s | ETA: {eta:.1f}s"
                            )
                        pbar.update(len(chunk))

                logger.info(f"✓ {filename} ({downloaded / 1048576:.2f} MB)")
                return True

        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logger.error(f"✗ {filename}: {e}")
        except Exception as e:
            logger.error(f"✗ {filename}: {e}")

        if retry_count < self.max_retries:
            wait = 2 ** retry_count
            logger.info(f"↻ Retrying {filename} in {wait}s ({retry_count + 1}/{self.max_retries})")
            await asyncio.sleep(wait)
            return await self.download_single_file(session, url, filepath, filename, pbar, retry_count + 1)

        logger.error(f"✗ {filename} failed after {self.max_retries} attempts")
        return False

    async def download_files(
        self,
        df: pl.DataFrame,
        local_folder: str,
        filename_col: str = 'filename',
        url_col: str = 'url',
        size_col: str = 'size_bytes',
    ) -> dict:
        """Download all files from a Polars DataFrame. Returns stats dict."""
        local_path = Path(local_folder)
        local_path.mkdir(parents=True, exist_ok=True)

        total_size = df[size_col].sum()
        total_files = len(df)

        logger.info(f"Downloading {total_files} files ({total_size / 1048576:.2f} MB) → {local_folder}")

        start_time = time.time()
        successful = 0
        failed = 0

        async with aiohttp.ClientSession(
            connector=self.connector, timeout=self.timeout_config,
        ) as session:
            with tqdm(total=float(total_size), unit='B', unit_scale=True, desc="Total") as pbar:
                tasks = []
                for row in df.iter_rows(named=True):
                    filepath = local_path / row[filename_col]

                    if filepath.exists() and filepath.stat().st_size == row[size_col]:
                        logger.info(f"⊙ Skipping {row[filename_col]} (exists)")
                        pbar.update(row[size_col])
                        successful += 1
                        continue

                    tasks.append(
                        self.download_single_file(
                            session, row[url_col], filepath, row[filename_col], pbar,
                        )
                    )

                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception) or not result:
                            failed += 1
                        else:
                            successful += 1

        elapsed = time.time() - start_time
        logger.info(f"Done: {successful}/{total_files} ok, {failed} failed, {elapsed:.1f}s")

        return {
            'total_files': total_files,
            'successful': successful,
            'failed': failed,
            'elapsed_time': elapsed,
        }


async def download_files_df(
    df: pl.DataFrame,
    local_folder: str,
    max_concurrent: int = 5,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_retries: int = 3,
) -> dict:
    """Download files from a Polars DataFrame with url/filename/size_bytes columns."""
    optimizer = DownloadOptimizer(
        max_concurrent=max_concurrent,
        chunk_size=chunk_size,
        max_retries=max_retries,
    )
    return await optimizer.download_files(df, local_folder)


async def batch_download_multiple(
    configs_with_dfs: list[tuple[DataConfig, pl.DataFrame]],
    base_folder: str,
    max_concurrent_per_config: int = 3,
    max_concurrent_configs: int = 2,
) -> list[dict]:
    """Download multiple (DataConfig, DataFrame) pairs concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent_configs)

    async def download_config(config: DataConfig, df: pl.DataFrame, folder: str) -> dict:
        async with semaphore:
            return await download_files_df(df, folder, max_concurrent=max_concurrent_per_config)

    tasks = []
    for config, df in configs_with_dfs:
        market_name = config.market.value.replace('/', '_')
        folder_name = f"{config.symbol}_{config.data_type.value}_{market_name}"
        if config.interval:
            folder_name += f"_{config.interval}"
        tasks.append(download_config(config, df, str(Path(base_folder) / folder_name)))

    logger.info(f"Batch download: {len(configs_with_dfs)} configs, max {max_concurrent_configs} concurrent")

    results = await asyncio.gather(*tasks)

    total_files = sum(r['total_files'] for r in results)
    total_ok = sum(r['successful'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    logger.info(f"Batch done: {total_ok}/{total_files} ok, {total_failed} failed")

    return results
