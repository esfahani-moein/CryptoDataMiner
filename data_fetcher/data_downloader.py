import aiohttp
import asyncio
import os
import time
from tqdm.asyncio import tqdm
import aiofiles
import polars as pl

async def download_files_df(df: pl.DataFrame, local_folder: str, max_concurrent: int = 5) -> None:
    """
    Download files from URLs in the DataFrame concurrently with progress tracking.
    
    Args:
        df: Polars DataFrame with 'url' and 'filename' columns
        local_folder: Local folder path to save files
        max_concurrent: Max concurrent downloads (default 5 to avoid overload)
    """
    os.makedirs(local_folder, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def download_single(url: str, filename: str, pbar: tqdm):
        async with semaphore:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        total_size = int(response.headers.get('Content-Length', 0))
                        file_path = os.path.join(local_folder, filename)
                        
                        downloaded = 0
                        start_time = time.time()
                        
                        async with aiofiles.open(file_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    percentage = (downloaded / total_size) * 100
                                    elapsed = time.time() - start_time
                                    speed = downloaded / elapsed if elapsed > 0 else 0
                                    eta = (total_size - downloaded) / speed if speed > 0 else 0
                                    pbar.set_description(f"{filename}: {percentage:.1f}% | ETA: {eta:.1f}s")
                                    pbar.update(len(chunk))
                                else:
                                    pbar.update(len(chunk))
                        
                        print(f"Downloaded: {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
    
    with tqdm(total=sum(row['size_bytes'] for row in df.iter_rows(named=True)), unit='B', unit_scale=True, desc="Total Download") as pbar:
        tasks = [download_single(row['url'], row['filename'], pbar) for row in df.iter_rows(named=True)]
        await asyncio.gather(*tasks)
    
    print(f"All downloads completed to {local_folder}")