import polars as pl
import requests
from bs4 import BeautifulSoup
import urllib
import re
from datetime import datetime

def fetch_available_files(base_url: str, prefix: str, symbol: str, interval: str) -> tuple[pl.DataFrame, str, str]:
    """
    Fetch all available files from a Binance public data repository
    and create a Polars DataFrame with file information
    
    Args:
        base_url: Base URL for the data repository (e.g., 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision')
        prefix: S3 prefix path (e.g., 'data/spot/daily/klines/BTCUSDT/1d/')
        symbol: Trading pair symbol ('BTCUSDT')
        interval: Kline interval ('1d', '1h', '15m')
    
    Returns:
        Polars DataFrame with columns: filename, date, url, size_bytes, last_modified
    """
    files_data = []
    continuation_token = None
    
    while True:
        url = f"{base_url}/?list-type=2&prefix={prefix}"
        if continuation_token:
            encoded_token = urllib.parse.quote(continuation_token)
            url += f"&continuation-token={encoded_token}"
        
        print(f"Fetching file list from: {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        frequency = prefix.split('/')[2]

        soup = BeautifulSoup(response.content, 'xml')
        
        for content in soup.find_all('Contents'):
            key = content.find('Key').text
            size = int(content.find('Size').text)
            last_modified = content.find('LastModified').text
            
            filename = key.split('/')[-1]
            
            if frequency == 'monthly':
                date_pattern = r'\d{4}-\d{2}'
                if date_match := re.search(date_pattern, filename):
                    year, month = map(int, date_match.group().split('-'))
                    file_date = datetime(year, month, 1).date()  # First day of the month
            else:  
                date_pattern = r'\d{4}-\d{2}-\d{2}'
                if date_match := re.search(date_pattern, filename):
                    file_date = datetime.strptime(date_match.group(), '%Y-%m-%d').date()
        
            files_data.append({
                'filename': filename,
                'date': file_date,
                'url': f"{base_url}/{key}",
                'size_bytes': size,
                'last_modified': last_modified
            })
        
        is_truncated = soup.find('IsTruncated')
        if is_truncated and is_truncated.text == 'true':
            next_token = soup.find('NextContinuationToken')
            if next_token:
                continuation_token = next_token.text
            else:
                break
        else:
            break
    
    df = pl.DataFrame(files_data)
    df = df.sort('date', nulls_last=True)

    # Compute and print start and end dates
    date_stats = df.select(
        pl.col('date').min().alias('start_date'),
        pl.col('date').max().alias('end_date')
    )
    start_date = date_stats['start_date'][0]
    end_date = date_stats['end_date'][0]
    
    print(f"Total files found: {len(df)} with Start date: {start_date} and End date: {end_date}")
    return df, start_date, end_date


def filter_by_date_range(df: pl.DataFrame, start_date: str, end_date: str) -> pl.DataFrame:
    """
    Filter DataFrame by file type and date range
    
    Args:
        df: Polars DataFrame from fetch_available_files
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        Filtered Polars DataFrame (only ZIP files within date range)
    """
    # Efficiently count ZIP and CHECKSUM files using Polars
    zip_count = df.filter(pl.col('filename').str.ends_with('.zip')).select(pl.len()).item()
    checksum_count = df.filter(pl.col('filename').str.ends_with('.CHECKSUM')).select(pl.len()).item()
    
    print(f"Total ZIP files: {zip_count}")
    print(f"Total CHECKSUM files: {checksum_count}")
    
    # Filter to only ZIP files (exclude CHECKSUM)
    filtered_df = df.filter(pl.col('filename').str.ends_with('.zip'))
    
    # Parse dates once for efficiency
    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # Filter by date range
    filtered_df = filtered_df.filter(
        (pl.col('date') >= start) & (pl.col('date') <= end)
    )
    
    print(f"Filtered to {len(filtered_df)} ZIP files between {start_date} and {end_date}")
    
    return filtered_df