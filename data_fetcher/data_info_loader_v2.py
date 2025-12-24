"""
Enhanced Data Info Loader with Smart Monthly/Daily Selection
Fetches file metadata from Binance data repository
"""

import polars as pl
import requests
from bs4 import BeautifulSoup
import urllib
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Tuple, List
from .binance_config import DataConfig, BinanceDataRepository, Frequency


def parse_date_range(start_date: str, end_date: str) -> Tuple[datetime, datetime]:
    """Parse date strings into datetime objects"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    return start, end


def determine_optimal_frequency(start_date: str, end_date: str) -> List[Tuple[str, str, Frequency]]:
    """
    Intelligently determine whether to use monthly or daily data based on date range.
    
    Strategy:
    - For complete months: use monthly data
    - For partial months: use daily data
    - Returns list of (start_date, end_date, frequency) tuples for each segment
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        List of tuples: [(start_date, end_date, Frequency), ...]
        
    Examples:
        "2024-05-03" to "2024-11-20" ->
            [("2024-05-03", "2024-05-31", DAILY),   # partial May
             ("2024-06-01", "2024-10-31", MONTHLY), # full Jun-Oct
             ("2024-11-01", "2024-11-20", DAILY)]   # partial Nov
    """
    start, end = parse_date_range(start_date, end_date)
    segments = []
    
    current = start
    
    while current <= end:
        # Check if current is the first day of the month
        is_month_start = current.day == 1
        
        # Get last day of current month
        next_month = current.replace(day=28) + timedelta(days=4)
        last_day_of_month = (next_month - timedelta(days=next_month.day)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        # Check if we can cover a full month
        if is_month_start and last_day_of_month <= end:
            # Use monthly data for this full month
            month_end = last_day_of_month
            segments.append((
                current.strftime('%Y-%m-%d'),
                month_end.strftime('%Y-%m-%d'),
                Frequency.MONTHLY
            ))
            current = month_end + timedelta(days=1)
        else:
            # Use daily data for partial month
            segment_end = min(last_day_of_month, end)
            segments.append((
                current.strftime('%Y-%m-%d'),
                segment_end.strftime('%Y-%m-%d'),
                Frequency.DAILY
            ))
            current = segment_end + timedelta(days=1)
    
    # Consolidate consecutive segments of the same frequency
    consolidated = []
    for seg_start, seg_end, freq in segments:
        if consolidated and consolidated[-1][2] == freq:
            # Extend previous segment
            consolidated[-1] = (consolidated[-1][0], seg_end, freq)
        else:
            consolidated.append((seg_start, seg_end, freq))
    
    return consolidated


def fetch_files_for_config(config: DataConfig, frequency: Frequency) -> Tuple[pl.DataFrame, str, str]:
    """
    Fetch all available files for a given configuration and frequency
    
    Args:
        config: DataConfig instance
        frequency: Daily or monthly frequency
        
    Returns:
        Tuple of (DataFrame with file info, earliest_date, latest_date)
    """
    # Validate configuration
    BinanceDataRepository.validate_config(config)
    
    # Build prefix
    prefix = BinanceDataRepository.build_prefix(config, frequency)
    base_url = BinanceDataRepository.BASE_URL
    
    files_data = []
    continuation_token = None
    
    print(f"\n{'='*70}")
    print(f"Fetching {frequency.value} {config.data_type.value} data for {config.symbol}")
    print(f"Market: {config.market.value}")
    if config.interval:
        print(f"Interval: {config.interval}")
    print(f"{'='*70}")
    
    while True:
        url = f"{base_url}/?list-type=2&prefix={prefix}"
        if continuation_token:
            encoded_token = urllib.parse.quote(continuation_token)
            url += f"&continuation-token={encoded_token}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching file list: {e}")
            break
        
        soup = BeautifulSoup(response.content, 'xml')
        
        for content in soup.find_all('Contents'):
            key = content.find('Key').text
            size = int(content.find('Size').text)
            last_modified = content.find('LastModified').text
            
            filename = key.split('/')[-1]
            
            # Parse date from filename
            if frequency == Frequency.MONTHLY:
                date_pattern = r'\d{4}-\d{2}'
                if date_match := re.search(date_pattern, filename):
                    year, month = map(int, date_match.group().split('-'))
                    file_date = datetime(year, month, 1).date()
            else:
                date_pattern = r'\d{4}-\d{2}-\d{2}'
                if date_match := re.search(date_pattern, filename):
                    file_date = datetime.strptime(date_match.group(), '%Y-%m-%d').date()
            
            files_data.append({
                'filename': filename,
                'date': file_date,
                'url': f"{base_url}/{key}",
                'size_bytes': size,
                'last_modified': last_modified,
                'symbol': config.symbol,
                'data_type': config.data_type.value,
                'market': config.market.value,
                'frequency': frequency.value
            })
        
        # Check for more pages
        is_truncated = soup.find('IsTruncated')
        if is_truncated and is_truncated.text == 'true':
            next_token = soup.find('NextContinuationToken')
            if next_token:
                continuation_token = next_token.text
            else:
                break
        else:
            break
    
    if not files_data:
        print(f"⚠️  No files found for this configuration")
        return pl.DataFrame(), None, None
    
    df = pl.DataFrame(files_data)
    df = df.sort('date', nulls_last=True)
    
    # Compute date range
    date_stats = df.select(
        pl.col('date').min().alias('start_date'),
        pl.col('date').max().alias('end_date')
    )
    start_date = date_stats['start_date'][0]
    end_date = date_stats['end_date'][0]
    
    print(f"✓ Found {len(df)} files")
    print(f"  Date range: {start_date} to {end_date}")
    
    return df, str(start_date), str(end_date)


def fetch_and_combine_smart(config: DataConfig) -> pl.DataFrame:
    """
    Intelligently fetch files using monthly data where possible, daily where necessary
    
    Args:
        config: DataConfig with start_date and end_date
        
    Returns:
        Combined DataFrame with all available files in the date range
    """
    print(f"\n{'#'*70}")
    print(f"SMART FETCH: {config.symbol} - {config.data_type.value}")
    print(f"Date Range: {config.start_date} to {config.end_date}")
    print(f"{'#'*70}")
    
    # Determine optimal frequency segments
    segments = determine_optimal_frequency(config.start_date, config.end_date)
    
    print(f"\nOptimized fetch strategy:")
    for i, (seg_start, seg_end, freq) in enumerate(segments, 1):
        print(f"  Segment {i}: {seg_start} to {seg_end} -> {freq.value.upper()}")
    
    all_dfs = []
    
    for seg_start, seg_end, frequency in segments:
        # Fetch files for this segment
        df, _, _ = fetch_files_for_config(config, frequency)
        
        if df.is_empty():
            continue
        
        # Filter by segment date range
        start_date = datetime.strptime(seg_start, '%Y-%m-%d').date()
        end_date = datetime.strptime(seg_end, '%Y-%m-%d').date()
        
        df_filtered = df.filter(
            (pl.col('date') >= start_date) & (pl.col('date') <= end_date)
        )
        
        # Filter to only ZIP files
        df_filtered = df_filtered.filter(pl.col('filename').str.ends_with('.zip'))
        
        if not df_filtered.is_empty():
            all_dfs.append(df_filtered)
            print(f"  ✓ {len(df_filtered)} files selected from {frequency.value}")
    
    if not all_dfs:
        print("\n No files found in date range")
        return pl.DataFrame()
    
    # Combine all segments
    combined_df = pl.concat(all_dfs)
    combined_df = combined_df.unique(subset=['filename']).sort('date')
    
    total_size_mb = combined_df['size_bytes'].sum() / (1024**2)
    print(f"\n{'='*70}")
    print(f"TOTAL: {len(combined_df)} files selected")
    print(f"Total size: {total_size_mb:.2f} MB")
    print(f"{'='*70}\n")
    
    return combined_df


# Backward compatibility function
def fetch_available_files(base_url: str, prefix: str, symbol: str, interval: str) -> Tuple[pl.DataFrame, str, str]:
    """
    DEPRECATED: Backward compatibility wrapper
    Use fetch_files_for_config() instead
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
                    file_date = datetime(year, month, 1).date()
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
    DEPRECATED: Backward compatibility wrapper
    Filter DataFrame by date range
    """
    zip_count = df.filter(pl.col('filename').str.ends_with('.zip')).select(pl.len()).item()
    checksum_count = df.filter(pl.col('filename').str.ends_with('.CHECKSUM')).select(pl.len()).item()
    
    print(f"Total ZIP files: {zip_count}")
    print(f"Total CHECKSUM files: {checksum_count}")
    
    filtered_df = df.filter(pl.col('filename').str.ends_with('.zip'))
    
    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    filtered_df = filtered_df.filter(
        (pl.col('date') >= start) & (pl.col('date') <= end)
    )
    
    print(f"Filtered to {len(filtered_df)} ZIP files between {start_date} and {end_date}")
    
    return filtered_df
