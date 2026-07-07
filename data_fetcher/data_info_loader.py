"""
Data Info Loader with Smart Monthly/Daily Selection
Fetches file metadata from Binance S3 data repository.
"""

import logging
import re
import urllib.parse
from datetime import datetime, timedelta

import polars as pl
import requests
from bs4 import BeautifulSoup

from .binance_config import BinanceDataRepository, DataConfig, Frequency

logger = logging.getLogger(__name__)


def parse_date_range(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    """Parse date strings into datetime objects."""
    return (
        datetime.strptime(start_date, '%Y-%m-%d'),
        datetime.strptime(end_date, '%Y-%m-%d'),
    )


def determine_optimal_frequency(start_date: str, end_date: str) -> list[tuple[str, str, Frequency]]:
    """
    Intelligently determine whether to use monthly or daily data based on date range.

    Strategy:
    - For complete months: use monthly data
    - For partial months: use daily data

    Returns list of (start_date, end_date, Frequency) tuples for each segment.

    Examples:
        "2024-05-03" to "2024-11-20" ->
            [("2024-05-03", "2024-05-31", DAILY),   # partial May
             ("2024-06-01", "2024-10-31", MONTHLY), # full Jun-Oct
             ("2024-11-01", "2024-11-20", DAILY)]   # partial Nov
    """
    start, end = parse_date_range(start_date, end_date)
    segments: list[tuple[str, str, Frequency]] = []

    current = start
    while current <= end:
        is_month_start = current.day == 1

        # Get last day of current month
        next_month = current.replace(day=28) + timedelta(days=4)
        last_day_of_month = (next_month - timedelta(days=next_month.day)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )

        if is_month_start and last_day_of_month <= end:
            segments.append((
                current.strftime('%Y-%m-%d'),
                last_day_of_month.strftime('%Y-%m-%d'),
                Frequency.MONTHLY,
            ))
            current = last_day_of_month + timedelta(days=1)
        else:
            segment_end = min(last_day_of_month, end)
            segments.append((
                current.strftime('%Y-%m-%d'),
                segment_end.strftime('%Y-%m-%d'),
                Frequency.DAILY,
            ))
            current = segment_end + timedelta(days=1)

    # Consolidate consecutive segments of the same frequency
    consolidated: list[tuple[str, str, Frequency]] = []
    for seg_start, seg_end, freq in segments:
        if consolidated and consolidated[-1][2] == freq:
            consolidated[-1] = (consolidated[-1][0], seg_end, freq)
        else:
            consolidated.append((seg_start, seg_end, freq))

    return consolidated


def fetch_files_for_config(
    config: DataConfig, frequency: Frequency,
) -> tuple[pl.DataFrame, str | None, str | None]:
    """
    Fetch all available files for a given configuration and frequency.

    Returns (DataFrame with file info, earliest_date, latest_date).
    """
    BinanceDataRepository.validate_config(config)

    prefix = BinanceDataRepository.build_prefix(config, frequency)
    base_url = BinanceDataRepository.BASE_URL

    files_data: list[dict] = []
    continuation_token: str | None = None

    logger.info(f"Fetching {frequency.value} {config.data_type.value} for {config.symbol} ({config.market.value})")

    while True:
        url = f"{base_url}/?list-type=2&prefix={prefix}"
        if continuation_token:
            encoded_token = urllib.parse.quote(continuation_token)
            url += f"&continuation-token={encoded_token}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Error fetching file list: {e}")
            break

        soup = BeautifulSoup(response.content, 'xml')

        for content in soup.find_all('Contents'):
            key_elem = content.find('Key')
            size_elem = content.find('Size')
            modified_elem = content.find('LastModified')
            if not key_elem or not size_elem or not modified_elem:
                continue

            key = key_elem.text
            size = int(size_elem.text)
            last_modified = modified_elem.text

            filename = key.split('/')[-1]

            # Parse date from filename
            file_date: datetime | None = None
            if frequency == Frequency.MONTHLY:
                if date_match := re.search(r'\d{4}-\d{2}', filename):
                    year, month = map(int, date_match.group().split('-'))
                    file_date = datetime(year, month, 1)
            else:
                if date_match := re.search(r'\d{4}-\d{2}-\d{2}', filename):
                    file_date = datetime.strptime(date_match.group(), '%Y-%m-%d')

            if file_date is None:
                continue

            files_data.append({
                'filename': filename,
                'date': file_date.date(),
                'url': f"{base_url}/{key}",
                'size_bytes': size,
                'last_modified': last_modified,
                'symbol': config.symbol,
                'data_type': config.data_type.value,
                'market': config.market.value,
                'frequency': frequency.value,
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

    if not files_data:
        logger.warning(f"No files found for {config.data_type.value} {config.symbol}")
        return pl.DataFrame(), None, None

    df = pl.DataFrame(files_data)
    df = df.sort('date', nulls_last=True)

    date_stats = df.select(
        pl.col('date').min().alias('start_date'),
        pl.col('date').max().alias('end_date'),
    )
    start_date = date_stats['start_date'][0]
    end_date = date_stats['end_date'][0]

    logger.info(f"Found {len(df)} files, date range: {start_date} to {end_date}")

    return df, str(start_date), str(end_date)


def fetch_and_combine_smart(config: DataConfig) -> pl.DataFrame:
    """
    Intelligently fetch files using monthly data where possible, daily where necessary.

    Returns combined DataFrame with all available files in the date range.
    """
    logger.info(f"Smart fetch: {config.symbol} - {config.data_type.value} ({config.start_date} to {config.end_date})")

    segments = determine_optimal_frequency(config.start_date, config.end_date)

    logger.info("Optimized fetch strategy:")
    for i, (seg_start, seg_end, freq) in enumerate(segments, 1):
        logger.info(f"  Segment {i}: {seg_start} to {seg_end} -> {freq.value.upper()}")

    all_dfs: list[pl.DataFrame] = []

    for seg_start, seg_end, frequency in segments:
        df, _, _ = fetch_files_for_config(config, frequency)

        if df.is_empty():
            continue

        start_date = datetime.strptime(seg_start, '%Y-%m-%d').date()
        end_date = datetime.strptime(seg_end, '%Y-%m-%d').date()

        df_filtered = df.filter(
            (pl.col('date') >= start_date) & (pl.col('date') <= end_date),
        ).filter(pl.col('filename').str.ends_with('.zip'))

        if not df_filtered.is_empty():
            all_dfs.append(df_filtered)
            logger.info(f"  {len(df_filtered)} files from {frequency.value}")

    if not all_dfs:
        logger.warning("No files found in date range")
        return pl.DataFrame()

    combined_df = pl.concat(all_dfs).unique(subset=['filename']).sort('date')

    total_size_mb = combined_df['size_bytes'].sum() / (1024**2)
    logger.info(f"Total: {len(combined_df)} files, {total_size_mb:.2f} MB")

    return combined_df
