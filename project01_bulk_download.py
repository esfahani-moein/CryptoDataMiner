"""
Binance Bulk Data Downloader
=============================
Downloads historical data from Binance data vision (spot & futures).
Each file is processed to parquet individually — no combining.

Usage:
    python project01_bulk_download.py -s BTCUSDT --start-date 2025-11-01 --end-date 2025-11-30
    python project01_bulk_download.py -s BTCUSDT --market spot --data-types klines trades
    python project01_bulk_download.py -s BTCUSDT --frequency daily --start-date 2025-11-01 --end-date 2025-11-30
"""

import argparse
import asyncio
import logging
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from data_fetcher.binance_config import DataType, get_data_type_schema
from data_fetcher.data_downloader import download_files_df

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MARKETS = ['spot', 'futures/um', 'futures/cm']

SPOT_DATA_TYPES = ['klines', 'trades', 'aggTrades', 'bookTicker', 'bookDepth']
FUTURES_DATA_TYPES = [
    'klines', 'trades', 'aggTrades', 'bookTicker', 'bookDepth',
    'metrics', 'fundingRate', 'premiumIndexKlines', 'markPriceKlines', 'indexPriceKlines',
]

INTERVAL_TYPES = {'klines', 'premiumIndexKlines', 'markPriceKlines', 'indexPriceKlines'}
DEFAULT_INTERVAL = '1m'


def get_available_data_types(market: str) -> list[str]:
    """Return data types available for a given market."""
    return SPOT_DATA_TYPES if market == 'spot' else FUTURES_DATA_TYPES


def generate_date_list(start_date: str, end_date: str, frequency: str) -> list[str]:
    """Generate date strings: YYYY-MM for monthly, YYYY-MM-DD for daily."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    if start > end:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")

    dates: list[str] = []
    if frequency == 'monthly':
        current = start.replace(day=1)
        while current <= end:
            dates.append(current.strftime('%Y-%m'))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
    else:
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
    return dates


def build_file_url(
    market: str, frequency: str, data_type: str, symbol: str,
    date_str: str, interval: str | None = None,
) -> tuple[str, str]:
    """Build Binance data URL and filename. Returns (url, filename)."""
    base_url = "https://data.binance.vision"
    if interval:
        path = f"data/{market}/{frequency}/{data_type}/{symbol}/{interval}"
        filename = f"{symbol}-{interval}-{date_str}.zip"
    else:
        path = f"data/{market}/{frequency}/{data_type}/{symbol}"
        filename = f"{symbol}-{data_type}-{date_str}.zip"
    return f"{base_url}/{path}/{filename}", filename


def get_output_path(
    symbol: str, data_type: str, date_str: str, base_dir: Path, filename: str,
) -> Path:
    """Output path: {base_dir}/dataset_{SYMBOL}/{YEAR}_{MONTH}/{data_type}/{stem}.parquet"""
    year_month = date_str[:7].replace('-', '_')
    output_dir = base_dir / f"dataset_{symbol}" / year_month / data_type
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / (Path(filename).stem + '.parquet')


def process_zip_to_parquet(zip_path: Path, output_path: Path, data_type: str) -> bool:
    """Extract CSV from ZIP and convert to parquet with proper schema.

    Handles both headered (newer Binance files) and headerless (older files) CSVs
    by peeking at the first line to detect a header.
    """
    try:
        schema = get_data_type_schema(DataType(data_type))

        with zipfile.ZipFile(zip_path, 'r') as zf:
            csv_name = next((n for n in zf.namelist() if n.endswith('.csv')), None)
            if not csv_name:
                logger.error(f"No CSV in {zip_path.name}")
                return False

            with zf.open(csv_name) as csv_file:
                first_line = csv_file.readline()
                csv_file.seek(0)

                has_header = any(c.isalpha() for c in first_line.decode('utf-8', errors='replace').split(',')[0])

                if has_header:
                    df = pl.read_csv(
                        csv_file,
                        has_header=True,
                        schema_overrides=schema['dtypes'],
                    )
                    df = df.rename({old: new for old, new in zip(df.columns, schema['columns'])})
                else:
                    df = pl.read_csv(
                        csv_file,
                        has_header=False,
                        new_columns=schema['columns'],
                        schema_overrides=schema['dtypes'],
                    )

        # Sort by first timestamp column
        ts_cols = schema.get('timestamp_cols', [])
        if ts_cols and ts_cols[0] in df.columns:
            df = df.sort(ts_cols[0])

        df.write_parquet(
            output_path,
            compression='zstd',
            compression_level=18,
            statistics=True,
            row_group_size=1000_000,
        )
        logger.info(f"Saved: {output_path.name} ({len(df):,} rows)")
        return True
    except Exception as e:
        logger.error(f"Error processing {zip_path.name}: {e}")
        return False


async def download_and_process_datatype(
    symbol: str,
    data_type: str,
    market: str,
    frequency: str,
    start_date: str,
    end_date: str,
    base_dir: Path,
    max_concurrent: int = 5,
    max_retries: int = 3,
) -> dict:
    """Download and process all files for a specific data type."""
    logger.info(f"{'='*70}")
    logger.info(f"Processing {data_type} for {symbol} ({market}, {frequency})")
    logger.info(f"{'='*70}")

    interval = DEFAULT_INTERVAL if data_type in INTERVAL_TYPES else None
    dates = generate_date_list(start_date, end_date, frequency)
    logger.info(f"{len(dates)} {frequency} periods")

    temp_dir = base_dir / 'temp_downloads' / data_type
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Build download list (skip if parquet already exists)
    download_info: list[dict] = []
    for date_str in dates:
        url, filename = build_file_url(market, frequency, data_type, symbol, date_str, interval)
        output_path = get_output_path(symbol, data_type, date_str, base_dir, filename)
        if output_path.exists():
            logger.info(f"⊙ Skipping {output_path.name} (already processed)")
            continue
        if (temp_dir / filename).exists():
            continue
        download_info.append({'url': url, 'filename': filename, 'size_bytes': 0})

    # Download
    failed_count = 0
    if download_info:
        df = pl.DataFrame(download_info)
        logger.info(f"Downloading {len(df)} files...")
        try:
            stats = await download_files_df(
                df, str(temp_dir),
                max_concurrent=max_concurrent,
                max_retries=max_retries,
            )
            failed_count = stats.get('failed', 0)
        except Exception as e:
            logger.error(f"Download error for {data_type}: {e}")
            failed_count = len(download_info)

    # Process each downloaded file → parquet
    for date_str in dates:
        _, filename = build_file_url(market, frequency, data_type, symbol, date_str, interval)
        downloaded_path = temp_dir / filename
        if not downloaded_path.exists():
            continue
        output_path = get_output_path(symbol, data_type, date_str, base_dir, filename)
        if not output_path.exists():
            process_zip_to_parquet(downloaded_path, output_path, data_type)
        downloaded_path.unlink(missing_ok=True)

    shutil.rmtree(temp_dir, ignore_errors=True)

    status = 'completed' if failed_count == 0 else 'partial'
    logger.info(f"✓ {data_type}: {status} ({failed_count} failed)")
    return {'data_type': data_type, 'status': status, 'failed': failed_count}


async def bulk_download(
    symbol: str,
    start_date: str,
    end_date: str,
    data_types: list[str],
    market: str = 'futures/um',
    frequency: str = 'monthly',
    output_dir: str = 'dataset',
    max_concurrent: int = 5,
    max_retries: int = 3,
) -> None:
    """Download and process all specified data types concurrently."""
    logger.info(f"\n{'#'*70}")
    logger.info("BINANCE BULK DOWNLOADER")
    logger.info(f"{'#'*70}")
    logger.info(f"Symbol: {symbol} | Market: {market} | Frequency: {frequency}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Data types: {', '.join(data_types)}")
    logger.info(f"Output: {output_dir} | Concurrency: {max_concurrent} | Retries: {max_retries}")
    logger.info(f"{'#'*70}\n")

    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    results = await asyncio.gather(*[
        download_and_process_datatype(
            symbol, dt, market, frequency, start_date, end_date, base_dir,
            max_concurrent, max_retries,
        )
        for dt in data_types
    ], return_exceptions=True)

    logger.info(f"\n{'#'*70}")
    logger.info("BULK DOWNLOAD COMPLETED")
    logger.info(f"{'#'*70}")
    for result in results:
        if isinstance(result, dict):
            logger.info(f"  {result['data_type']}: {result['status']} ({result['failed']} failed)")
        else:
            logger.error(f"  ERROR: {result}")
    logger.info(f"{'#'*70}\n")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Binance Bulk Data Downloader (spot & futures)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  # Futures UM monthly (default)\n'
            '  python project01_bulk_download.py -s BTCUSDT --start-date 2025-11-01 --end-date 2025-11-30\n\n'
            '  # Spot data\n'
            '  python project01_bulk_download.py -s BTCUSDT --market spot --data-types klines trades\n\n'
            '  # Daily frequency\n'
            '  python project01_bulk_download.py -s BTCUSDT --frequency daily --start-date 2025-11-01 --end-date 2025-11-30\n'
        ),
    )
    parser.add_argument('-s', '--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
    parser.add_argument('--start-date', required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--market', default='futures/um', choices=MARKETS,
                        help='Market: spot, futures/um, futures/cm (default: futures/um)')
    parser.add_argument('--frequency', default='monthly', choices=['monthly', 'daily'],
                        help='Data frequency (default: monthly)')
    parser.add_argument('--data-types', nargs='+', default=None,
                        help='Data types to download (default: all available for market)')
    parser.add_argument('--output-dir', default='dataset', help='Output directory (default: dataset)')
    parser.add_argument('--max-concurrent', type=int, default=5,
                        help='Max concurrent downloads per data type (default: 5)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Max retry attempts per file (default: 3)')

    args = parser.parse_args()

    data_types = args.data_types or get_available_data_types(args.market)

    asyncio.run(bulk_download(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        data_types=data_types,
        market=args.market,
        frequency=args.frequency,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
    ))


if __name__ == '__main__':
    main()
