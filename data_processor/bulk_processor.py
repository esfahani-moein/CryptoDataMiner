"""
Bulk Data Processor for Binance Futures Data
Handles processing of all 7 data types with new directory structure
"""

import zipfile
import os
import polars as pl
import time
from pathlib import Path
from typing import Optional, List, Tuple
import logging
from datetime import datetime
import sys
from pathlib import Path
from .data_save import save_df_to_parquet
# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetcher.binance_config import DataType, get_data_type_schema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_output_path(
    symbol: str,
    data_type: str,
    date_str: str,
    base_dir: str,
    filename: str
) -> Path:
    """
    Generate output path with new structure:
    dataset/dataset_{SYMBOL}/{YEAR}_{MONTH}/{data_type}/{filename}.parquet
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        data_type: Data type (e.g., 'trades', 'metrics')
        date_str: Date string (YYYY-MM or YYYY-MM-DD)
        base_dir: Base directory
        filename: Original filename
        
    Returns:
        Path object for output file
    """
    # Extract year and month from date string
    if len(date_str) == 7:  # YYYY-MM
        year_month = date_str.replace('-', '_')
    else:  # YYYY-MM-DD
        year_month = date_str[:7].replace('-', '_')
    
    # Build path
    output_dir = Path(base_dir) / f"dataset_{symbol}" / year_month / data_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    output_filename = Path(filename).stem + '.parquet'
    
    return output_dir / output_filename


def process_csv_to_parquet(
    csv_path: Path,
    output_path: Path,
    data_type: str,
    symbol: str
) -> bool:
    """
    Process CSV file and save as Parquet with proper types
    - Unix timestamps as Int64
    - Decimal for monetary values with exact precision
    """
    try:
        # Get schema for this data type
        try:
            schema = get_data_type_schema(DataType(data_type))
        except (ValueError, KeyError):
            logger.error(f"Unknown data type: {data_type}")
            return False
        
        # Build dtypes dict for CSV reading - read Decimal columns as strings initially
        csv_dtypes = {}
        decimal_cols = []
        for col, dtype in schema.get('dtypes', {}).items():
            if str(dtype).startswith('Decimal'):
                csv_dtypes[col] = pl.Utf8  # Read as string to preserve precision
                decimal_cols.append((col, dtype))
            else:
                csv_dtypes[col] = dtype
        
        # Read CSV with appropriate dtypes
        df = pl.read_csv(csv_path, has_header=True, dtypes=csv_dtypes)
        
        # Convert string columns to Decimal with exact precision
        for col, dtype in decimal_cols:
            if col in df.columns:
                try:
                    df = df.with_columns([
                        pl.col(col).cast(dtype, strict=False)
                    ])
                except Exception as e:
                    logger.warning(f"Could not cast {col} to {dtype}: {e}")
        
        # Save to Parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_df_to_parquet(df, output_path)
        logger.info(f"Saved: {output_path.name} ({len(df):,} rows)")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {csv_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_zip_to_parquet(
    zip_path: Path,
    output_path: Path,
    data_type: str,
    symbol: str,
    temp_dir: Path
) -> bool:
    """
    Extract ZIP, process CSV, and save as Parquet
    """
    try:
        # Extract ZIP
        extract_dir = temp_dir / f"extract_{zip_path.stem}"
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find CSV file
        csv_files = list(extract_dir.glob('*.csv'))
        if not csv_files:
            logger.error(f"No CSV found in {zip_path.name}")
            return False
        
        csv_path = csv_files[0]
        
        # Process CSV to Parquet
        result = process_csv_to_parquet(csv_path, output_path, data_type, symbol)
        
        # Cleanup
        import shutil
        shutil.rmtree(extract_dir, ignore_errors=True)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {zip_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def combine_daily_to_monthly(
    daily_files: List[Path],
    output_path: Path,
    data_type: str
) -> bool:
    """
    Combine daily parquet files into a single monthly file
    """
    try:
        if not daily_files:
            logger.warning("No daily files to combine")
            return False
        
        # Read and concatenate all daily files
        dfs = []
        for file_path in daily_files:
            try:
                df = pl.read_parquet(file_path)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Could not read {file_path.name}: {e}")
        
        if not dfs:
            logger.error("No valid daily files to combine")
            return False
        
        # Concatenate
        combined_df = pl.concat(dfs)
        
        # Sort by timestamp (first timestamp column in schema)
        try:
            schema = get_data_type_schema(DataType(data_type))
            if 'timestamp_cols' in schema and schema['timestamp_cols']:
                first_ts_col = schema['timestamp_cols'][0]
                if first_ts_col in combined_df.columns:
                    combined_df = combined_df.sort(first_ts_col)
        except:
            pass  # If sorting fails, continue without sorting
        
        # Save combined file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_df_to_parquet(combined_df, output_path)
        logger.info(f"Combined {len(daily_files)} files into {output_path.name} ({len(combined_df):,} rows)")
        
        # Cleanup daily files
        for file_path in daily_files:
            file_path.unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        logger.error(f"Error combining daily files: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_data_frequency(data_type: str) -> str:
    """
    Determine if data type is monthly or daily
    
    Returns:
        'monthly' or 'daily'
    """
    # Data types that are only available daily
    daily_types = {'metrics', 'bookDepth'}
    
    if data_type in daily_types:
        return 'daily'
    else:
        return 'monthly'


def get_file_extension(data_type: str) -> str:
    """
    Determine file extension for data type
    
    Returns:
        'zip' or 'csv'
    """
    # Most types are ZIP files
    # Only metrics and bookDepth might be CSV
    daily_types = {'metrics', 'bookDepth'}
    
    if data_type in daily_types:
        return 'zip'  # Daily files are typically ZIP
    else:
        return 'zip'  # Monthly files are typically ZIP
