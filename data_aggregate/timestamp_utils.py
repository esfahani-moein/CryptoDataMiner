"""
Timestamp format detection and normalization utilities.

Handles various timestamp formats from different exchanges:
- Unix timestamps in seconds
- Unix timestamps in milliseconds
- Unix timestamps in microseconds
- Datetime objects with various time units
- Mixed formats (datetime[μs] containing millisecond values - Binance quirk)
"""

from enum import Enum
from typing import Tuple, Optional
import polars as pl
from data_aggregate.config import MILLISECOND_THRESHOLD, MICROSECOND_THRESHOLD


class TimestampFormat(Enum):
    """Enumeration of supported timestamp formats."""
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MICROSECONDS = "microseconds"
    DATETIME_MS = "datetime_ms"
    DATETIME_US = "datetime_us"
    DATETIME_NS = "datetime_ns"
    DATETIME_QUIRK = "datetime_quirk"  # Binance quirk: datetime[μs] with ms values


def detect_timestamp_format(df: pl.DataFrame, timestamp_col: str = 'open_time') -> TimestampFormat:
    """
    Automatically detect the timestamp format in a DataFrame.
    
    Args:
        df: Input DataFrame
        timestamp_col: Name of the timestamp column
        
    Returns:
        Detected timestamp format
        
    Raises:
        ValueError: If timestamp format cannot be determined
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")
    
    dtype = df[timestamp_col].dtype
    
    # Handle datetime types
    if isinstance(dtype, pl.Datetime):
        time_unit = dtype.time_unit
        
        # Check for Binance quirk: datetime[μs] with millisecond values
        # Binance stores millisecond epoch values in microsecond datetime fields
        # This causes dates to display as 1970 instead of actual year (e.g., 2024)
        if time_unit == 'us':
            sample_epoch = df[timestamp_col].dt.epoch('us').head(100)
            avg_value = sample_epoch.mean()
            
            # If values are in millisecond range (>1e12) but in μs field, it's the quirk
            # Normal μs timestamps would be >1e15 (year 2001+)
            if avg_value > MILLISECOND_THRESHOLD:
                return TimestampFormat.DATETIME_QUIRK
            
            return TimestampFormat.DATETIME_US
        elif time_unit == 'ms':
            return TimestampFormat.DATETIME_MS
        elif time_unit == 'ns':
            return TimestampFormat.DATETIME_NS
    
    # Handle integer/float timestamp columns
    elif dtype in [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Float64]:
        # Sample values to determine the format
        sample_values = df[timestamp_col].head(100)
        avg_value = sample_values.mean()
        
        if avg_value > MICROSECOND_THRESHOLD:
            return TimestampFormat.MICROSECONDS
        elif avg_value > MILLISECOND_THRESHOLD:
            return TimestampFormat.MILLISECONDS
        else:
            return TimestampFormat.SECONDS
    
    raise ValueError(f"Unsupported timestamp dtype: {dtype}")


def normalize_timestamp_to_datetime(
    df: pl.DataFrame,
    timestamp_col: str,
    format_hint: Optional[TimestampFormat] = None
) -> Tuple[pl.DataFrame, TimestampFormat]:
    """
    Normalize timestamp column to proper datetime for aggregation.
    
    Converts any timestamp format to a standardized datetime[ms] format
    suitable for time-based operations.
    
    Args:
        df: Input DataFrame
        timestamp_col: Name of the timestamp column
        format_hint: Optional hint about the timestamp format (auto-detected if None)
        
    Returns:
        Tuple of (normalized DataFrame, detected format)
    """
    # Detect format if not provided
    if format_hint is None:
        format_hint = detect_timestamp_format(df, timestamp_col)
    
    df_norm = df.clone()
    
    # Determine which columns to normalize
    timestamp_columns = [timestamp_col]
    if 'close_time' in df.columns:
        timestamp_columns.append('close_time')
    
    # Apply normalization based on detected format
    if format_hint == TimestampFormat.DATETIME_QUIRK:
        # Binance quirk: Extract "microsecond" values which are actually milliseconds
        for col in timestamp_columns:
            if col in df_norm.columns:
                df_norm = df_norm.with_columns([
                    pl.from_epoch(pl.col(col).dt.epoch('us'), time_unit='ms').alias(col)
                ])
    
    elif format_hint == TimestampFormat.DATETIME_US:
        # Already datetime[μs], convert to datetime[ms] for consistency
        for col in timestamp_columns:
            if col in df_norm.columns:
                df_norm = df_norm.with_columns([
                    pl.from_epoch(pl.col(col).dt.epoch('us') // 1000, time_unit='ms').alias(col)
                ])
    
    elif format_hint == TimestampFormat.DATETIME_NS:
        # Convert nanoseconds to milliseconds
        for col in timestamp_columns:
            if col in df_norm.columns:
                df_norm = df_norm.with_columns([
                    pl.from_epoch(pl.col(col).dt.epoch('ns') // 1_000_000, time_unit='ms').alias(col)
                ])
    
    elif format_hint == TimestampFormat.DATETIME_MS:
        # Already in correct format, no conversion needed
        pass
    
    elif format_hint == TimestampFormat.SECONDS:
        # Convert seconds to datetime[ms]
        for col in timestamp_columns:
            if col in df_norm.columns:
                df_norm = df_norm.with_columns([
                    pl.from_epoch(pl.col(col), time_unit='s').alias(col)
                ])
    
    elif format_hint == TimestampFormat.MILLISECONDS:
        # Convert milliseconds to datetime[ms]
        for col in timestamp_columns:
            if col in df_norm.columns:
                df_norm = df_norm.with_columns([
                    pl.from_epoch(pl.col(col), time_unit='ms').alias(col)
                ])
    
    elif format_hint == TimestampFormat.MICROSECONDS:
        # Convert microseconds to datetime[ms]
        for col in timestamp_columns:
            if col in df_norm.columns:
                df_norm = df_norm.with_columns([
                    pl.from_epoch((pl.col(col) // 1000).cast(pl.Int64), time_unit='ms').alias(col)
                ])
    
    return df_norm, format_hint


def restore_original_timestamp_format(
    df: pl.DataFrame,
    timestamp_col: str,
    original_format: TimestampFormat
) -> pl.DataFrame:
    """
    Restore timestamps to their original format after aggregation.
    
    Args:
        df: DataFrame with normalized timestamps
        timestamp_col: Name of the timestamp column
        original_format: Original timestamp format to restore
        
    Returns:
        DataFrame with timestamps in original format
    """
    df_restored = df.clone()
    
    timestamp_columns = [timestamp_col]
    if 'close_time' in df.columns:
        timestamp_columns.append('close_time')
    
    if original_format == TimestampFormat.DATETIME_QUIRK:
        # Restore Binance quirk format: datetime[μs] with millisecond values
        for col in timestamp_columns:
            if col in df_restored.columns:
                df_restored = df_restored.with_columns([
                    pl.from_epoch(pl.col(col).dt.epoch('ms'), time_unit='us').alias(col)
                ])
    
    elif original_format == TimestampFormat.DATETIME_US:
        # Restore to datetime[μs]
        for col in timestamp_columns:
            if col in df_restored.columns:
                df_restored = df_restored.with_columns([
                    pl.from_epoch(pl.col(col).dt.epoch('ms') * 1000, time_unit='us').alias(col)
                ])
    
    elif original_format == TimestampFormat.DATETIME_NS:
        # Restore to datetime[ns]
        for col in timestamp_columns:
            if col in df_restored.columns:
                df_restored = df_restored.with_columns([
                    pl.from_epoch(pl.col(col).dt.epoch('ms') * 1_000_000, time_unit='ns').alias(col)
                ])
    
    elif original_format == TimestampFormat.SECONDS:
        # Restore to integer seconds
        for col in timestamp_columns:
            if col in df_restored.columns:
                df_restored = df_restored.with_columns([
                    (pl.col(col).dt.epoch('s').cast(pl.Int64)).alias(col)
                ])
    
    elif original_format == TimestampFormat.MILLISECONDS:
        # Restore to integer milliseconds
        for col in timestamp_columns:
            if col in df_restored.columns:
                df_restored = df_restored.with_columns([
                    (pl.col(col).dt.epoch('ms').cast(pl.Int64)).alias(col)
                ])
    
    elif original_format == TimestampFormat.MICROSECONDS:
        # Restore to integer microseconds
        for col in timestamp_columns:
            if col in df_restored.columns:
                df_restored = df_restored.with_columns([
                    (pl.col(col).dt.epoch('ms') * 1000).alias(col)
                ])
    
    return df_restored


def fix_binance_timestamp_display(
    df: pl.DataFrame,
    timestamp_col: str = 'open_time',
    fix_close_time: bool = True
) -> pl.DataFrame:
    """
    Fix Binance timestamp display issue (datetime[μs] showing 1970 dates).
    
    Binance exports often have datetime[μs] fields containing millisecond epoch values,
    causing dates to display as 1970 instead of the actual year (e.g., 2024).
    
    This function converts those timestamps to proper datetime[ms] format so dates
    display correctly.
    
    Args:
        df: Input DataFrame with Binance quirk timestamps
        timestamp_col: Name of the primary timestamp column
        fix_close_time: If True, also fix close_time column if present
        
    Returns:
        DataFrame with corrected timestamp display (datetime[ms])
        
    Example:
        >>> df = pl.read_parquet("BTCUSDT-1s-2024-11.parquet")
        >>> print(df['open_time'][0])  # Shows: 1970-01-21 00:40:19.200000
        >>> df_fixed = fix_binance_timestamp_display(df)
        >>> print(df_fixed['open_time'][0])  # Shows: 2024-11-01 00:00:00
    """
    df_fixed = df.clone()
    
    # Determine which columns to fix
    cols_to_fix = [timestamp_col]
    if fix_close_time and 'close_time' in df.columns:
        cols_to_fix.append('close_time')
    
    # Check if we need to fix (datetime with microsecond unit)
    dtype = df[timestamp_col].dtype
    if not isinstance(dtype, pl.Datetime):
        raise ValueError(
            f"Column '{timestamp_col}' is not a datetime type (found: {dtype}). "
            "This function only fixes datetime columns with Binance's quirk."
        )
    
    if dtype.time_unit != 'us':
        raise ValueError(
            f"Column '{timestamp_col}' is not datetime[μs] (found: datetime[{dtype.time_unit}]). "
            "This function is specifically for Binance's datetime[μs] quirk."
        )
    
    # Convert: extract the "microsecond" values which are actually milliseconds,
    # then create proper datetime[ms]
    for col in cols_to_fix:
        if col in df_fixed.columns:
            df_fixed = df_fixed.with_columns([
                pl.from_epoch(
                    pl.col(col).dt.epoch('us'),  # Get the raw values (milliseconds)
                    time_unit='ms'                # Interpret as milliseconds
                ).alias(col)
            ])
    
    return df_fixed
