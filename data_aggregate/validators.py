"""
Validation and comparison utilities for aggregated OHLCV data.

Provides tools to validate aggregation accuracy by comparing against ground truth data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import polars as pl

from data_aggregate.config import DEFAULT_TOLERANCE


@dataclass
class ValidationResult:
    """
    Results from comparing two OHLCV DataFrames.
    
    Attributes:
        status: Overall validation status ('PERFECT_MATCH', 'MISMATCHES_FOUND', 'FAILED')
        rows_original: Number of rows in original DataFrame
        rows_aggregated: Number of rows in aggregated DataFrame
        rows_matched: Number of rows that matched on timestamp
        match_rate: Percentage of rows that matched (0-100)
        perfect_match: True if all columns match within tolerance
        column_results: Dictionary of per-column comparison metrics
    """
    status: str
    rows_original: int
    rows_aggregated: int
    rows_matched: int
    match_rate: float
    perfect_match: bool
    column_results: Dict[str, Dict] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Format validation results as human-readable string."""
        lines = [
            "=" * 80,
            "OHLCV AGGREGATION VALIDATION REPORT",
            "=" * 80,
            f"Status: {self.status}",
            "",
            "Row Counts:",
            f"  Original:   {self.rows_original:,}",
            f"  Aggregated: {self.rows_aggregated:,}",
            f"  Matched:    {self.rows_matched:,} ({self.match_rate:.2f}%)",
            "",
        ]
        
        if self.column_results:
            lines.append(f"{'Column':<30} {'Status':<12} {'Max Diff':<15} {'Mean Diff':<15} {'Mismatches'}")
            lines.append("-" * 80)
            
            for col, metrics in self.column_results.items():
                status = "✓ PASS" if metrics['perfect'] else "✗ FAIL"
                max_diff = metrics['max_diff']
                mean_diff = metrics['mean_diff']
                mismatches = metrics['mismatches']
                
                lines.append(
                    f"{col:<30} {status:<12} {max_diff:<15.2e} {mean_diff:<15.2e} {mismatches}"
                )
        
        lines.append("=" * 80)
        lines.append(f"Overall: {'✓ ALL CHECKS PASSED' if self.perfect_match else '✗ VALIDATION FAILED'}")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def compare_dataframes(
    df_original: pl.DataFrame,
    df_aggregated: pl.DataFrame,
    tolerance: float = DEFAULT_TOLERANCE,
    timestamp_col: str = 'open_time'
) -> ValidationResult:
    """
    Compare original data with aggregated data to validate accuracy.
    
    Performs a comprehensive comparison including:
    - Row count validation
    - Timestamp alignment
    - Per-column accuracy metrics (max diff, mean diff, mismatch count)
    - Sample mismatch identification
    
    Args:
        df_original: Original DataFrame (ground truth)
        df_aggregated: Aggregated DataFrame (to validate)
        tolerance: Tolerance for floating-point comparisons (default: 1e-10)
        timestamp_col: Name of timestamp column for joining
        
    Returns:
        ValidationResult object with detailed comparison metrics
    """
    # Normalize timestamps for comparison if needed
    df_orig = df_original.clone()
    df_agg = df_aggregated.clone()
    
    # Ensure compatible timestamp types for joining
    if df_orig[timestamp_col].dtype != df_agg[timestamp_col].dtype:
        # Convert both to millisecond epoch integers for comparison
        if isinstance(df_orig[timestamp_col].dtype, pl.Datetime):
            df_orig = df_orig.with_columns(
                pl.col(timestamp_col).dt.epoch('ms').cast(pl.Int64).alias(timestamp_col)
            )
        if isinstance(df_agg[timestamp_col].dtype, pl.Datetime):
            df_agg = df_agg.with_columns(
                pl.col(timestamp_col).dt.epoch('ms').cast(pl.Int64).alias(timestamp_col)
            )
    
    # Join on timestamp
    joined = df_orig.join(
        df_agg,
        on=timestamp_col,
        how='inner',
        suffix='_agg'
    )
    
    # Calculate match rate
    rows_original = len(df_orig)
    rows_aggregated = len(df_agg)
    rows_matched = len(joined)
    match_rate = (rows_matched / max(rows_original, rows_aggregated)) * 100 if max(rows_original, rows_aggregated) > 0 else 0.0
    
    # Check for join failure
    if rows_matched == 0:
        return ValidationResult(
            status='FAILED: No matching timestamps',
            rows_original=rows_original,
            rows_aggregated=rows_aggregated,
            rows_matched=0,
            match_rate=0.0,
            perfect_match=False,
            column_results={}
        )
    
    # Determine columns to compare
    columns_to_compare = _get_comparable_columns(joined)
    
    # Compare each column
    column_results = {}
    all_perfect = True
    
    for col in columns_to_compare:
        col_agg = f'{col}_agg'
        if col not in joined.columns or col_agg not in joined.columns:
            continue
        
        # Calculate absolute differences
        diff = (joined[col] - joined[col_agg]).abs()
        
        max_diff = diff.max()
        mean_diff = diff.mean()
        mismatches = (diff > tolerance).sum()
        is_perfect = mismatches == 0
        
        column_results[col] = {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'mismatches': mismatches,
            'perfect': is_perfect,
        }
        
        if not is_perfect:
            all_perfect = False
            
            # Capture sample mismatches for debugging
            mismatch_rows = joined.filter(diff > tolerance).select([
                timestamp_col, col, col_agg,
                (pl.col(col) - pl.col(col_agg)).alias('diff')
            ]).head(5)
            
            column_results[col]['sample_mismatches'] = mismatch_rows
    
    # Determine overall status
    status = 'PERFECT_MATCH' if all_perfect else 'MISMATCHES_FOUND'
    
    return ValidationResult(
        status=status,
        rows_original=rows_original,
        rows_aggregated=rows_aggregated,
        rows_matched=rows_matched,
        match_rate=match_rate,
        perfect_match=all_perfect,
        column_results=column_results
    )


def _get_comparable_columns(joined_df: pl.DataFrame) -> List[str]:
    """
    Extract list of comparable columns from joined DataFrame.
    
    Looks for columns that have both original and '_agg' suffixed versions.
    """
    columns = []
    
    # Standard OHLCV columns
    standard_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
    ]
    
    for col in standard_cols:
        if col in joined_df.columns and f'{col}_agg' in joined_df.columns:
            columns.append(col)
    
    return columns


def validate_row_counts(
    df: pl.DataFrame,
    source_interval_seconds: int,
    target_interval_seconds: int,
    expected_duration_seconds: Optional[int] = None
) -> bool:
    """
    Validate that aggregated row count is mathematically correct.
    
    Args:
        df: Aggregated DataFrame
        source_interval_seconds: Source interval
        target_interval_seconds: Target interval
        expected_duration_seconds: Optional expected total duration to validate against
        
    Returns:
        True if row count is valid, False otherwise
    """
    actual_rows = len(df)
    reduction_factor = target_interval_seconds / source_interval_seconds
    
    if expected_duration_seconds:
        expected_rows = expected_duration_seconds / target_interval_seconds
        return abs(actual_rows - expected_rows) < 2  # Allow 1 row tolerance for edge effects
    
    # Without expected duration, just verify reduction factor makes sense
    return reduction_factor >= 1
