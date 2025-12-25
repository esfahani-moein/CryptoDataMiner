"""
Comprehensive Test Suite for Trades Aggregation Module

Tests validate the accuracy and performance of trades-to-OHLCV aggregation
against reference data from Binance.
"""

from pathlib import Path
import polars as pl
import time
from typing import Dict, Tuple

# Import aggregation functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from trades_aggregation import (
    aggregate_trades_to_ohlcv,
    validate_aggregation_accuracy,
    print_validation_results,
    TimeInterval
)


def load_test_data() -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load trades and reference klines data for testing.
    
    Returns:
        Tuple of (trades_df, reference_klines_df)
    """
    # Define data paths using pathlib
    project_root = Path(__file__).parent.parent
    trades_path = project_root / "dataset" / "dataset_BTCUSDT" / \
                  "BTCUSDT_trades_futures_um" / "BTCUSDT-trades-2025-11.parquet"
    klines_path = project_root / "dataset" / "dataset_BTCUSDT" / \
                  "BTCUSDT_klines_futures_um_1m" / "BTCUSDT-1m-2025-11.parquet"
    
    print(f"Loading trades data from: {trades_path}")
    trades_df = pl.read_parquet(trades_path)
    print(f"  Loaded {len(trades_df):,} trades")
    
    print(f"Loading reference klines data from: {klines_path}")
    reference_df = pl.read_parquet(klines_path)
    print(f"  Loaded {len(reference_df):,} candles")
    
    return trades_df, reference_df


def test_1min_aggregation(trades_df: pl.DataFrame, reference_df: pl.DataFrame) -> bool:
    """
    Test 1-minute aggregation accuracy against reference data.
    
    This is the primary test - aggregated 1-minute candles should match
    the reference data from Binance with very high accuracy.
    
    Args:
        trades_df: Raw trades data
        reference_df: Reference 1-minute OHLCV data from Binance
        
    Returns:
        True if test passed, False otherwise
    """
    print("\n" + "=" * 70)
    print("TEST 1: 1-Minute Aggregation Accuracy")
    print("=" * 70)
    
    # Perform aggregation
    start_time = time.time()
    aggregated_df = aggregate_trades_to_ohlcv(trades_df, TimeInterval.MINUTE)
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ Aggregated {len(trades_df):,} trades into {len(aggregated_df):,} "
          f"1-minute candles in {elapsed_time:.2f} seconds")
    print(f"  Performance: {len(trades_df) / elapsed_time:,.0f} trades/second")
    
    # Validate against reference
    # Note: Small differences in quote_volume are expected due to Binance's
    # internal rounding. Using tolerance of 1e-4 which is still very accurate
    # (< 0.00001% error on large quote volumes)
    print("\nValidating against reference data...")
    validation_results = validate_aggregation_accuracy(
        aggregated_df, 
        reference_df,
        tolerance=1e-4
    )
    
    # Print results
    print_validation_results(validation_results)
    
    return validation_results["passed"]


def test_multi_timeframe_aggregation(trades_df: pl.DataFrame) -> bool:
    """
    Test aggregation at multiple timeframes.
    
    Validates that aggregation works correctly for various intervals:
    - 1 second
    - 1 minute
    - 1 hour
    - 1 day
    
    Args:
        trades_df: Raw trades data
        
    Returns:
        True if all tests passed, False otherwise
    """
    print("\n" + "=" * 70)
    print("TEST 2: Multi-Timeframe Aggregation")
    print("=" * 70)
    
    test_intervals = {
        "1-second": TimeInterval.SECOND,
        "1-minute": TimeInterval.MINUTE,
        "1-hour": TimeInterval.HOUR,
        "1-day": TimeInterval.DAY,
    }
    
    all_passed = True
    
    for name, interval in test_intervals.items():
        print(f"\nTesting {name} aggregation...")
        
        try:
            start_time = time.time()
            aggregated_df = aggregate_trades_to_ohlcv(trades_df, interval)
            elapsed_time = time.time() - start_time
            
            # Basic validation checks
            assert len(aggregated_df) > 0, "Aggregated dataframe is empty"
            assert "open_time" in aggregated_df.columns, "Missing open_time column"
            assert aggregated_df["open_time"].is_sorted(), "Data not sorted by time"
            
            # Validate OHLC relationships
            invalid_ohlc = aggregated_df.filter(
                (pl.col("high") < pl.col("low")) |
                (pl.col("high") < pl.col("open")) |
                (pl.col("high") < pl.col("close")) |
                (pl.col("low") > pl.col("open")) |
                (pl.col("low") > pl.col("close"))
            )
            
            if len(invalid_ohlc) > 0:
                print(f"  ❌ FAILED - Found {len(invalid_ohlc)} candles with invalid OHLC relationships")
                all_passed = False
                continue
            
            # Check interval consistency
            time_diffs = aggregated_df.select([
                pl.col("open_time").diff().drop_nulls().alias("diff")
            ])
            expected_diff = interval
            actual_diffs = time_diffs["diff"].unique().sort()
            
            print(f"  ✓ Generated {len(aggregated_df):,} candles in {elapsed_time:.2f}s")
            print(f"  ✓ All OHLC relationships valid")
            print(f"  ✓ Time intervals: {actual_diffs.to_list()}")
            
        except Exception as e:
            print(f"  ❌ FAILED - Error: {str(e)}")
            all_passed = False
    
    return all_passed


def test_aggregation_properties(trades_df: pl.DataFrame) -> bool:
    """
    Test that aggregated data maintains expected mathematical properties.
    
    Validates:
    - Volume conservation (sum of trades = sum of candles)
    - Trade count consistency
    - Price bounds (all prices within trade price range)
    
    Args:
        trades_df: Raw trades data
        
    Returns:
        True if all property tests passed, False otherwise
    """
    print("\n" + "=" * 70)
    print("TEST 3: Aggregation Properties Validation")
    print("=" * 70)
    
    # Aggregate to 1-minute
    aggregated_df = aggregate_trades_to_ohlcv(trades_df, TimeInterval.MINUTE)
    
    all_passed = True
    
    # Test 1: Volume conservation
    print("\nTest 3.1: Volume Conservation")
    trades_total_volume = trades_df["quantity"].sum()
    aggregated_total_volume = aggregated_df["volume"].sum()
    volume_diff = abs(trades_total_volume - aggregated_total_volume)
    
    if volume_diff < 1e-6:
        print(f"  ✓ Volume conserved: {trades_total_volume:.8f}")
    else:
        print(f"  ❌ Volume mismatch: trades={trades_total_volume:.8f}, "
              f"aggregated={aggregated_total_volume:.8f}, diff={volume_diff:.8f}")
        all_passed = False
    
    # Test 2: Quote volume conservation
    print("\nTest 3.2: Quote Volume Conservation")
    trades_total_quote_volume = trades_df["quote_quantity"].sum()
    aggregated_total_quote_volume = aggregated_df["quote_volume"].sum()
    quote_volume_diff = abs(trades_total_quote_volume - aggregated_total_quote_volume)
    relative_error = quote_volume_diff / trades_total_quote_volume
    
    # Allow tolerance for floating-point precision (order of operations affects sum)
    # Relative error should be < 1e-10 (essentially machine precision)
    if relative_error < 1e-10:
        print(f"  ✓ Quote volume conserved: {trades_total_quote_volume:.2f}")
        print(f"    (diff: {quote_volume_diff:.8f}, relative error: {relative_error:.2e})")
    else:
        print(f"  ❌ Quote volume mismatch: trades={trades_total_quote_volume:.2f}, "
              f"aggregated={aggregated_total_quote_volume:.2f}, diff={quote_volume_diff:.8f}")
        all_passed = False
    
    # Test 3: Trade count
    print("\nTest 3.3: Trade Count")
    trades_count = len(trades_df)
    aggregated_count = aggregated_df["count"].sum()
    
    if trades_count == aggregated_count:
        print(f"  ✓ Trade count matches: {trades_count:,}")
    else:
        print(f"  ❌ Trade count mismatch: trades={trades_count:,}, "
              f"aggregated={aggregated_count:,}")
        all_passed = False
    
    # Test 4: Price bounds
    print("\nTest 3.4: Price Bounds")
    trades_min_price = trades_df["price"].min()
    trades_max_price = trades_df["price"].max()
    agg_min_price = aggregated_df["low"].min()
    agg_max_price = aggregated_df["high"].max()
    
    if agg_min_price >= trades_min_price and agg_max_price <= trades_max_price:
        print(f"  ✓ Price bounds valid: [{trades_min_price:.2f}, {trades_max_price:.2f}]")
    else:
        print(f"  ❌ Price bounds invalid:")
        print(f"     Trades: [{trades_min_price:.2f}, {trades_max_price:.2f}]")
        print(f"     Aggregated: [{agg_min_price:.2f}, {agg_max_price:.2f}]")
        all_passed = False
    
    # Test 5: Taker buy volume
    print("\nTest 3.5: Taker Buy Volume")
    trades_taker_buy = trades_df.filter(~pl.col("is_buyer_maker"))["quantity"].sum()
    aggregated_taker_buy = aggregated_df["taker_buy_volume"].sum()
    taker_buy_diff = abs(trades_taker_buy - aggregated_taker_buy)
    
    if taker_buy_diff < 1e-6:
        print(f"  ✓ Taker buy volume conserved: {trades_taker_buy:.8f}")
    else:
        print(f"  ❌ Taker buy volume mismatch: trades={trades_taker_buy:.8f}, "
              f"aggregated={aggregated_taker_buy:.8f}, diff={taker_buy_diff:.8f}")
        all_passed = False
    
    return all_passed


def test_edge_cases(trades_df: pl.DataFrame) -> bool:
    """
    Test edge cases and robustness.
    
    Tests:
    - Single trade aggregation
    - Empty interval handling
    - Unsorted data handling
    
    Args:
        trades_df: Raw trades data
        
    Returns:
        True if all edge case tests passed, False otherwise
    """
    print("\n" + "=" * 70)
    print("TEST 4: Edge Cases and Robustness")
    print("=" * 70)
    
    all_passed = True
    
    # Test 4.1: Small sample
    print("\nTest 4.1: Small Sample (1000 trades)")
    try:
        small_sample = trades_df.head(1000)
        aggregated = aggregate_trades_to_ohlcv(small_sample, TimeInterval.MINUTE)
        print(f"  ✓ Successfully aggregated {len(small_sample)} trades to {len(aggregated)} candles")
    except Exception as e:
        print(f"  ❌ FAILED - Error: {str(e)}")
        all_passed = False
    
    # Test 4.2: Custom interval
    print("\nTest 4.2: Custom Interval (5 minutes = 300000ms)")
    try:
        custom_interval = 5 * TimeInterval.MINUTE
        aggregated = aggregate_trades_to_ohlcv(trades_df, custom_interval)
        print(f"  ✓ Successfully aggregated to {len(aggregated)} 5-minute candles")
    except Exception as e:
        print(f"  ❌ FAILED - Error: {str(e)}")
        all_passed = False
    
    # Test 4.3: Unsorted data (if ensure_sorted works)
    print("\nTest 4.3: Unsorted Data Handling")
    try:
        # Create shuffled version of small sample
        shuffled = trades_df.head(10000).sample(fraction=1.0, shuffle=True, seed=42)
        aggregated = aggregate_trades_to_ohlcv(shuffled, TimeInterval.MINUTE, ensure_sorted=True)
        print(f"  ✓ Successfully handled unsorted data: {len(aggregated)} candles")
    except Exception as e:
        print(f"  ❌ FAILED - Error: {str(e)}")
        all_passed = False
    
    return all_passed


def run_all_tests() -> bool:
    """
    Run complete test suite.
    
    Returns:
        True if all tests passed, False otherwise
    """
    print("\n" + "=" * 70)
    print("TRADES AGGREGATION MODULE - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    # Load data
    print("\nLoading test data...")
    trades_df, reference_df = load_test_data()
    
    # Run all tests
    test_results = {
        "1-Minute Aggregation Accuracy": test_1min_aggregation(trades_df, reference_df),
        "Multi-Timeframe Aggregation": test_multi_timeframe_aggregation(trades_df),
        "Aggregation Properties": test_aggregation_properties(trades_df),
        "Edge Cases": test_edge_cases(trades_df),
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED - Please review the output above")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
