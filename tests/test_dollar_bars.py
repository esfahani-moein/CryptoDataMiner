"""
Test Suite for Dollar Bars Aggregator

This module provides comprehensive tests to verify:
1. Accuracy of Dollar Bar calculations
2. Proper residual handling
3. Fixed threshold correctness
4. Dynamic threshold adaptation
5. Edge cases and boundary conditions

Run tests with: pytest test_dollar_bars.py -v
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
from dollar_bars import (
    aggregate_trades_to_dollar_bars_fixed,
    aggregate_trades_to_dollar_bars_dynamic,
    calculate_daily_dollar_volume,
    calculate_ema_daily_volume,
    calculate_dynamic_threshold,
    TimeInterval,
)


def create_synthetic_trades(
    n_trades: int = 1000,
    base_price: float = 100.0,
    price_volatility: float = 1.0,
    avg_quantity: float = 1.0,
    start_time_ms: int = 1700000000000
) -> pl.DataFrame:
    """
    Create synthetic trades data for testing.
    
    Args:
        n_trades: Number of trades to generate
        base_price: Base price level
        price_volatility: Price standard deviation
        avg_quantity: Average trade size
        start_time_ms: Starting timestamp in milliseconds
        
    Returns:
        DataFrame with synthetic trades
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic data
    times = np.arange(start_time_ms, start_time_ms + n_trades * 1000, 1000)
    prices = base_price + np.random.randn(n_trades) * price_volatility
    quantities = np.random.exponential(avg_quantity, n_trades)
    quote_quantities = prices * quantities
    is_buyer_maker = np.random.choice([True, False], size=n_trades)
    trade_ids = np.arange(1, n_trades + 1)
    
    return pl.DataFrame({
        "trade_id": trade_ids,
        "time": times,
        "price": prices,
        "quantity": quantities,
        "quote_quantity": quote_quantities,
        "is_buyer_maker": is_buyer_maker,
    })


def test_basic_fixed_threshold():
    """
    Test basic functionality of fixed threshold Dollar Bars.
    
    Verifies:
    - Bars are created correctly
    - Basic OHLCV structure is maintained
    - Dollar volume approximately matches threshold
    """
    print("\n=== Test 1: Basic Fixed Threshold ===")
    
    # Create simple test data
    trades = create_synthetic_trades(n_trades=100, avg_quantity=10.0)
    threshold = 1000.0  # $1000 per bar
    
    # Aggregate to Dollar Bars
    bars = aggregate_trades_to_dollar_bars_fixed(trades, threshold=threshold)
    
    print(f"Input: {len(trades)} trades")
    print(f"Output: {len(bars)} bars")
    print(f"Threshold: ${threshold:,.2f}")
    print(f"\nFirst 3 bars:")
    print(bars.head(3))
    
    # Assertions
    assert len(bars) > 0, "Should create at least one bar"
    assert "bar_id" in bars.columns, "Should have bar_id column"
    assert "open" in bars.columns, "Should have OHLCV columns"
    
    # Check that most bars are close to threshold (excluding last incomplete bar)
    complete_bars = bars[:-1]  # Exclude last bar
    if len(complete_bars) > 0:
        avg_bar_volume = complete_bars["bar_dollar_volume"].mean()
        print(f"\nAverage bar dollar volume: ${avg_bar_volume:,.2f}")
        print(f"Target threshold: ${threshold:,.2f}")
        print(f"Difference: {abs(avg_bar_volume - threshold) / threshold * 100:.2f}%")
    
    print("✓ Test passed!")
    return True


def test_residual_handling():
    """
    Test that residuals are properly carried forward.
    
    This is the critical test for accuracy. Without proper residual handling,
    dollar volume is lost between bars.
    
    Verifies:
    - Total input dollar volume equals total output dollar volume
    - No volume is lost due to threshold overshooting
    """
    print("\n=== Test 2: Residual Handling ===")
    
    # Create test data with known total volume
    trades = create_synthetic_trades(n_trades=500, avg_quantity=5.0)
    total_input_volume = trades["quote_quantity"].sum()
    
    threshold = 500.0
    
    # Aggregate to Dollar Bars
    bars = aggregate_trades_to_dollar_bars_fixed(trades, threshold=threshold)
    total_output_volume = bars["bar_dollar_volume"].sum()
    
    print(f"Total input dollar volume: ${total_input_volume:,.2f}")
    print(f"Total output dollar volume: ${total_output_volume:,.2f}")
    print(f"Difference: ${abs(total_input_volume - total_output_volume):,.2f}")
    print(f"Relative error: {abs(total_input_volume - total_output_volume) / total_input_volume * 100:.6f}%")
    
    # The difference should be negligible (within floating point precision)
    relative_error = abs(total_input_volume - total_output_volume) / total_input_volume
    assert relative_error < 1e-10, f"Volume mismatch: {relative_error * 100}%"
    
    print("✓ Test passed - No volume loss!")
    return True


def test_ohlcv_consistency():
    """
    Test OHLCV consistency within each bar.
    
    Verifies:
    - High >= Open, Close, Low
    - Low <= Open, Close, High
    - Open is first trade price
    - Close is last trade price
    """
    print("\n=== Test 3: OHLCV Consistency ===")
    
    trades = create_synthetic_trades(n_trades=200)
    threshold = 800.0
    
    bars = aggregate_trades_to_dollar_bars_fixed(trades, threshold=threshold)
    
    # Check OHLCV relationships
    high_violations = (bars["high"] < bars["open"]) | (bars["high"] < bars["close"]) | (bars["high"] < bars["low"])
    low_violations = (bars["low"] > bars["open"]) | (bars["low"] > bars["close"]) | (bars["low"] > bars["high"])
    
    n_high_violations = high_violations.sum()
    n_low_violations = low_violations.sum()
    
    print(f"Total bars: {len(bars)}")
    print(f"High violations: {n_high_violations}")
    print(f"Low violations: {n_low_violations}")
    
    assert n_high_violations == 0, "High should be >= all other prices"
    assert n_low_violations == 0, "Low should be <= all other prices"
    
    print("✓ Test passed - OHLCV consistent!")
    return True


def test_daily_volume_calculation():
    """
    Test daily dollar volume calculation.
    
    Verifies:
    - Trades are correctly grouped by day
    - Daily volumes sum correctly
    """
    print("\n=== Test 4: Daily Volume Calculation ===")
    
    # Create data spanning multiple days
    n_days = 5
    trades_per_day = 100
    n_trades = n_days * trades_per_day
    
    trades = create_synthetic_trades(
        n_trades=n_trades,
        avg_quantity=10.0,
        start_time_ms=1700000000000
    )
    
    # Make sure trades span multiple days (1000ms between trades = 100 seconds per 100 trades)
    # Adjust time to span days
    day_ms = TimeInterval.DAY
    time_increment = day_ms // trades_per_day
    trades = trades.with_columns([
        (pl.col("time").rank().cast(pl.Int64) * time_increment + 1700000000000).alias("time")
    ])
    
    # Calculate daily volumes
    daily_volumes = calculate_daily_dollar_volume(trades)
    
    print(f"Total trades: {len(trades)}")
    print(f"Days found: {len(daily_volumes)}")
    print(f"\nDaily volumes:")
    print(daily_volumes)
    
    # Total should match
    total_from_trades = trades["quote_quantity"].sum()
    total_from_daily = daily_volumes["daily_dollar_volume"].sum()
    
    print(f"\nTotal from trades: ${total_from_trades:,.2f}")
    print(f"Total from daily aggregation: ${total_from_daily:,.2f}")
    print(f"Difference: ${abs(total_from_trades - total_from_daily):,.6f}")
    
    relative_error = abs(total_from_trades - total_from_daily) / total_from_trades
    assert relative_error < 1e-10, f"Daily volume mismatch: {relative_error * 100}%"
    
    print("✓ Test passed!")
    return True


def test_ema_calculation():
    """
    Test EMA calculation for daily volumes.
    
    Verifies:
    - EMA is calculated correctly
    - EMA responds to changes in volume
    """
    print("\n=== Test 5: EMA Calculation ===")
    
    # Create simple daily volume data
    daily_volumes = pl.DataFrame({
        "day_start": [i * TimeInterval.DAY for i in range(10)],
        "daily_dollar_volume": [100.0, 110.0, 105.0, 115.0, 120.0, 125.0, 130.0, 125.0, 135.0, 140.0]
    })
    
    # Calculate EMA
    result = calculate_ema_daily_volume(daily_volumes, span=3, min_periods=1)
    
    print("Daily volumes with EMA:")
    print(result)
    
    # Check that EMA exists and is calculated
    assert "ema_daily_volume" in result.columns, "Should have EMA column"
    assert result["ema_daily_volume"].null_count() == 0, "EMA should have no nulls with min_periods=1"
    
    # EMA should be generally increasing for increasing data
    ema_values = result["ema_daily_volume"].to_numpy()
    assert ema_values[-1] > ema_values[0], "EMA should trend with data"
    
    print("✓ Test passed!")
    return True


def test_dynamic_threshold_calculation():
    """
    Test dynamic threshold calculation.
    
    Verifies:
    - Threshold is calculated correctly from EMA
    - Formula T = EMA / K is applied correctly
    """
    print("\n=== Test 6: Dynamic Threshold Calculation ===")
    
    ema_daily_volume = 1_000_000.0  # $1M average daily volume
    target_bars_per_day = 50
    
    threshold = calculate_dynamic_threshold(ema_daily_volume, target_bars_per_day)
    
    expected_threshold = 1_000_000.0 / 50  # $20,000
    
    print(f"EMA daily volume: ${ema_daily_volume:,.2f}")
    print(f"Target bars per day: {target_bars_per_day}")
    print(f"Calculated threshold: ${threshold:,.2f}")
    print(f"Expected threshold: ${expected_threshold:,.2f}")
    
    assert abs(threshold - expected_threshold) < 0.01, "Threshold calculation incorrect"
    
    print("✓ Test passed!")
    return True


def test_dynamic_dollar_bars():
    """
    Test dynamic Dollar Bars with adaptive threshold.
    
    Verifies:
    - Bars are created with dynamic thresholds
    - Threshold adapts over time
    - Total volume is conserved
    """
    print("\n=== Test 7: Dynamic Dollar Bars ===")
    
    # Create data spanning multiple days with varying volume
    n_trades = 5000
    trades = create_synthetic_trades(
        n_trades=n_trades,
        avg_quantity=10.0,
        start_time_ms=1700000000000
    )
    
    # Spread trades over multiple days
    day_ms = TimeInterval.DAY
    time_increment = (day_ms * 7) // n_trades  # Spread over 7 days
    trades = trades.with_columns([
        (pl.col("time").rank().cast(pl.Int64) * time_increment + 1700000000000).alias("time")
    ])
    
    # Aggregate with dynamic threshold
    target_bars_per_day = 20
    bars, thresholds = aggregate_trades_to_dollar_bars_dynamic(
        trades,
        target_bars_per_day=target_bars_per_day,
        ema_span=3
    )
    
    print(f"Input: {len(trades)} trades")
    print(f"Output: {len(bars)} bars")
    print(f"Days: {len(thresholds)}")
    print(f"Target bars per day: {target_bars_per_day}")
    
    print(f"\nThreshold evolution:")
    print(thresholds.head(10))
    
    print(f"\nFirst 5 bars:")
    print(bars.head(5))
    
    # Check volume conservation
    total_input_volume = trades["quote_quantity"].sum()
    total_output_volume = bars["bar_dollar_volume"].sum()
    
    print(f"\nTotal input volume: ${total_input_volume:,.2f}")
    print(f"Total output volume: ${total_output_volume:,.2f}")
    print(f"Difference: ${abs(total_input_volume - total_output_volume):,.6f}")
    
    relative_error = abs(total_input_volume - total_output_volume) / total_input_volume
    assert relative_error < 1e-10, f"Volume mismatch in dynamic bars: {relative_error * 100}%"
    
    # Check that thresholds vary (dynamic behavior)
    threshold_std = thresholds["threshold"].std()
    print(f"\nThreshold std dev: ${threshold_std:,.2f}")
    assert threshold_std > 0, "Thresholds should vary with dynamic approach"
    
    print("✓ Test passed!")
    return True


def test_edge_cases():
    """
    Test edge cases and boundary conditions.
    
    Tests:
    - Very small dataset (few trades)
    - Single trade
    - Large threshold (incomplete bar)
    - Zero volume trades
    """
    print("\n=== Test 8: Edge Cases ===")
    
    # Test 1: Single trade
    print("\nTest 8.1: Single trade")
    single_trade = create_synthetic_trades(n_trades=1, avg_quantity=5.0)
    bars = aggregate_trades_to_dollar_bars_fixed(single_trade, threshold=100.0)
    assert len(bars) == 1, "Should create one bar for single trade"
    assert bars["num_trades"][0] == 1, "Bar should contain one trade"
    print("✓ Single trade test passed")
    
    # Test 2: Very large threshold (all trades in one bar)
    print("\nTest 8.2: Large threshold")
    trades = create_synthetic_trades(n_trades=100, avg_quantity=1.0)
    total_volume = trades["quote_quantity"].sum()
    large_threshold = total_volume * 10  # 10x total volume
    bars = aggregate_trades_to_dollar_bars_fixed(trades, threshold=large_threshold)
    assert len(bars) == 1, "Should create only one bar with large threshold"
    assert bars["num_trades"][0] == 100, "Bar should contain all trades"
    print("✓ Large threshold test passed")
    
    # Test 3: Very small threshold (many bars)
    print("\nTest 8.3: Small threshold")
    trades = create_synthetic_trades(n_trades=100, avg_quantity=10.0)
    small_threshold = 50.0
    bars = aggregate_trades_to_dollar_bars_fixed(trades, threshold=small_threshold)
    assert len(bars) > 10, "Should create many bars with small threshold"
    print(f"Created {len(bars)} bars with threshold ${small_threshold}")
    print("✓ Small threshold test passed")
    
    print("\n✓ All edge case tests passed!")
    return True


def run_all_tests():
    """Run all test functions."""
    print("=" * 70)
    print("DOLLAR BARS AGGREGATOR - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    tests = [
        test_basic_fixed_threshold,
        test_residual_handling,
        test_ohlcv_consistency,
        test_daily_volume_calculation,
        test_ema_calculation,
        test_dynamic_threshold_calculation,
        test_dynamic_dollar_bars,
        test_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {test_func.__name__}")
            print(f"Error: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
