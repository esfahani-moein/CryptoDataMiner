#!/usr/bin/env python3
"""
Simple Demo Script - Aggregate BTCUSDT Trades to OHLCV

This script demonstrates aggregating the November 2025 BTCUSDT trades data
to 1-minute OHLCV candles and validating against Binance reference data.
"""

from pathlib import Path
import polars as pl
from trades_aggregation import (
    aggregate_trades_to_ohlcv,
    validate_aggregation_accuracy,
    print_validation_results,
    TimeInterval
)


def main():
    print("=" * 70)
    print("BTCUSDT Trades to OHLCV Aggregation Demo")
    print("=" * 70)
    
    # Define paths using pathlib
    project_root = Path(__file__).parent
    trades_path = project_root / "dataset" / "dataset_BTCUSDT" / \
                  "BTCUSDT_trades_futures_um" / "BTCUSDT-trades-2025-11.parquet"
    reference_path = project_root / "dataset" / "dataset_BTCUSDT" / \
                     "BTCUSDT_klines_futures_um_1m" / "BTCUSDT-1m-2025-11.parquet"
    
    # Load trades data
    print(f"\n1. Loading trades data...")
    print(f"   Path: {trades_path}")
    trades_df = pl.read_parquet(trades_path)
    print(f"   ✓ Loaded {len(trades_df):,} trades")
    print(f"   Time range: {trades_df['time'].min()} to {trades_df['time'].max()}")
    
    # Aggregate to 1-minute candles
    print(f"\n2. Aggregating to 1-minute OHLCV candles...")
    import time
    start_time = time.time()
    klines_1m = aggregate_trades_to_ohlcv(trades_df, TimeInterval.MINUTE)
    elapsed = time.time() - start_time
    
    print(f"   ✓ Generated {len(klines_1m):,} candles in {elapsed:.2f} seconds")
    print(f"   Performance: {len(trades_df) / elapsed:,.0f} trades/second")
    
    # Show sample candle
    print(f"\n3. Sample candle (first 1-minute candle):")
    print(klines_1m.head(1))
    
    # Validate against reference
    print(f"\n4. Validating against Binance reference data...")
    print(f"   Path: {reference_path}")
    reference_df = pl.read_parquet(reference_path)
    print(f"   ✓ Loaded {len(reference_df):,} reference candles")
    
    results = validate_aggregation_accuracy(klines_1m, reference_df, tolerance=1e-4)
    print_validation_results(results)
    
    # Show aggregate statistics
    print("\n" + "=" * 70)
    print("5. Aggregate Statistics")
    print("=" * 70)
    print(f"Price Range:     ${klines_1m['low'].min():,.2f} - ${klines_1m['high'].max():,.2f}")
    print(f"Total Volume:    {klines_1m['volume'].sum():,.2f} BTC")
    print(f"Total Trades:    {klines_1m['count'].sum():,}")
    print(f"Avg Trades/Min:  {klines_1m['count'].mean():.0f}")
    
    taker_buy_ratio = (klines_1m['taker_buy_volume'].sum() / klines_1m['volume'].sum()) * 100
    print(f"Taker Buy Ratio: {taker_buy_ratio:.2f}%")
    
    # Demonstrate multi-timeframe aggregation
    print("\n" + "=" * 70)
    print("6. Multi-Timeframe Aggregation Demo")
    print("=" * 70)
    
    timeframes = {
        "1-second": TimeInterval.SECOND,
        "1-minute": TimeInterval.MINUTE,
        "5-minute": 5 * TimeInterval.MINUTE,
        "15-minute": 15 * TimeInterval.MINUTE,
        "1-hour": TimeInterval.HOUR,
        "4-hour": 4 * TimeInterval.HOUR,
        "1-day": TimeInterval.DAY,
    }
    
    print(f"\nAggregating {len(trades_df):,} trades to various timeframes:\n")
    for name, interval in timeframes.items():
        klines = aggregate_trades_to_ohlcv(trades_df, interval)
        print(f"  {name:12s}: {len(klines):>8,} candles")
    
    print("\n" + "=" * 70)
    print("✅ Demo completed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Run full test suite: python tests/test_trades_aggregation.py")
    print("  - See more examples: python examples/example_trades_aggregation.py")
    print("  - Read documentation: trades_aggregation/README.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
