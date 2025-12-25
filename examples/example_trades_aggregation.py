"""
Example Usage of Trades Aggregation Module

This script demonstrates how to use the trades aggregation module
to convert raw trades data into OHLCV candlestick format.
"""

from pathlib import Path
import polars as pl
from trades_aggregation import (
    aggregate_trades_to_ohlcv,
    aggregate_trades_from_file,
    validate_aggregation_accuracy,
    print_validation_results,
    TimeInterval
)


def example_basic_aggregation():
    """Example: Basic aggregation from loaded DataFrame"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Aggregation")
    print("=" * 70)
    
    # Define paths
    project_root = Path(__file__).parent
    trades_path = project_root / "dataset" / "dataset_BTCUSDT" / \
                  "BTCUSDT_trades_futures_um" / "BTCUSDT-trades-2025-11.parquet"
    
    # Load trades data
    print(f"\nLoading trades from: {trades_path}")
    trades_df = pl.read_parquet(trades_path)
    print(f"Loaded {len(trades_df):,} trades")
    
    # Aggregate to 1-minute candles
    print("\nAggregating to 1-minute OHLCV candles...")
    klines_1m = aggregate_trades_to_ohlcv(trades_df, TimeInterval.MINUTE)
    
    print(f"Generated {len(klines_1m):,} candles")
    print("\nFirst 5 candles:")
    print(klines_1m.head(5))
    
    return klines_1m


def example_multi_timeframe():
    """Example: Aggregate to multiple timeframes"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Multi-Timeframe Aggregation")
    print("=" * 70)
    
    # Load trades
    project_root = Path(__file__).parent
    trades_path = project_root / "dataset" / "dataset_BTCUSDT" / \
                  "BTCUSDT_trades_futures_um" / "BTCUSDT-trades-2025-11.parquet"
    
    trades_df = pl.read_parquet(trades_path)
    
    # Aggregate to different timeframes
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
    
    results = {}
    for name, interval in timeframes.items():
        klines = aggregate_trades_to_ohlcv(trades_df, interval)
        results[name] = klines
        print(f"  {name:15s}: {len(klines):>6,} candles")
    
    return results


def example_file_aggregation_with_save():
    """Example: Aggregate directly from file and save output"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: File-to-File Aggregation")
    print("=" * 70)
    
    # Define paths
    project_root = Path(__file__).parent
    trades_path = project_root / "dataset" / "dataset_BTCUSDT" / \
                  "BTCUSDT_trades_futures_um" / "BTCUSDT-trades-2025-11.parquet"
    
    # Create output directory
    output_dir = project_root / "trades_aggregation" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate and save 1-hour candles
    output_path = output_dir / "BTCUSDT-1h-2025-11-aggregated.parquet"
    
    print(f"\nAggregating trades to 1-hour candles...")
    print(f"Input:  {trades_path}")
    print(f"Output: {output_path}")
    
    klines_1h = aggregate_trades_from_file(
        trades_path,
        TimeInterval.HOUR,
        output_path
    )
    
    print(f"\n✓ Saved {len(klines_1h):,} 1-hour candles to {output_path}")
    
    return klines_1h


def example_validation():
    """Example: Validate aggregation accuracy"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Validation Against Reference Data")
    print("=" * 70)
    
    # Define paths
    project_root = Path(__file__).parent
    trades_path = project_root / "dataset" / "dataset_BTCUSDT" / \
                  "BTCUSDT_trades_futures_um" / "BTCUSDT-trades-2025-11.parquet"
    reference_path = project_root / "dataset" / "dataset_BTCUSDT" / \
                     "BTCUSDT_klines_futures_um_1m" / "BTCUSDT-1m-2025-11.parquet"
    
    # Load and aggregate
    print("\nLoading data...")
    trades_df = pl.read_parquet(trades_path)
    reference_df = pl.read_parquet(reference_path)
    
    print("Aggregating trades to 1-minute candles...")
    aggregated_df = aggregate_trades_to_ohlcv(trades_df, TimeInterval.MINUTE)
    
    # Validate
    print("\nValidating against reference data from Binance...")
    results = validate_aggregation_accuracy(aggregated_df, reference_df)
    
    # Print detailed results
    print_validation_results(results)
    
    return results


def example_custom_analysis():
    """Example: Custom analysis with aggregated data"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Custom Analysis with Aggregated Data")
    print("=" * 70)
    
    # Load and aggregate
    project_root = Path(__file__).parent
    trades_path = project_root / "dataset" / "dataset_BTCUSDT" / \
                  "BTCUSDT_trades_futures_um" / "BTCUSDT-trades-2025-11.parquet"
    
    trades_df = pl.read_parquet(trades_path)
    klines_1h = aggregate_trades_to_ohlcv(trades_df, TimeInterval.HOUR)
    
    # Calculate some statistics
    print("\n1-Hour Candle Statistics:")
    print("-" * 70)
    
    # Price statistics
    print(f"\nPrice Range:")
    print(f"  Highest High:  ${klines_1h['high'].max():,.2f}")
    print(f"  Lowest Low:    ${klines_1h['low'].min():,.2f}")
    print(f"  Average Close: ${klines_1h['close'].mean():,.2f}")
    
    # Volume statistics
    print(f"\nVolume Statistics:")
    print(f"  Total Volume:     {klines_1h['volume'].sum():,.2f} BTC")
    print(f"  Average Volume:   {klines_1h['volume'].mean():,.2f} BTC/hour")
    print(f"  Max Volume Hour:  {klines_1h['volume'].max():,.2f} BTC")
    
    # Trade count statistics
    print(f"\nTrade Count:")
    print(f"  Total Trades:       {klines_1h['count'].sum():,}")
    print(f"  Avg Trades/Hour:    {klines_1h['count'].mean():.0f}")
    print(f"  Max Trades/Hour:    {klines_1h['count'].max():,}")
    
    # Taker buy ratio
    taker_buy_ratio = (klines_1h['taker_buy_volume'].sum() / klines_1h['volume'].sum()) * 100
    print(f"\nTaker Buy Ratio: {taker_buy_ratio:.2f}%")
    
    # Find most volatile hour
    klines_with_volatility = klines_1h.with_columns([
        ((pl.col("high") - pl.col("low")) / pl.col("low") * 100).alias("volatility_pct")
    ])
    most_volatile = klines_with_volatility.sort("volatility_pct", descending=True).head(1)
    
    print(f"\nMost Volatile Hour:")
    print(f"  Open Time:    {most_volatile['open_time'][0]}")
    print(f"  Volatility:   {most_volatile['volatility_pct'][0]:.2f}%")
    print(f"  Price Range:  ${most_volatile['low'][0]:,.2f} - ${most_volatile['high'][0]:,.2f}")
    
    return klines_1h


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TRADES AGGREGATION MODULE - USAGE EXAMPLES")
    print("=" * 70)
    
    # Run all examples
    example_basic_aggregation()
    example_multi_timeframe()
    example_file_aggregation_with_save()
    example_validation()
    example_custom_analysis()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
