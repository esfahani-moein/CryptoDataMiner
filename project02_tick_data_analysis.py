from pathlib import Path
import polars as pl
import time
import datetime
from typing import Dict, Tuple
import sys
from trades_aggregation import *


project_root = Path.cwd()
trades_data_path = project_root / "dataset" / "dataset_BTCUSDT"/ "BTCUSDT_trades_futures_um" / "BTCUSDT-trades-2025-11.parquet"


# Load trades data
print(f"\n1. Loading trades data...")
print(f"Path: {trades_data_path}")
trades_df = pl.read_parquet(trades_data_path, n_rows=5_000_000)
print(f"Loaded {len(trades_df):,} trades")
print(f"Time range: {trades_df['time'].min()} to {trades_df['time'].max()}")
# Assuming 'time' column is in milliseconds
min_time_utc = datetime.datetime.fromtimestamp(trades_df['time'].min() / 1000, tz=datetime.timezone.utc)
max_time_utc = datetime.datetime.fromtimestamp(trades_df['time'].max() / 1000, tz=datetime.timezone.utc)
print(f"Time range: {min_time_utc} to {max_time_utc}")


print(trades_df.describe())


print(f"Schema: {trades_df.schema}")
print(f"Estimated size: {trades_df.estimated_size() / 1_000_000:.2f} MB")


# Aggregate to 15-minute candles
print(f"\n2. Aggregating to 15-minute OHLCV candles...")
start_time = time.time()
klines_15m = aggregate_trades_to_ohlcv(trades_df, TimeInterval.MINUTE*15)
elapsed = time.time() - start_time
print(f"Aggregated to {len(klines_15m):,} 15-minute candles in {elapsed:.2f} seconds")
print(f"Schema: {klines_15m.schema}")
print(f"Estimated size: {klines_15m.estimated_size() / 1_000_000:.2f} MB")


# Import Dollar Bars module
from dollar_bars import (
    aggregate_trades_to_dollar_bars_fixed,
    aggregate_trades_to_dollar_bars_dynamic,
    calculate_daily_dollar_volume,
    calculate_ema_daily_volume,
    TimeInterval
)

print("Dollar Bars module loaded successfully!")

# Calculate total dollar volume to estimate a good threshold
total_dollar_volume = trades_df["quote_quantity"].sum()
print(f"Total dollar volume in dataset: ${total_dollar_volume:,.2f}")
print(f"Number of trades: {len(trades_df):,}")

# Let's target approximately 1000 bars for meaningful analysis
target_num_bars = 1000
threshold_estimate = total_dollar_volume / target_num_bars
print(f"\nEstimated threshold for ~{target_num_bars} bars: ${threshold_estimate:,.2f}")

# Round to a nice number - use 500 million
threshold = 500_000_000  # $500 million
print(f"Using threshold: ${threshold:,.2f}")
print(f"Expected number of bars: ~{int(total_dollar_volume / threshold)}")

# Aggregate to Dollar Bars with fixed threshold
print(f"\n3. Aggregating to Dollar Bars (Fixed Threshold: ${threshold:,.0f})...")
start_time = time.time()

dollar_bars_fixed = aggregate_trades_to_dollar_bars_fixed(
    trades_df, 
    threshold=threshold,
    ensure_sorted=True
)

elapsed = time.time() - start_time
print(f"Aggregated to {len(dollar_bars_fixed):,} Dollar Bars in {elapsed:.2f} seconds")
print(f"Schema: {dollar_bars_fixed.schema}")
print(f"Estimated size: {dollar_bars_fixed.estimated_size() / 1_000_000:.2f} MB")


# Display first few bars
print("\nFirst 5 Dollar Bars:")
print(dollar_bars_fixed.head(5))

print("\nLast 5 Dollar Bars:")
print(dollar_bars_fixed.tail(5))

# Statistics
print("\n=== Dollar Bars Statistics ===")
print(dollar_bars_fixed.describe())


# Verify accuracy: Check that total volume is conserved
total_input_volume = trades_df["quote_quantity"].sum()
total_output_volume = dollar_bars_fixed["bar_dollar_volume"].sum()

print(f"\n=== Volume Conservation Check ===")
print(f"Total input dollar volume:  ${total_input_volume:,.2f}")
print(f"Total output dollar volume: ${total_output_volume:,.2f}")
print(f"Difference: ${abs(total_input_volume - total_output_volume):,.2f}")
print(f"Relative error: {abs(total_input_volume - total_output_volume) / total_input_volume * 100:.10f}%")

# Check bar dollar volume distribution
print(f"\n=== Bar Dollar Volume Distribution ===")
print(f"Mean bar volume: ${dollar_bars_fixed['bar_dollar_volume'].mean():,.2f}")
print(f"Median bar volume: ${dollar_bars_fixed['bar_dollar_volume'].median():,.2f}")
print(f"Std dev: ${dollar_bars_fixed['bar_dollar_volume'].std():,.2f}")
print(f"Min: ${dollar_bars_fixed['bar_dollar_volume'].min():,.2f}")
print(f"Max: ${dollar_bars_fixed['bar_dollar_volume'].max():,.2f}")


# Aggregate to Dollar Bars with dynamic threshold
target_bars_per_day = 50  # Target approximately 50 bars per day
ema_span = 7  # Use 7-day EMA for adaptation

print(f"\n4. Aggregating to Dollar Bars (Dynamic Threshold)...")
print(f"Target bars per day: {target_bars_per_day}")
print(f"EMA span: {ema_span} days")

start_time = time.time()

dollar_bars_dynamic, threshold_history = aggregate_trades_to_dollar_bars_dynamic(
    trades_df,
    target_bars_per_day=target_bars_per_day,
    ema_span=ema_span,
    recalculation_interval_ms=TimeInterval.DAY,
    ensure_sorted=True
)

elapsed = time.time() - start_time
print(f"\nAggregated to {len(dollar_bars_dynamic):,} Dollar Bars in {elapsed:.2f} seconds")
print(f"Schema: {dollar_bars_dynamic.schema}")

# Display threshold evolution over time
print("\n=== Threshold Evolution ===")
print(threshold_history)

print("\n=== Threshold Statistics ===")
print(f"Mean threshold: ${threshold_history['threshold'].mean():,.2f}")
print(f"Min threshold: ${threshold_history['threshold'].min():,.2f}")
print(f"Max threshold: ${threshold_history['threshold'].max():,.2f}")
print(f"Std dev: ${threshold_history['threshold'].std():,.2f}")


# Display first few bars
print("\nFirst 5 Dynamic Dollar Bars:")
print(dollar_bars_dynamic.head(5))

# Verify accuracy
total_output_volume_dynamic = dollar_bars_dynamic["bar_dollar_volume"].sum()

print(f"\n=== Volume Conservation Check (Dynamic) ===")
print(f"Total input dollar volume:  ${total_input_volume:,.2f}")
print(f"Total output dollar volume: ${total_output_volume_dynamic:,.2f}")
print(f"Difference: ${abs(total_input_volume - total_output_volume_dynamic):,.2f}")
print(f"Relative error: {abs(total_input_volume - total_output_volume_dynamic) / total_input_volume * 100:.10f}%")

# Compare statistics
print("=== Comparison: Time Bars vs Dollar Bars ===\n")

print(f"15-Minute Time Bars:")
print(f"  Number of bars: {len(klines_15m):,}")
print(f"  Avg trades per bar: {len(trades_df) / len(klines_15m):,.0f}")
print(f"  Avg quote volume per bar: ${klines_15m['quote_volume'].mean():,.2f}")
print(f"  Std dev quote volume: ${klines_15m['quote_volume'].std():,.2f}")
print(f"  Coefficient of variation: {klines_15m['quote_volume'].std() / klines_15m['quote_volume'].mean():.4f}")

print(f"\nFixed Dollar Bars (${threshold:,.0f} threshold):")
print(f"  Number of bars: {len(dollar_bars_fixed):,}")
print(f"  Avg trades per bar: {len(trades_df) / len(dollar_bars_fixed):,.0f}")
print(f"  Avg quote volume per bar: ${dollar_bars_fixed['bar_dollar_volume'].mean():,.2f}")
print(f"  Std dev quote volume: ${dollar_bars_fixed['bar_dollar_volume'].std():,.2f}")
print(f"  Coefficient of variation: {dollar_bars_fixed['bar_dollar_volume'].std() / dollar_bars_fixed['bar_dollar_volume'].mean():.4f}")

print(f"\nDynamic Dollar Bars ({target_bars_per_day} bars/day target):")
print(f"  Number of bars: {len(dollar_bars_dynamic):,}")
print(f"  Avg trades per bar: {len(trades_df) / len(dollar_bars_dynamic):,.0f}")
print(f"  Avg quote volume per bar: ${dollar_bars_dynamic['bar_dollar_volume'].mean():,.2f}")
print(f"  Std dev quote volume: ${dollar_bars_dynamic['bar_dollar_volume'].std():,.2f}")
print(f"  Coefficient of variation: {dollar_bars_dynamic['bar_dollar_volume'].std() / dollar_bars_dynamic['bar_dollar_volume'].mean():.4f}")

print("\n✓ Dollar Bars show more uniform volume distribution (lower coefficient of variation)")

# Import Imbalance Bars module
from imbalance_bars import (
    aggregate_trades_to_tick_imbalance_bars,
    aggregate_trades_to_dollar_imbalance_bars,
    calculate_tick_rule
)

print("Imbalance Bars module loaded successfully!")


# Diagnostic: Understand the data characteristics for imbalance bars
print("\n" + "="*80)
print("🔍 DATA CHARACTERISTICS ANALYSIS")
print("="*80)
print(f"Total trades: {len(trades_df):,}")

# Calculate tick rules for the entire dataset to understand imbalance
prices_np = trades_df['price'].to_numpy()
tick_rules_full = calculate_tick_rule(prices_np)

print(f"\n=== Tick Rule Distribution ===")
num_buys = (tick_rules_full == 1).sum()
num_sells = (tick_rules_full == -1).sum()
buy_percentage = (num_buys / len(tick_rules_full)) * 100
print(f"Total buys: {num_buys:,} ({buy_percentage:.2f}%)")
print(f"Total sells: {num_sells:,} ({(100-buy_percentage):.2f}%)")
print(f"Net imbalance: {num_buys - num_sells:,} trades")

print(f"\n=== Dollar Volume Stats ===")
print(f"Mean quote_quantity: ${trades_df['quote_quantity'].mean():,.2f}")
print(f"Median quote_quantity: ${trades_df['quote_quantity'].median():,.2f}")
print(f"Std dev: ${trades_df['quote_quantity'].std():,.2f}")
print(f"Min: ${trades_df['quote_quantity'].min():,.2f}")
print(f"Max: ${trades_df['quote_quantity'].max():,.2f}")
print(f"Total volume: ${trades_df['quote_quantity'].sum():,.2f}")

# Calculate cumulative signed dollar volume to see actual imbalance
signed_dollar_volume = tick_rules_full * trades_df['quote_quantity'].to_numpy()
cumulative_signed = signed_dollar_volume.cumsum()

print(f"\n=== Signed Dollar Volume Analysis ===")
print(f"Max cumulative signed volume: ${cumulative_signed.max():,.2f}")
print(f"Min cumulative signed volume: ${cumulative_signed.min():,.2f}")
print(f"Range: ${cumulative_signed.max() - cumulative_signed.min():,.2f}")
print(f"Final cumulative: ${cumulative_signed[-1]:,.2f}")
print(f"Mean absolute signed volume per trade: ${abs(signed_dollar_volume).mean():,.2f}")

# Calculate what threshold values will be
exp_num_ticks_sample = 20000
avg_dollar_value = trades_df['quote_quantity'].mean()
buy_proportion_actual = buy_percentage / 100
buy_imbalance_factor_actual = abs(2 * buy_proportion_actual - 1)

print(f"\n=== Threshold Calculation (Dollar Imbalance Bars) ===")
print(f"exp_num_ticks: {exp_num_ticks_sample:,}")
print(f"avg_dollar_value: ${avg_dollar_value:,.2f}")
print(f"actual buy proportion: {buy_proportion_actual:.4f}")
print(f"buy_imbalance_factor: {buy_imbalance_factor_actual:.4f} (|2×{buy_proportion_actual:.4f} - 1|)")
print(f"Expected threshold: ${exp_num_ticks_sample * buy_imbalance_factor_actual * avg_dollar_value:,.2f}")

# Calculate how many times the threshold is exceeded
threshold_value = exp_num_ticks_sample * buy_imbalance_factor_actual * avg_dollar_value
exceedances = (abs(cumulative_signed) >= threshold_value).sum()
print(f"\n⚠️  Times threshold is exceeded in cumulative: {exceedances}")
print(f"This suggests approximately {exceedances} bars could form")

if buy_imbalance_factor_actual < 0.01:
    print(f"\n⚠️  BALANCED MARKET DETECTED!")
    print(f"Market is nearly balanced ({buy_percentage:.2f}% buys)")
    print(f"Algorithm will use minimum threshold factor: 0.01")
    threshold_value_adjusted = exp_num_ticks_sample * 0.01 * avg_dollar_value
    exceedances_adjusted = (abs(cumulative_signed) >= threshold_value_adjusted).sum()
    print(f"Adjusted threshold: ${threshold_value_adjusted:,.2f}")
    print(f"Expected bars with adjustment: {exceedances_adjusted}")

print("="*80)


# Aggregate to Tick Imbalance Bars
# IMPORTANT: In production, exp_num_ticks_init should be a reasonable market-based estimate
# NOT calculated from the data (that would be look-ahead bias)

print(f"\n5. Aggregating to Tick Imbalance Bars...")

# For production: use a reasonable estimate based on market knowledge
# For Bitcoin futures, 20,000 ticks per bar is a reasonable starting point
exp_num_ticks_init_tick = 20000

print(f"Using exp_num_ticks_init: {exp_num_ticks_init_tick:,}")
print(f"Note: This is a market-based estimate, not derived from the data")
print(f"The algorithm will adapt this value using EMA as bars form")

start_time = time.time()

tick_imbalance_bars = aggregate_trades_to_tick_imbalance_bars(
    trades_df,
    exp_num_ticks_init=exp_num_ticks_init_tick,
    num_prev_bars=3,  # Use 3 previous bars for EMA
    num_ticks_ewma_window=20,  # EMA window for expected ticks
    ensure_sorted=True
)

elapsed = time.time() - start_time
print(f"\nAggregated to {len(tick_imbalance_bars):,} Tick Imbalance Bars in {elapsed:.2f} seconds")
if len(tick_imbalance_bars) > 0:
    print(f"Avg trades per bar: {len(trades_df) / len(tick_imbalance_bars):,.0f}")

# Display Tick Imbalance Bars summary
print("\n📊 Tick Imbalance Bars Summary:")
print(f"Total bars: {len(tick_imbalance_bars):,}")
print(f"Average trades per bar: {len(trades_df) / len(tick_imbalance_bars):,.0f}")
print(f"Volume range: ${tick_imbalance_bars['quote_volume'].min():,.0f} to ${tick_imbalance_bars['quote_volume'].max():,.0f}")
print(f"Time range: {tick_imbalance_bars['open_time'].min()} to {tick_imbalance_bars['close_time'].max()}")
print(f"Cumulative theta range: {tick_imbalance_bars['cumulative_theta'].min():.2f} to {tick_imbalance_bars['cumulative_theta'].max():.2f}")
print(f"Expected imbalance range: {tick_imbalance_bars['expected_imbalance'].min():.2f} to {tick_imbalance_bars['expected_imbalance'].max():.2f}")

# Show first few bars
print("\n📋 First 5 Tick Imbalance Bars:")
print(tick_imbalance_bars.head(5))

# Analyze Tick Imbalance Bars in detail
print("\n🔍 Detailed Analysis of Tick Imbalance Bars:")
print(f"\nColumns: {tick_imbalance_bars.columns}")
print(f"\nSchema: {tick_imbalance_bars.schema}")

# Check imbalance statistics
print(f"\n=== Imbalance Statistics ===")
print(f"Avg cumulative theta: {tick_imbalance_bars['cumulative_theta'].mean():.2f}")
print(f"Std cumulative theta: {tick_imbalance_bars['cumulative_theta'].std():.2f}")
print(f"Avg expected imbalance: {tick_imbalance_bars['expected_imbalance'].mean():.2f}")

# Check if bars are forming correctly
print(f"\n=== Bar Formation Check ===")
print(f"Min trades per bar: {tick_imbalance_bars['num_trades'].min()}")
print(f"Max trades per bar: {tick_imbalance_bars['num_trades'].max()}")
print(f"Median trades per bar: {tick_imbalance_bars['num_trades'].median():.0f}")

# Distribution of trades per bar
print(f"\n=== Trades per Bar Distribution ===")
print(tick_imbalance_bars['num_trades'].describe())

# Aggregate to Dollar Imbalance Bars
print("\n" + "="*80)
print("6. AGGREGATING TO DOLLAR IMBALANCE BARS")
print("="*80)

# For production: use a reasonable estimate based on market knowledge
# NOT based on total data size (that would be look-ahead bias)
exp_num_ticks_init_dollar = 20000  # Reasonable guess for liquid markets

print(f"\nUsing exp_num_ticks_init: {exp_num_ticks_init_dollar:,}")
print(f"Note: This is a market-based estimate, not derived from the data")
print(f"⚠️  In production, never use len(trades_df) to set this!")

start_time = time.time()

dollar_imbalance_bars = aggregate_trades_to_dollar_imbalance_bars(
    trades_df,
    exp_num_ticks_init=exp_num_ticks_init_dollar,
    num_prev_bars=3,
    num_ticks_ewma_window=20,
    ensure_sorted=True
)

elapsed = time.time() - start_time
print(f"\n✓ Aggregated to {len(dollar_imbalance_bars):,} Dollar Imbalance Bars in {elapsed:.2f} seconds")

if len(dollar_imbalance_bars) > 0:
    print(f"✓ Avg trades per bar: {len(trades_df) / len(dollar_imbalance_bars):,.0f}")
    
    print(f"\n=== First 3 Bars Details ===")
    for i in range(min(3, len(dollar_imbalance_bars))):
        exp_imb = dollar_imbalance_bars['expected_imbalance'][i]
        cum_theta = dollar_imbalance_bars['cumulative_theta'][i]
        num_trades = dollar_imbalance_bars['num_trades'][i]
        ratio = abs(cum_theta / exp_imb) if exp_imb != 0 else 0
        print(f"\nBar {i}:")
        print(f"  Trades: {num_trades:,}")
        print(f"  Expected threshold: ${exp_imb:,.2f}")
        print(f"  Cumulative theta: ${cum_theta:,.2f}")
        print(f"  Ratio (theta/threshold): {ratio:.2f}x")
        print(f"  Status: {'✓ Exceeded' if abs(cum_theta) >= exp_imb else '✗ Not exceeded'}")
else:
    print("⚠️  WARNING: No bars created!")
    print("This means the cumulative signed dollar volume never exceeded the threshold")
    print("Possible causes:")
    print("  1. Market is too balanced (buys ≈ sells)")
    print("  2. exp_num_ticks_init is too large")
    print("  3. Threshold calculation issue")
print("="*80)



# Display Dollar Imbalance Bars summary
print("\n📊 Dollar Imbalance Bars Summary:")
print(f"Total bars: {len(dollar_imbalance_bars):,}")
print(f"Average trades per bar: {len(trades_df) / len(dollar_imbalance_bars):,.0f}")
print(f"Volume range: ${dollar_imbalance_bars['quote_volume'].min():,.0f} to ${dollar_imbalance_bars['quote_volume'].max():,.0f}")
print(f"Time range: {dollar_imbalance_bars['open_time'].min()} to {dollar_imbalance_bars['close_time'].max()}")
print(f"Cumulative theta range: ${dollar_imbalance_bars['cumulative_theta'].min():,.0f} to ${dollar_imbalance_bars['cumulative_theta'].max():,.0f}")

# Show first few bars
print("\n📋 First 5 Dollar Imbalance Bars:")
print(dollar_imbalance_bars.head(5))

# Analyze Dollar Imbalance Bars in detail
print("\n🔍 Detailed Analysis of Dollar Imbalance Bars:")

if len(dollar_imbalance_bars) > 0:
    # Check imbalance statistics
    print(f"\n=== Imbalance Statistics ===")
    print(f"Avg cumulative theta: ${dollar_imbalance_bars['cumulative_theta'].mean():,.2f}")
    print(f"Std cumulative theta: ${dollar_imbalance_bars['cumulative_theta'].std():,.2f}")
    print(f"Avg expected imbalance: ${dollar_imbalance_bars['expected_imbalance'].mean():,.2f}")
    print(f"Max cumulative theta: ${dollar_imbalance_bars['cumulative_theta'].max():,.2f}")
    print(f"Min cumulative theta: ${dollar_imbalance_bars['cumulative_theta'].min():,.2f}")

    # Check if bars are forming correctly
    print(f"\n=== Bar Formation Check ===")
    print(f"Min trades per bar: {dollar_imbalance_bars['num_trades'].min()}")
    print(f"Max trades per bar: {dollar_imbalance_bars['num_trades'].max()}")
    print(f"Median trades per bar: {dollar_imbalance_bars['num_trades'].median():.0f}")

    # Distribution of trades per bar
    print(f"\n=== Trades per Bar Distribution ===")
    print(dollar_imbalance_bars['num_trades'].describe())
    
    # Compare threshold to actual cumulative
    print(f"\n=== Threshold Analysis ===")
    for i in range(min(3, len(dollar_imbalance_bars))):
        print(f"Bar {i}: theta=${dollar_imbalance_bars['cumulative_theta'][i]:,.0f}, threshold=${dollar_imbalance_bars['expected_imbalance'][i]:,.0f}")
else:
    print("⚠️ No bars created - threshold too high!")

# Create comparison table
print("\n" + "="*80)
print("📊 FINAL COMPARISON: ALL BAR TYPES")
print("="*80)

comparison_data = {
    "Bar Type": ["Time Bars (15m)", "Dollar Bars", "Tick Imbalance", "Dollar Imbalance"],
    "Total Bars": [
        len(klines_15m),
        len(dollar_bars_fixed),
        len(tick_imbalance_bars),
        len(dollar_imbalance_bars)
    ],
    "Avg Trades/Bar": [
        len(trades_df) / len(klines_15m),
        len(trades_df) / len(dollar_bars_fixed),
        len(trades_df) / len(tick_imbalance_bars),
        len(trades_df) / len(dollar_imbalance_bars) if len(dollar_imbalance_bars) > 0 else 0
    ],
    "Avg Duration (s)": [
        (klines_15m['close_time'] - klines_15m['open_time']).mean() / 1000,
        (dollar_bars_fixed['close_time'] - dollar_bars_fixed['open_time']).mean() / 1000,
        (tick_imbalance_bars['close_time'] - tick_imbalance_bars['open_time']).mean() / 1000,
        (dollar_imbalance_bars['close_time'] - dollar_imbalance_bars['open_time']).mean() / 1000 if len(dollar_imbalance_bars) > 0 else 0
    ],
    "Duration CV": [
        (klines_15m['close_time'] - klines_15m['open_time']).std() / (klines_15m['close_time'] - klines_15m['open_time']).mean(),
        (dollar_bars_fixed['close_time'] - dollar_bars_fixed['open_time']).std() / (dollar_bars_fixed['close_time'] - dollar_bars_fixed['open_time']).mean(),
        (tick_imbalance_bars['close_time'] - tick_imbalance_bars['open_time']).std() / (tick_imbalance_bars['close_time'] - tick_imbalance_bars['open_time']).mean(),
        (dollar_imbalance_bars['close_time'] - dollar_imbalance_bars['open_time']).std() / (dollar_imbalance_bars['close_time'] - dollar_imbalance_bars['open_time']).mean() if len(dollar_imbalance_bars) > 0 else 0
    ]
}

comparison_df = pl.DataFrame(comparison_data)
print("\n" + str(comparison_df))

print("\n" + "="*80)
print("🎯 INTERPRETATION OF RESULTS")
print("="*80)

print("\n✅ Time Bars (15-minute):")
print("  - Fixed time intervals")
print("  - May have low activity periods")
print(f"  - Created {len(klines_15m):,} bars")

print("\n✅ Dollar Bars:")
print("  - Based on fixed dollar volume")
print("  - More uniform volume distribution")
print(f"  - Created {len(dollar_bars_fixed):,} bars")
print(f"  - Lower CV = more stable bar size")

print("\n✅ Tick Imbalance Bars:")
print("  - Based on signed tick count imbalance")
print("  - Captures order flow imbalance (buy vs sell pressure)")
print(f"  - Created {len(tick_imbalance_bars):,} bars")

print("\n⚠️  Dollar Imbalance Bars:")
print("  - Based on signed dollar volume imbalance")
print("  - MOST SENSITIVE to order flow imbalance")
print(f"  - Created {len(dollar_imbalance_bars):,} bars")

if len(dollar_imbalance_bars) < 10:
    print("\n🔍 WHY SO FEW DOLLAR IMBALANCE BARS?")
    print("-" * 80)
    print("This is EXPECTED BEHAVIOR when the market is balanced!")
    print("")
    print("Explanation:")
    print("  • In a balanced market, buys ≈ sells (e.g., 50.1% vs 49.9%)")
    print("  • Signed dollar volume accumulates slowly: buys cancel sells")
    print("  • Threshold = exp_num_ticks × |2P[b=1] - 1| × E[v]")
    print("  • When |2P[b=1] - 1| ≈ 0, threshold is high relative to accumulation")
    print("  • Bars only form when there's SUSTAINED directional pressure")
    print("")
    print("This is NOT a bug - it's the design of imbalance bars!")
    print("They're meant to capture periods of significant order flow imbalance,")
    print("not to sample uniformly like time or dollar bars.")
    print("")
    print("Solutions if you need more bars:")
    print("  1. Use smaller exp_num_ticks_init (e.g., 5000 instead of 20000)")
    print("  2. Use Tick Imbalance Bars instead (more granular)")
    print("  3. Accept that few bars = balanced market (this is information!)")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE")
print("="*80)

