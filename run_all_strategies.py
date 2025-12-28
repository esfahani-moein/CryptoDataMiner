#!/usr/bin/env python3
"""
Comprehensive Strategy Runner and Evaluator
============================================
Runs all strategies across specified date ranges and logs results.

Usage:
    python run_all_strategies.py --start-year 2024 --start-month 1 --end-year 2025 --end-month 11
    python run_all_strategies.py --period 2025_11  # Single period
    python run_all_strategies.py --quick  # Quick test with one period

Output:
    - Console: Summary table of all results
    - File: strategy_results_{timestamp}.json - Detailed results
    - File: strategy_results_{timestamp}.csv - Summary table
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import importlib.util
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_periods(start_year: int, start_month: int, end_year: int, end_month: int) -> List[str]:
    """Generate list of periods from start to end."""
    periods = []
    year, month = start_year, start_month
    
    while (year, month) <= (end_year, end_month):
        periods.append(f"{year}_{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    
    return periods


def discover_strategies() -> List[Tuple[str, str]]:
    """Discover all strategy files in the strategies folder."""
    strategies_dir = PROJECT_ROOT / "strategies"
    strategy_files = []
    
    for f in sorted(strategies_dir.glob("strategy*.py")):
        if f.name.startswith("strategy_base"):
            continue
        if f.name.startswith("__"):
            continue
        
        strategy_files.append((f.stem, str(f)))
    
    return strategy_files


def load_strategy_class(file_path: str, module_name: str):
    """Dynamically load a strategy class from file."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Find the strategy class (inherits from StrategyBase)
    from strategies.strategy_base import StrategyBase
    
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, StrategyBase) and obj is not StrategyBase:
            return obj
    
    return None


def check_data_availability(period: str, symbol: str = "BTCUSDT") -> bool:
    """Check if data exists for a given period."""
    data_path = PROJECT_ROOT / "dataset" / f"dataset_{symbol}" / period
    return data_path.exists() and (data_path / "trades").exists()


def run_single_strategy(
    strategy_class,
    symbol: str,
    period: str,
    timeframe: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run a single strategy and return results."""
    try:
        strategy = strategy_class(
            symbol=symbol,
            period=period,
            timeframe=timeframe
        )
        
        result = strategy.run(verbose=verbose)
        
        return {
            "status": "success",
            "strategy_name": result.strategy_name,
            "period": period,
            "timeframe": timeframe,
            "n_samples": result.n_samples,
            "n_features": result.n_features,
            "n_days": result.n_days,
            "n_trades": result.n_trades,
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1_score": result.f1_score,
            "auc": result.auc,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "calmar_ratio": result.calmar_ratio,
            "sortino_ratio": result.sortino_ratio,
            "top_features": dict(list(result.feature_importance.items())[:10]),
        }
    except Exception as e:
        return {
            "status": "error",
            "strategy_name": strategy_class.__name__ if strategy_class else "Unknown",
            "period": period,
            "timeframe": timeframe,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def format_results_table(results: List[Dict]) -> str:
    """Format results as a table string."""
    successful = [r for r in results if r.get("status") == "success"]
    
    if not successful:
        return "No successful results to display."
    
    # Header
    header = (
        f"{'Strategy':<35} {'Period':<8} {'TF':<6} "
        f"{'Acc':<6} {'AUC':<6} {'WinRate':<8} {'Return':<8} {'PF':<5} {'Days':<4}"
    )
    separator = "-" * len(header)
    
    lines = [separator, header, separator]
    
    for r in sorted(successful, key=lambda x: x['total_return'], reverse=True):
        n_days = r.get('n_days', '?')
        line = (
            f"{r['strategy_name'][:35]:<35} "
            f"{r['period']:<8} "
            f"{r['timeframe']:<6} "
            f"{r['accuracy']:.3f} "
            f"{r['auc']:.3f} "
            f"{r['win_rate']:>7.2%} "
            f"{r['total_return']:>7.2%} "
            f"{r['profit_factor']:>4.2f} "
            f"{n_days:>3}"
        )
        lines.append(line)
    
    lines.append(separator)
    
    # Add summary stats
    if len(successful) > 0:
        avg_sharpe = sum(r['sharpe_ratio'] for r in successful) / len(successful)
        avg_accuracy = sum(r['accuracy'] for r in successful) / len(successful)
        avg_winrate = sum(r['win_rate'] for r in successful) / len(successful)
        best_sharpe = max(successful, key=lambda x: x['sharpe_ratio'])
        
        lines.append(f"\nSUMMARY ({len(successful)} runs):")
        lines.append(f"  Average Accuracy: {avg_accuracy:.3f}")
        lines.append(f"  Average Sharpe:   {avg_sharpe:.2f}")
        lines.append(f"  Average Win Rate: {avg_winrate:.2%}")
        lines.append(f"  Best Strategy:    {best_sharpe['strategy_name']} (Sharpe: {best_sharpe['sharpe_ratio']:.2f})")
    
    errors = [r for r in results if r.get("status") == "error"]
    if errors:
        lines.append(f"\nERRORS ({len(errors)} failures):")
        for e in errors[:5]:
            lines.append(f"  - {e.get('strategy_name', 'Unknown')}: {e.get('error', 'Unknown error')[:60]}")
    
    return "\n".join(lines)


def save_results(results: List[Dict], output_dir: Path) -> Tuple[Path, Path]:
    """Save results to JSON and CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON with full details
    json_path = output_dir / f"strategy_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # CSV summary
    csv_path = output_dir / f"strategy_results_{timestamp}.csv"
    successful = [r for r in results if r.get("status") == "success"]
    
    if successful:
        import csv
        fieldnames = [
            'strategy_name', 'period', 'timeframe', 'n_samples',
            'accuracy', 'precision', 'recall', 'f1_score', 'auc',
            'total_return', 'sharpe_ratio', 'max_drawdown', 
            'win_rate', 'profit_factor', 'calmar_ratio', 'sortino_ratio'
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(successful)
    
    return json_path, csv_path


def validate_metrics(results: List[Dict]) -> List[Dict]:
    """
    Validate and flag suspicious metrics.
    
    Flags strategies where classification metrics don't align with trading metrics.
    """
    for r in results:
        if r.get("status") != "success":
            continue
        
        warnings = []
        
        # Flag 1: AUC near 0.5 with high Sharpe
        if r.get("auc", 0.5) < 0.52 and r.get("sharpe_ratio", 0) > 1.0:
            warnings.append("SUSPICIOUS: Random-like AUC with positive Sharpe")
        
        # Flag 2: Low accuracy with high Sharpe
        if r.get("accuracy", 0.5) < 0.4 and r.get("sharpe_ratio", 0) > 1.0:
            warnings.append("SUSPICIOUS: Low accuracy with positive Sharpe")
        
        # Flag 3: Very high Sharpe (unrealistic)
        if r.get("sharpe_ratio", 0) > 3.0:
            warnings.append("CAUTION: Sharpe > 3.0 is unusually high, verify calculation")
        
        # Flag 4: Win rate near 50% with very high Sharpe
        if 0.48 < r.get("win_rate", 0.5) < 0.52 and r.get("sharpe_ratio", 0) > 2.0:
            warnings.append("SUSPICIOUS: Coin-flip win rate with high Sharpe")
        
        # Flag 5: Zero max drawdown with positive return
        if r.get("max_drawdown", 0) == 0 and r.get("total_return", 0) > 0:
            warnings.append("SUSPICIOUS: No drawdown suggests data issue")
        
        r["validation_warnings"] = warnings
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run all trading strategies")
    parser.add_argument("--start-year", type=int, default=2025, help="Start year")
    parser.add_argument("--start-month", type=int, default=11, help="Start month")
    parser.add_argument("--end-year", type=int, default=2025, help="End year")
    parser.add_argument("--end-month", type=int, default=11, help="End month")
    parser.add_argument("--period", type=str, help="Single period (e.g., 2025_11)")
    parser.add_argument("--timeframes", type=str, default="5min,15min", 
                        help="Comma-separated timeframes")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--strategies", type=str, help="Comma-separated strategy names (default: all)")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--quick", action="store_true", help="Quick test with default period")
    
    args = parser.parse_args()
    
    # Determine periods
    if args.period:
        periods = [args.period]
    elif args.quick:
        periods = ["2025_11"]
    else:
        periods = get_periods(args.start_year, args.start_month, args.end_year, args.end_month)
    
    # Parse timeframes
    timeframes = [tf.strip() for tf in args.timeframes.split(",")]
    
    # Discover strategies
    all_strategies = discover_strategies()
    
    if args.strategies:
        selected = [s.strip() for s in args.strategies.split(",")]
        all_strategies = [(name, path) for name, path in all_strategies if name in selected]
    
    print("=" * 80)
    print("STRATEGY RUNNER")
    print("=" * 80)
    print(f"Periods:    {periods}")
    print(f"Timeframes: {timeframes}")
    print(f"Symbol:     {args.symbol}")
    print(f"Strategies: {len(all_strategies)} found")
    print("=" * 80)
    
    # Check data availability
    available_periods = []
    for period in periods:
        if check_data_availability(period, args.symbol):
            available_periods.append(period)
        else:
            print(f"WARNING: No data for period {period}, skipping")
    
    if not available_periods:
        print("ERROR: No data available for any specified period")
        sys.exit(1)
    
    # Run all strategies
    all_results = []
    total_runs = len(all_strategies) * len(available_periods) * len(timeframes)
    current_run = 0
    
    for strategy_name, strategy_path in all_strategies:
        print(f"\n{'='*60}")
        print(f"Loading: {strategy_name}")
        print(f"{'='*60}")
        
        try:
            strategy_class = load_strategy_class(strategy_path, strategy_name)
            if strategy_class is None:
                print(f"  ERROR: Could not load strategy class from {strategy_path}")
                continue
        except Exception as e:
            print(f"  ERROR loading {strategy_name}: {e}")
            continue
        
        for period in available_periods:
            for timeframe in timeframes:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] {strategy_name} | {period} | {timeframe}")
                
                result = run_single_strategy(
                    strategy_class,
                    args.symbol,
                    period,
                    timeframe,
                    verbose=args.verbose
                )
                
                if result["status"] == "success":
                    print(f"  ✓ Sharpe: {result['sharpe_ratio']:.2f} | "
                          f"Accuracy: {result['accuracy']:.3f} | "
                          f"WinRate: {result['win_rate']:.2%}")
                else:
                    print(f"  ✗ Error: {result.get('error', 'Unknown')[:50]}")
                
                all_results.append(result)
    
    # Validate results
    all_results = validate_metrics(all_results)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(format_results_table(all_results))
    
    # Print validation warnings
    warned = [r for r in all_results if r.get("validation_warnings")]
    if warned:
        print("\n" + "=" * 80)
        print("VALIDATION WARNINGS")
        print("=" * 80)
        for r in warned:
            print(f"\n{r['strategy_name']} ({r['period']}, {r['timeframe']}):")
            for w in r["validation_warnings"]:
                print(f"  ⚠ {w}")
    
    # Save results
    output_dir = Path(args.output_dir)
    json_path, csv_path = save_results(all_results, output_dir)
    
    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"JSON (detailed): {json_path}")
    print(f"CSV (summary):   {csv_path}")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    results = main()
