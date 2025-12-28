"""
Strategy Comparison Runner
==========================

Runs all strategies across multiple timeframes and compiles results.
Provides comprehensive comparison and analysis.

Usage:
    python run_strategy_comparison.py --all
    python run_strategy_comparison.py --strategy momentum --timeframe 15min
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import polars as pl

from strategies.strategy_base import StrategyResult, save_results


def run_momentum_ensemble(timeframes: List[str]) -> List[StrategyResult]:
    """Run momentum ensemble strategy."""
    from strategies.strategy01_momentum_ensemble import MomentumEnsembleStrategy
    
    results = []
    for tf in timeframes:
        for method in ["voting", "stacking"]:
            print(f"\n{'='*70}")
            print(f"Running Momentum Ensemble ({method}) - {tf}")
            print(f"{'='*70}")
            try:
                strategy = MomentumEnsembleStrategy(
                    timeframe=tf,
                    ensemble_method=method
                )
                result = strategy.run(verbose=True)
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")
    return results


def run_volatility_regime(timeframes: List[str]) -> List[StrategyResult]:
    """Run volatility regime strategy."""
    from strategies.strategy02_volatility_regime import VolatilityRegimeStrategy
    
    results = []
    for tf in timeframes:
        for method in ["gmm", "kmeans"]:
            print(f"\n{'='*70}")
            print(f"Running Volatility Regime ({method}) - {tf}")
            print(f"{'='*70}")
            try:
                strategy = VolatilityRegimeStrategy(
                    timeframe=tf,
                    regime_method=method
                )
                result = strategy.run(verbose=True)
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")
    return results


def run_orderflow(timeframes: List[str]) -> List[StrategyResult]:
    """Run order flow strategy."""
    from strategies.strategy03_orderflow import OrderFlowStrategy
    
    results = []
    for tf in timeframes:
        print(f"\n{'='*70}")
        print(f"Running Order Flow - {tf}")
        print(f"{'='*70}")
        try:
            strategy = OrderFlowStrategy(timeframe=tf)
            result = strategy.run(verbose=True)
            results.append(result)
        except Exception as e:
            print(f"Error: {e}")
    return results


def run_sentiment_fusion(timeframes: List[str]) -> List[StrategyResult]:
    """Run sentiment fusion strategy."""
    from strategies.strategy04_sentiment_fusion import SentimentFusionStrategy
    
    results = []
    for tf in timeframes:
        print(f"\n{'='*70}")
        print(f"Running Sentiment Fusion - {tf}")
        print(f"{'='*70}")
        try:
            strategy = SentimentFusionStrategy(timeframe=tf)
            result = strategy.run(verbose=True)
            results.append(result)
        except Exception as e:
            print(f"Error: {e}")
    return results


def run_pca_features(timeframes: List[str]) -> List[StrategyResult]:
    """Run PCA features strategy."""
    from strategies.strategy05_pca_features import PCAFeatureStrategy
    
    results = []
    for tf in timeframes:
        for method in ["pca", "mutual_info", "rfe", "combined"]:
            print(f"\n{'='*70}")
            print(f"Running PCA Features ({method}) - {tf}")
            print(f"{'='*70}")
            try:
                strategy = PCAFeatureStrategy(
                    timeframe=tf,
                    feature_method=method
                )
                result = strategy.run(verbose=True)
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")
    return results


def run_dollar_bars(bars_per_day_list: List[int] = [30, 50, 100]) -> List[StrategyResult]:
    """Run dollar bars strategy."""
    from strategies.strategy06_dollar_bars import DollarBarsStrategy
    
    results = []
    for bpd in bars_per_day_list:
        print(f"\n{'='*70}")
        print(f"Running Dollar Bars - {bpd} bars/day")
        print(f"{'='*70}")
        try:
            strategy = DollarBarsStrategy(target_bars_per_day=bpd)
            result = strategy.run(verbose=True)
            results.append(result)
        except Exception as e:
            print(f"Error: {e}")
    return results


def run_stacked_models(timeframes: List[str]) -> List[StrategyResult]:
    """Run stacked models strategy."""
    from strategies.strategy07_stacked_models import StackedModelsStrategy
    
    results = []
    for tf in timeframes:
        for meta in ["logistic", "xgb"]:
            print(f"\n{'='*70}")
            print(f"Running Stacked Models ({meta}) - {tf}")
            print(f"{'='*70}")
            try:
                strategy = StackedModelsStrategy(
                    timeframe=tf,
                    meta_model=meta
                )
                result = strategy.run(verbose=True)
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")
    return results


def print_comparison_table(results: List[StrategyResult]):
    """Print formatted comparison table."""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "=" * 120)
    print("STRATEGY COMPARISON RESULTS")
    print("=" * 120)
    
    # Header
    header = f"{'Strategy':<45} {'TF':<8} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}"
    print(header)
    print("-" * 120)
    
    # Sort by Sharpe ratio
    sorted_results = sorted(results, key=lambda x: x.sharpe_ratio if not np.isnan(x.sharpe_ratio) else -999, reverse=True)
    
    for r in sorted_results:
        row = (
            f"{r.strategy_name:<45} "
            f"{r.timeframe:<8} "
            f"{r.accuracy:>7.4f} "
            f"{r.f1_score:>7.4f} "
            f"{r.auc:>7.4f} "
            f"{r.total_return:>9.4f} "
            f"{r.sharpe_ratio:>8.2f} "
            f"{r.max_drawdown:>8.4f} "
            f"{r.win_rate:>8.4f}"
        )
        print(row)
    
    print("=" * 120)
    
    # Best strategy summary
    best_sharpe = sorted_results[0]
    best_accuracy = max(results, key=lambda x: x.accuracy)
    best_f1 = max(results, key=lambda x: x.f1_score)
    best_return = max(results, key=lambda x: x.total_return if not np.isnan(x.total_return) else -999)
    
    print("\nBEST PERFORMING STRATEGIES:")
    print(f"  Best Sharpe Ratio:  {best_sharpe.strategy_name} ({best_sharpe.sharpe_ratio:.4f})")
    print(f"  Best Accuracy:      {best_accuracy.strategy_name} ({best_accuracy.accuracy:.4f})")
    print(f"  Best F1 Score:      {best_f1.strategy_name} ({best_f1.f1_score:.4f})")
    print(f"  Best Total Return:  {best_return.strategy_name} ({best_return.total_return:.4f})")


def print_feature_importance_summary(results: List[StrategyResult]):
    """Print aggregated feature importance across strategies."""
    print("\n" + "=" * 80)
    print("TOP FEATURES ACROSS ALL STRATEGIES")
    print("=" * 80)
    
    # Aggregate feature importance
    all_importance = {}
    
    for r in results:
        for feat, imp in r.feature_importance.items():
            if feat not in all_importance:
                all_importance[feat] = []
            all_importance[feat].append(imp)
    
    # Calculate mean importance
    mean_importance = {
        feat: np.mean(imps) 
        for feat, imps in all_importance.items()
    }
    
    # Sort and display top features
    sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Feature':<50} {'Mean Importance':>15} {'Occurrences':>12}")
    print("-" * 80)
    
    for feat, imp in sorted_features[:30]:
        count = len(all_importance[feat])
        print(f"{feat:<50} {imp:>15.6f} {count:>12}")


def run_all_strategies(timeframes: List[str], output_dir: Optional[Path] = None):
    """Run all strategies and compile results."""
    all_results = []
    
    # Run each strategy type
    print("\n" + "#" * 80)
    print("# RUNNING ALL STRATEGIES")
    print("#" * 80)
    
    # 1. Momentum Ensemble
    all_results.extend(run_momentum_ensemble(timeframes))
    
    # 2. Volatility Regime
    all_results.extend(run_volatility_regime(timeframes))
    
    # 3. Order Flow
    all_results.extend(run_orderflow(timeframes))
    
    # 4. Sentiment Fusion
    all_results.extend(run_sentiment_fusion(timeframes))
    
    # 5. PCA Features
    all_results.extend(run_pca_features(timeframes))
    
    # 6. Dollar Bars
    all_results.extend(run_dollar_bars([30, 50, 100]))
    
    # 7. Stacked Models
    all_results.extend(run_stacked_models(timeframes))
    
    # Print comparison
    print_comparison_table(all_results)
    print_feature_importance_summary(all_results)
    
    # Save results
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"strategy_comparison_{timestamp}.json"
    save_results(all_results, output_file)
    
    return all_results


def run_single_strategy(strategy_name: str, timeframe: str) -> Optional[StrategyResult]:
    """Run a single strategy with specific parameters."""
    strategy_map = {
        "momentum": run_momentum_ensemble,
        "volatility": run_volatility_regime,
        "orderflow": run_orderflow,
        "sentiment": run_sentiment_fusion,
        "pca": run_pca_features,
        "dollar": run_dollar_bars,
        "stacked": run_stacked_models,
    }
    
    if strategy_name not in strategy_map:
        print(f"Unknown strategy: {strategy_name}")
        print(f"Available: {list(strategy_map.keys())}")
        return None
    
    if strategy_name == "dollar":
        results = run_dollar_bars([50])
    else:
        results = strategy_map[strategy_name]([timeframe])
    
    if results:
        print_comparison_table(results)
        return results[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run and compare trading strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_strategy_comparison.py --all
  python run_strategy_comparison.py --all --timeframes 15min 1hr
  python run_strategy_comparison.py --strategy momentum --timeframe 15min
  python run_strategy_comparison.py --strategy dollar
        """
    )
    
    parser.add_argument(
        "--all", action="store_true",
        help="Run all strategies"
    )
    parser.add_argument(
        "--strategy", type=str,
        choices=["momentum", "volatility", "orderflow", "sentiment", "pca", "dollar", "stacked"],
        help="Run a specific strategy"
    )
    parser.add_argument(
        "--timeframe", type=str, default="15min",
        choices=["5min", "15min", "1hr"],
        help="Timeframe for single strategy run"
    )
    parser.add_argument(
        "--timeframes", nargs="+", type=str,
        default=["5min", "15min", "1hr"],
        help="Timeframes for all strategies"
    )
    parser.add_argument(
        "--output", type=str,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    
    if args.all:
        run_all_strategies(args.timeframes, output_dir)
    elif args.strategy:
        run_single_strategy(args.strategy, args.timeframe)
    else:
        # Default: run quick comparison with 15min
        print("Running quick comparison (15min timeframe only)")
        print("Use --all for full comparison, or --help for options")
        run_all_strategies(["15min"], output_dir)


if __name__ == "__main__":
    main()
