"""
Base Strategy Framework for Quantitative Research
==================================================

Provides base classes and utilities for comparing different trading strategies.
All strategies inherit from StrategyBase for consistent evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import polars as pl
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StrategyResult:
    """Container for strategy evaluation results."""
    strategy_name: str
    timeframe: str
    bar_type: str
    
    # Data info
    n_samples: int
    n_features: int
    feature_names: List[str]
    
    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    
    # Trading metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    n_trades: int = 0
    n_days: int = 0
    
    # Additional info
    train_time_seconds: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_params: Dict[str, Any] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'strategy_name': self.strategy_name,
            'timeframe': self.timeframe,
            'bar_type': self.bar_type,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc': self.auc,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio,
            'train_time_seconds': self.train_time_seconds,
            'top_features': dict(list(self.feature_importance.items())[:20]),
        }


class StrategyBase(ABC):
    """
    Abstract base class for all trading strategies.
    
    Each strategy must implement:
    - get_feature_columns(): Define which features to use
    - create_features(): Generate features from raw data
    - train_model(): Train the ML model
    - predict(): Make predictions
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "5min",  # 5min, 15min, 1hr
        data_path: Optional[Path] = None,
        random_state: int = 42
    ):
        self.symbol = symbol
        self.period = period
        self.timeframe = timeframe
        self.random_state = random_state
        
        # Parse period for year/month
        period_parts = period.split("_")
        self.start_year = int(period_parts[0])
        self.start_month = int(period_parts[1])
        self.end_year = self.start_year
        self.end_month = self.start_month
        
        if data_path is None:
            self.data_path = Path(__file__).parent.parent / "dataset"
        else:
            self.data_path = Path(data_path)
        
        self.model = None
        self.feature_names: List[str] = []
        self.label_encoder = None
        self.scaler = None
        
    @property
    def timeframe_minutes(self) -> int:
        """Convert timeframe string to minutes."""
        mapping = {
            '1min': 1,
            '5min': 5,
            '15min': 15,
            '30min': 30,
            '1hr': 60,
            '4hr': 240,
            'dollar': 5,  # Default for dollar bars
        }
        return mapping.get(self.timeframe, 5)
    
    @property
    def bars_per_day(self) -> int:
        """Number of bars per day for this timeframe."""
        return 24 * 60 // self.timeframe_minutes
    
    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name."""
        pass
    
    @abstractmethod
    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names to use."""
        pass
    
    @abstractmethod
    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create features from OHLCV data."""
        pass
    
    @abstractmethod
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Any:
        """Train the ML model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions. Returns (predictions, probabilities)."""
        pass
    
    def load_data(self) -> pl.DataFrame:
        """Load and aggregate raw trade data."""
        from quant_features.data_loader import load_trades
        from trades_aggregation import aggregate_trades_to_ohlcv
        
        print(f"  Loading trades data...")
        trades_df = load_trades(
            self.data_path, 
            symbol=self.symbol,
            start_year=self.start_year,
            start_month=self.start_month,
            end_year=self.end_year,
            end_month=self.end_month
        )
        print(f"    Loaded {len(trades_df):,} trades")
        
        print(f"  Aggregating to {self.timeframe} bars...")
        
        # Rename columns to match aggregator expectations
        if "qty" in trades_df.columns:
            trades_df = trades_df.rename({"qty": "quantity", "quote_qty": "quote_quantity"})
        
        ohlcv = aggregate_trades_to_ohlcv(
            trades_df,
            interval_ms=self.timeframe_minutes * 60 * 1000
        )
        
        # Rename open_time to timestamp for consistency
        if "open_time" in ohlcv.columns and "timestamp" not in ohlcv.columns:
            ohlcv = ohlcv.rename({"open_time": "timestamp"})
        
        print(f"    Created {len(ohlcv):,} bars")
        
        return ohlcv
    
    def load_supplementary_data(self, ohlcv: pl.DataFrame) -> pl.DataFrame:
        """Load and merge supplementary data (metrics, funding, etc.)."""
        from quant_features.data_loader import (
            load_metrics, load_funding_rate, load_klines,
            merge_features_to_ohlcv
        )
        
        df = ohlcv
        
        # Load metrics
        try:
            metrics = load_metrics(
                self.data_path, symbol=self.symbol,
                start_year=self.start_year, start_month=self.start_month,
                end_year=self.end_year, end_month=self.end_month
            )
            df = merge_features_to_ohlcv(df, metrics, "timestamp")
            print(f"    Merged {len(metrics):,} metrics records")
        except Exception as e:
            print(f"    Warning: Could not load metrics: {e}")
        
        # Load funding rate
        try:
            funding = load_funding_rate(
                self.data_path, symbol=self.symbol,
                start_year=self.start_year, start_month=self.start_month,
                end_year=self.end_year, end_month=self.end_month
            )
            # load_funding_rate renames calc_time to time
            df = merge_features_to_ohlcv(df, funding, "time")
            print(f"    Merged {len(funding):,} funding records")
        except Exception as e:
            print(f"    Warning: Could not load funding: {e}")
        
        # Load mark price klines
        try:
            mark_klines = load_klines(
                self.data_path, symbol=self.symbol,
                kline_type="markPriceKlines",
                start_year=self.start_year, start_month=self.start_month,
                end_year=self.end_year, end_month=self.end_month
            )
            # Standard klines format: open_time, open, high, low, close columns
            if "close" in mark_klines.columns and "open_time" in mark_klines.columns:
                # Rename open_time to timestamp for merging
                mark_klines = mark_klines.select(["open_time", "close"]).rename({
                    "open_time": "timestamp", 
                    "close": "mark_price"
                })
                df = merge_features_to_ohlcv(df, mark_klines, "timestamp")
                print(f"    Merged mark price data")
        except Exception as e:
            print(f"    Warning: Could not load mark price: {e}")
        
        return df
    
    def create_labels(
        self,
        df: pl.DataFrame,
        horizon: int = None,
        threshold: float = 0.001
    ) -> pl.DataFrame:
        """Create classification labels."""
        from quant_features.labeling import (
            add_forward_returns, add_sample_weights
        )
        
        if horizon is None:
            # Use 1 bar forward for prediction
            horizon = 1
        
        df = add_forward_returns(df, periods=[horizon])
        
        # Create classification labels
        ret_col = f"fwd_ret_{horizon}"
        df = df.with_columns([
            pl.when(pl.col(ret_col) > threshold)
            .then(pl.lit(1))
            .when(pl.col(ret_col) < -threshold)
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .alias("label")
        ])
        
        # Add sample weights
        df = add_sample_weights(df, return_col=ret_col)
        
        return df
    
    def time_series_split(
        self,
        df: pl.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        gap_bars: int = None
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split data chronologically with gap to prevent look-ahead bias."""
        n = len(df)
        
        if gap_bars is None:
            # Gap = 1 day worth of bars
            gap_bars = self.bars_per_day
        
        train_end = int(n * train_ratio)
        val_start = train_end + gap_bars
        val_end = val_start + int(n * val_ratio)
        test_start = val_end + gap_bars
        
        train_df = df.slice(0, train_end)
        val_df = df.slice(val_start, val_end - val_start)
        test_df = df.slice(test_start, n - test_start)
        
        return train_df, val_df, test_df
    
    def prepare_features(
        self,
        df: pl.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract feature matrix, labels, and weights."""
        # Get available feature columns
        available = [c for c in feature_cols if c in df.columns]
        
        X = df.select(available).to_numpy()
        y = df["label"].to_numpy()
        
        if "sample_weight" in df.columns:
            w = df["sample_weight"].to_numpy()
        else:
            w = np.ones(len(y))
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.nan_to_num(w, nan=1.0)
        w = np.clip(w, 0.01, 100.0)
        
        self.feature_names = available
        
        return X, y, w
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )
        
        # Classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] > 2:
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim == 2 else y_prob)
        except Exception:
            auc = 0.5
        
        # Trading metrics
        trading_metrics = self._calculate_trading_metrics(y_true, y_pred, returns)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            **trading_metrics
        }
    
    def _calculate_trading_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate trading performance metrics with proper methodology.
        
        IMPORTANT: This uses a simple PnL model where:
        - y_pred = 1 means go long (profit if price goes up)
        - y_pred = -1 means go short (profit if price goes down)  
        - y_pred = 0 means no position
        
        Note: Sharpe ratios from short test periods (< 30 days) are unreliable.
        """
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip extreme returns (data errors)
        returns = np.clip(returns, -0.1, 0.1)  # Cap at ±10% per bar
        
        # Ensure y_pred is numeric and properly scaled
        y_pred = np.array(y_pred, dtype=float)
        
        # Clip predictions to valid range
        y_pred = np.clip(y_pred, -1, 1)
        
        # Strategy returns: position * forward_return
        strategy_returns = y_pred * returns
        
        # Count actual trades (position changes)
        positions = np.sign(y_pred)
        position_changes = np.diff(np.concatenate([[0], positions]))
        n_trades = np.sum(np.abs(position_changes) > 0)
        
        # Cumulative returns
        cum_returns = np.cumprod(1 + strategy_returns)
        total_return = cum_returns[-1] - 1 if len(cum_returns) > 0 else 0
        
        # Aggregate to daily returns for Sharpe calculation
        bars_per_day = self.bars_per_day
        n_full_days = len(strategy_returns) // bars_per_day
        
        if n_full_days >= 5:  # Need at least 5 days for any reliability
            daily_returns = []
            for i in range(n_full_days):
                start_idx = i * bars_per_day
                end_idx = (i + 1) * bars_per_day
                day_ret = np.prod(1 + strategy_returns[start_idx:end_idx]) - 1
                daily_returns.append(day_ret)
            daily_returns = np.array(daily_returns)
            
            # Sharpe ratio with small sample adjustment
            if np.std(daily_returns) > 1e-10:
                raw_sharpe = np.mean(daily_returns) / np.std(daily_returns)
                # Annualize with small sample penalty
                # For n < 30 days, apply sqrt(n/252) instead of sqrt(252)
                if n_full_days < 30:
                    sharpe = raw_sharpe * np.sqrt(n_full_days)  # Don't annualize, just scale by sample
                else:
                    sharpe = raw_sharpe * np.sqrt(252)  # Standard annualization
            else:
                sharpe = 0.0
                
            # Sortino ratio
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 0 and np.std(downside_returns) > 1e-10:
                raw_sortino = np.mean(daily_returns) / np.std(downside_returns)
                if n_full_days < 30:
                    sortino = raw_sortino * np.sqrt(n_full_days)
                else:
                    sortino = raw_sortino * np.sqrt(252)
            else:
                sortino = 0.0
        else:
            # Not enough data - return non-annualized metrics
            if np.std(strategy_returns) > 1e-10:
                sharpe = np.mean(strategy_returns) / np.std(strategy_returns)
            else:
                sharpe = 0.0
            sortino = 0.0
        
        # Max drawdown
        peak = np.maximum.accumulate(cum_returns)
        drawdown = np.where(peak > 0, (peak - cum_returns) / peak, 0)
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Win rate: proportion of active bars that were profitable
        active_mask = y_pred != 0
        if np.sum(active_mask) > 0:
            winning_bars = np.sum((strategy_returns > 0) & active_mask)
            total_active_bars = np.sum(active_mask)
            win_rate = winning_bars / total_active_bars
        else:
            win_rate = 0.0
        
        # Profit factor
        gross_profit = np.sum(strategy_returns[strategy_returns > 0])
        gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else 0
        
        # Calmar ratio (return / max drawdown, not annualized for short periods)
        calmar = total_return / max_dd if max_dd > 1e-10 else 0.0
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'calmar_ratio': float(calmar),
            'sortino_ratio': float(sortino),
            'n_trades': int(n_trades),
            'n_days': int(n_full_days),
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self.model is None:
            return {}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'feature_importance'):
                importances = self.model.feature_importance()
            else:
                return {}
            
            importance_dict = dict(zip(self.feature_names, importances))
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            return {}
    
    def run(self, verbose: bool = True) -> StrategyResult:
        """Execute the full strategy pipeline."""
        import time
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"STRATEGY: {self.get_name()}")
            print(f"Timeframe: {self.timeframe}")
            print(f"{'='*70}")
        
        start_time = time.time()
        
        # Step 1: Load data
        if verbose:
            print("\n[STEP 1] Loading data...")
        ohlcv = self.load_data()
        df = self.load_supplementary_data(ohlcv)
        
        # Step 2: Create features
        if verbose:
            print("\n[STEP 2] Creating features...")
        df = self.create_features(df)
        feature_cols = self.get_feature_columns()
        available_features = [c for c in feature_cols if c in df.columns]
        if verbose:
            print(f"  Requested features: {len(feature_cols)}")
            print(f"  Available features: {len(available_features)}")
        
        # Step 3: Create labels
        if verbose:
            print("\n[STEP 3] Creating labels...")
        # Use smaller threshold for more balanced classes
        df = self.create_labels(df, horizon=1, threshold=0.0001)
        
        # Step 4: Split data
        if verbose:
            print("\n[STEP 4] Splitting data...")
        train_df, val_df, test_df = self.time_series_split(df)
        if verbose:
            print(f"  Train: {len(train_df):,}")
            print(f"  Val:   {len(val_df):,}")
            print(f"  Test:  {len(test_df):,}")
        
        # Step 5: Prepare features
        if verbose:
            print("\n[STEP 5] Preparing features...")
        X_train, y_train, w_train = self.prepare_features(train_df, available_features)
        X_val, y_val, w_val = self.prepare_features(val_df, available_features)
        X_test, y_test, w_test = self.prepare_features(test_df, available_features)
        
        if verbose:
            print(f"  Feature matrix shape: {X_train.shape}")
        
        # Step 6: Train model
        if verbose:
            print("\n[STEP 6] Training model...")
        self.model = self.train_model(X_train, y_train, X_val, y_val, w_train)
        
        train_time = time.time() - start_time
        
        # Step 7: Evaluate
        if verbose:
            print("\n[STEP 7] Evaluating...")
        y_pred, y_prob = self.predict(X_test)
        
        # Get forward returns for trading metrics
        if "fwd_ret_1" in test_df.columns:
            test_returns = test_df["fwd_ret_1"].to_numpy()
        else:
            test_returns = np.zeros(len(y_test))
        test_returns = np.nan_to_num(test_returns, nan=0.0)
        
        metrics = self.evaluate(y_test, y_pred, y_prob, test_returns)
        
        # Get feature importance
        importance = self.get_feature_importance()
        
        # Create result
        result = StrategyResult(
            strategy_name=self.get_name(),
            timeframe=self.timeframe,
            bar_type="time",
            n_samples=len(test_df),
            n_features=len(self.feature_names),
            feature_names=self.feature_names,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            auc=metrics['auc'],
            total_return=metrics['total_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            calmar_ratio=metrics['calmar_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            n_trades=metrics.get('n_trades', 0),
            n_days=metrics.get('n_days', 0),
            train_time_seconds=train_time,
            feature_importance=importance,
            confusion_matrix=metrics.get('confusion_matrix'),
        )
        
        if verbose:
            self._print_results(result)
        
        return result
    
    def _print_results(self, result: StrategyResult):
        """Print formatted results."""
        print(f"\n{'='*70}")
        print(f"RESULTS: {result.strategy_name}")
        print(f"{'='*70}")
        
        print("\nClassification Metrics:")
        print(f"  Accuracy:  {result.accuracy:.4f}")
        print(f"  Precision: {result.precision:.4f}")
        print(f"  Recall:    {result.recall:.4f}")
        print(f"  F1 Score:  {result.f1_score:.4f}")
        print(f"  AUC:       {result.auc:.4f}")
        
        print("\nTrading Metrics:")
        print(f"  Total Return:  {result.total_return:.4f}")
        print(f"  Sharpe Ratio:  {result.sharpe_ratio:.4f}")
        print(f"  Max Drawdown:  {result.max_drawdown:.4f}")
        print(f"  Win Rate:      {result.win_rate:.4f}")
        print(f"  Profit Factor: {result.profit_factor:.4f}")
        print(f"  Calmar Ratio:  {result.calmar_ratio:.4f}")
        print(f"  Sortino Ratio: {result.sortino_ratio:.4f}")
        
        print("\nTop 10 Features:")
        for i, (feat, imp) in enumerate(list(result.feature_importance.items())[:10], 1):
            print(f"  {i:2d}. {feat:<40} {imp:.4f}")


def remove_correlated_features(
    X: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.95
) -> Tuple[np.ndarray, List[str]]:
    """Remove highly correlated features."""
    corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # Find features to drop
    to_drop = set()
    n_features = len(feature_names)
    
    for i in range(n_features):
        if i in to_drop:
            continue
        for j in range(i + 1, n_features):
            if j in to_drop:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                to_drop.add(j)
    
    # Keep features not in to_drop
    keep_idx = [i for i in range(n_features) if i not in to_drop]
    
    return X[:, keep_idx], [feature_names[i] for i in keep_idx]


def save_results(results: List[StrategyResult], output_path: Path):
    """Save strategy results to JSON."""
    data = [r.to_dict() for r in results]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
