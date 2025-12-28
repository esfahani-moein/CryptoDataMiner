"""
Strategy 12: Probability Trading - Kelly Criterion Position Sizing
================================================================================
Uses calibrated probability predictions with Kelly criterion for optimal
position sizing, focusing on expected value rather than classification accuracy.
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from strategies.strategy_base import StrategyBase, StrategyResult


class ProbabilityTradingStrategy(StrategyBase):
    """
    Probability-based trading with Kelly criterion position sizing.
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        kelly_fraction: float = 0.25,  # Conservative Kelly
        min_prob: float = 0.55,  # Minimum probability to trade
        **kwargs
    ):
        super().__init__(symbol, period, timeframe, **kwargs)
        self.kelly_fraction = kelly_fraction
        self.min_prob = min_prob
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.calibrated_model = None
    
    def get_name(self) -> str:
        return f"Prob_Trading_Kelly{self.kelly_fraction:.0%}_{self.timeframe}"
    
    def get_feature_columns(self) -> List[str]:
        return [
            # Returns at multiple horizons
            'ret_1', 'ret_3', 'ret_5', 'ret_10',
            'cumret_10', 'cumret_20',
            
            # Volatility
            'vol_5', 'vol_10', 'vol_20',
            'realized_vol', 'vol_of_vol',
            
            # Risk-adjusted metrics
            'sharpe_10', 'sharpe_20',
            'sortino_10',
            'calmar_10',
            
            # Momentum quality
            'mom_strength', 'mom_consistency',
            'up_ratio_10', 'down_ratio_10',
            
            # Probability inputs
            'win_rate_10', 'win_rate_20',
            'avg_win', 'avg_loss',
            'expected_value',
            
            # Technical
            'rsi_14', 'rsi_zscore',
            'macd_signal', 'macd_hist',
            
            # Volume
            'volume_ratio', 'volume_trend',
            
            # Spread/cost
            'spread_estimate', 'cost_adjusted_ret',
        ]
    
    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create probability-focused features."""
        
        # === RETURNS ===
        for period in [1, 3, 5, 10]:
            df = df.with_columns([
                pl.col("close").pct_change(period).alias(f"ret_{period}")
            ])
        
        for period in [10, 20]:
            df = df.with_columns([
                (pl.col("close") / pl.col("close").shift(period) - 1).alias(f"cumret_{period}")
            ])
        
        # === VOLATILITY ===
        for period in [5, 10, 20]:
            df = df.with_columns([
                pl.col("ret_1").rolling_std(period).alias(f"vol_{period}")
            ])
        
        df = df.with_columns([
            (pl.col("vol_10") * np.sqrt(252 * 24)).alias("realized_vol"),  # Annualized
        ])
        
        df = df.with_columns([
            pl.col("vol_10").rolling_std(10).alias("vol_of_vol"),
        ])
        
        # === RISK-ADJUSTED METRICS ===
        df = df.with_columns([
            (pl.col("cumret_10") / (pl.col("vol_10") * np.sqrt(10) + 1e-10)).alias("sharpe_10"),
            (pl.col("cumret_20") / (pl.col("vol_20") * np.sqrt(20) + 1e-10)).alias("sharpe_20"),
        ])
        
        # Sortino (downside only)
        df = df.with_columns([
            pl.when(pl.col("ret_1") < 0).then(pl.col("ret_1") ** 2).otherwise(0.0).alias("_neg_sq")
        ])
        df = df.with_columns([
            pl.col("_neg_sq").rolling_mean(10).sqrt().alias("_downside_vol")
        ])
        df = df.with_columns([
            (pl.col("cumret_10") / (pl.col("_downside_vol") + 1e-10)).alias("sortino_10")
        ])
        
        # Calmar-like (return / max drawdown proxy)
        df = df.with_columns([
            pl.col("close").rolling_max(10).alias("_rolling_max")
        ])
        df = df.with_columns([
            ((pl.col("_rolling_max") - pl.col("close")) / pl.col("_rolling_max")).alias("_drawdown")
        ])
        df = df.with_columns([
            pl.col("_drawdown").rolling_max(10).alias("_max_dd")
        ])
        df = df.with_columns([
            (pl.col("cumret_10") / (pl.col("_max_dd") + 1e-10)).alias("calmar_10")
        ])
        
        # === MOMENTUM QUALITY ===
        df = df.with_columns([
            (pl.col("cumret_10").abs() / (pl.col("vol_10") * np.sqrt(10) + 1e-10)).alias("mom_strength"),
        ])
        
        df = df.with_columns([
            pl.when(pl.col("ret_1") > 0).then(1).otherwise(0).alias("_up")
        ])
        df = df.with_columns([
            pl.col("_up").rolling_mean(10).alias("up_ratio_10"),
            (1 - pl.col("_up")).rolling_mean(10).alias("down_ratio_10"),
        ])
        
        df = df.with_columns([
            pl.when(pl.col("cumret_10") > 0)
            .then(pl.col("up_ratio_10") - 0.5)
            .otherwise(pl.col("down_ratio_10") - 0.5)
            .alias("mom_consistency")
        ])
        
        # === WIN RATE FEATURES ===
        for period in [10, 20]:
            df = df.with_columns([
                pl.when(pl.col("ret_1") > 0).then(1).otherwise(0).rolling_mean(period).alias(f"win_rate_{period}")
            ])
        
        df = df.with_columns([
            pl.when(pl.col("ret_1") > 0).then(pl.col("ret_1")).otherwise(0.0).alias("_wins"),
            pl.when(pl.col("ret_1") < 0).then(pl.col("ret_1").abs()).otherwise(0.0).alias("_losses"),
        ])
        df = df.with_columns([
            pl.col("_wins").rolling_mean(10).alias("avg_win"),
            pl.col("_losses").rolling_mean(10).alias("avg_loss"),
        ])
        
        # Expected value = win_rate * avg_win - (1 - win_rate) * avg_loss
        df = df.with_columns([
            (pl.col("win_rate_10") * pl.col("avg_win") - 
             (1 - pl.col("win_rate_10")) * pl.col("avg_loss")).alias("expected_value")
        ])
        
        # === RSI ===
        df = df.with_columns([
            pl.when(pl.col("ret_1") > 0).then(pl.col("ret_1")).otherwise(0.0).alias("_gain"),
            pl.when(pl.col("ret_1") < 0).then(pl.col("ret_1").abs()).otherwise(0.0).alias("_loss"),
        ])
        df = df.with_columns([
            pl.col("_gain").rolling_mean(14).alias("_avg_gain"),
            pl.col("_loss").rolling_mean(14).alias("_avg_loss"),
        ])
        df = df.with_columns([
            (100 - 100 / (1 + pl.col("_avg_gain") / (pl.col("_avg_loss") + 1e-10))).alias("rsi_14")
        ])
        df = df.with_columns([
            ((pl.col("rsi_14") - 50) / (pl.col("rsi_14").rolling_std(20) + 1e-10)).alias("rsi_zscore")
        ])
        
        # === MACD ===
        df = df.with_columns([
            pl.col("close").ewm_mean(span=12, adjust=False).alias("_ema12"),
            pl.col("close").ewm_mean(span=26, adjust=False).alias("_ema26"),
        ])
        df = df.with_columns([
            (pl.col("_ema12") - pl.col("_ema26")).alias("_macd")
        ])
        df = df.with_columns([
            pl.col("_macd").ewm_mean(span=9, adjust=False).alias("macd_signal")
        ])
        df = df.with_columns([
            (pl.col("_macd") - pl.col("macd_signal")).alias("macd_hist")
        ])
        
        # === VOLUME ===
        df = df.with_columns([
            (pl.col("volume") / (pl.col("volume").rolling_mean(20) + 1e-10)).alias("volume_ratio"),
            (pl.col("volume").rolling_mean(5) / (pl.col("volume").rolling_mean(20) + 1e-10)).alias("volume_trend"),
        ])
        
        # === SPREAD/COST ===
        df = df.with_columns([
            ((pl.col("high") - pl.col("low")) / (pl.col("close") + 1e-10) * 0.1).alias("spread_estimate"),
        ])
        df = df.with_columns([
            (pl.col("ret_1") - pl.col("spread_estimate")).alias("cost_adjusted_ret"),
        ])
        
        return df
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Any:
        """Train calibrated probability model."""
        
        y_train_enc = self.label_encoder.fit_transform(y_train)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Base model
        if HAS_LGB:
            base_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=30,
                random_state=42,
                verbose=-1
            )
        else:
            base_model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.05,
                random_state=42
            )
        
        print("  Training base model...")
        if sample_weights is not None:
            base_model.fit(X_train_scaled, y_train_enc, sample_weight=sample_weights)
        else:
            base_model.fit(X_train_scaled, y_train_enc)
        
        # Calibrate probabilities using isotonic regression
        print("  Calibrating probabilities...")
        self.calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method='isotonic',
            cv=3,
            n_jobs=-1
        )
        
        try:
            self.calibrated_model.fit(X_train_scaled, y_train_enc)
            print("  Calibration successful")
        except Exception as e:
            print(f"  Calibration failed: {e}, using uncalibrated model")
            self.calibrated_model = base_model
        
        # Store base model for feature importance
        self.base_model = base_model
        
        return self.calibrated_model
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with position sizing based on Kelly criterion."""
        
        X_scaled = self.scaler.transform(X)
        
        # Get calibrated probabilities
        probas = self.model.predict_proba(X_scaled)
        
        # Calculate Kelly-optimal position sizes
        max_probs = np.max(probas, axis=1)
        pred_classes = np.argmax(probas, axis=1)
        
        # Kelly fraction: f = (bp - q) / b where b=1 (even odds assumed)
        # f = p - q = 2p - 1
        kelly_fractions = (2 * max_probs - 1) * self.kelly_fraction
        kelly_fractions = np.clip(kelly_fractions, 0, 1)
        
        # Only trade when probability exceeds threshold
        low_conf_mask = max_probs < self.min_prob
        
        # Find hold class if exists
        hold_class = None
        for i, cls in enumerate(self.label_encoder.classes_):
            if cls == 0:
                hold_class = i
                break
        
        if hold_class is not None:
            pred_classes[low_conf_mask] = hold_class
        
        predictions = self.label_encoder.inverse_transform(pred_classes)
        
        # Add position sizing to probabilities (encoded in first column)
        probas_with_sizing = probas.copy()
        probas_with_sizing[:, 0] = kelly_fractions  # Store sizing info
        
        return predictions, probas
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from base model."""
        importance = {}
        
        if hasattr(self, 'base_model') and hasattr(self.base_model, 'feature_importances_'):
            for i, imp in enumerate(self.base_model.feature_importances_):
                if i < len(self.feature_names):
                    importance[self.feature_names[i]] = float(imp)
        
        return importance
    
    def calculate_kelly_position(self, prob: float, win_loss_ratio: float = 1.0) -> float:
        """
        Calculate optimal Kelly position size.
        
        f* = (p * b - q) / b
        where p = win probability, q = 1-p, b = win/loss ratio
        """
        if prob <= 0.5:
            return 0.0
        
        q = 1 - prob
        kelly = (prob * win_loss_ratio - q) / win_loss_ratio
        
        # Apply fractional Kelly for safety
        return max(0, kelly * self.kelly_fraction)


def run_strategy(timeframe: str = "15min", kelly: float = 0.25, min_prob: float = 0.55) -> StrategyResult:
    strategy = ProbabilityTradingStrategy(
        symbol="BTCUSDT",
        period="2025_11",
        timeframe=timeframe,
        kelly_fraction=kelly,
        min_prob=min_prob,
    )
    return strategy.run(verbose=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Probability Trading Strategy")
    parser.add_argument("--timeframe", type=str, default="15min", choices=["5min", "15min", "1hr"])
    parser.add_argument("--kelly", type=float, default=0.25)
    parser.add_argument("--min-prob", type=float, default=0.55)
    args = parser.parse_args()
    result = run_strategy(args.timeframe, args.kelly, args.min_prob)
    print(f"\nFinal Sharpe Ratio: {result.sharpe_ratio:.4f}")
