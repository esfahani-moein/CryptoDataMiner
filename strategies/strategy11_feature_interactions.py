"""
Strategy 11: Feature Interactions - Polynomial and Cross-Feature Engineering
================================================================================
Creates polynomial features, feature ratios, and interaction terms to capture
non-linear relationships between market indicators.
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_classif

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from strategies.strategy_base import StrategyBase, StrategyResult


class FeatureInteractionsStrategy(StrategyBase):
    """
    Creates polynomial and interaction features for improved prediction.
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        n_poly_features: int = 50,
        **kwargs
    ):
        super().__init__(symbol, period, timeframe, **kwargs)
        self.n_poly_features = n_poly_features
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.poly = None
        self.selector = None
    
    def get_name(self) -> str:
        return f"Feature_Interactions_{self.n_poly_features}poly_{self.timeframe}"
    
    def get_feature_columns(self) -> List[str]:
        return [
            # Base features
            'ret_1', 'ret_3', 'ret_5',
            'vol_5', 'vol_10', 'vol_20',
            'momentum_3', 'momentum_10',
            'rsi_14',
            
            # Ratios
            'ret_vol_ratio_3', 'ret_vol_ratio_5',
            'vol_term_structure',
            'momentum_vol_ratio',
            
            # Squared
            'ret_1_sq', 'ret_3_sq',
            'vol_5_sq', 'vol_10_sq',
            'momentum_3_sq',
            
            # Cross terms
            'ret_x_vol', 'ret_x_mom',
            'vol_x_mom', 'vol_x_rsi',
            
            # Triple interactions
            'ret_vol_mom', 'vol_mom_rsi',
            
            # Trend interactions
            'trend_x_vol', 'trend_x_mom',
            'trend_x_ret',
            
            # Range features
            'high_low_range', 'range_x_vol',
            'close_position', 'close_pos_x_vol',
            
            # Lag interactions
            'ret1_x_ret3', 'ret3_x_ret5',
            'vol5_x_vol10', 'vol10_x_vol20',
            
            # Sign features
            'ret_sign', 'ret_sign_x_vol',
            'momentum_sign', 'mom_sign_x_vol',
            
            # Normalized
            'ret_zscore', 'vol_zscore', 'mom_zscore',
            'zscore_product',
        ]
    
    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create interaction features."""
        
        # === BASE FEATURES ===
        for period in [1, 3, 5]:
            df = df.with_columns([
                pl.col("close").pct_change(period).alias(f"ret_{period}")
            ])
        
        for period in [5, 10, 20]:
            df = df.with_columns([
                pl.col("ret_1").rolling_std(period).alias(f"vol_{period}")
            ])
        
        for period in [3, 10]:
            df = df.with_columns([
                (pl.col("close") / pl.col("close").shift(period) - 1).alias(f"momentum_{period}")
            ])
        
        # RSI
        df = df.with_columns([
            pl.col("ret_1").alias("_change"),
        ])
        df = df.with_columns([
            pl.when(pl.col("_change") > 0).then(pl.col("_change")).otherwise(0.0).alias("_gain"),
            pl.when(pl.col("_change") < 0).then(pl.col("_change").abs()).otherwise(0.0).alias("_loss"),
        ])
        df = df.with_columns([
            pl.col("_gain").rolling_mean(14).alias("_avg_gain"),
            pl.col("_loss").rolling_mean(14).alias("_avg_loss"),
        ])
        df = df.with_columns([
            (100 - 100 / (1 + pl.col("_avg_gain") / (pl.col("_avg_loss") + 1e-10))).alias("rsi_14")
        ])
        
        # === RATIOS ===
        df = df.with_columns([
            (pl.col("ret_3").abs() / (pl.col("vol_5") + 1e-10)).alias("ret_vol_ratio_3"),
            (pl.col("ret_5").abs() / (pl.col("vol_10") + 1e-10)).alias("ret_vol_ratio_5"),
            (pl.col("vol_5") / (pl.col("vol_20") + 1e-10)).alias("vol_term_structure"),
            (pl.col("momentum_3").abs() / (pl.col("vol_5") + 1e-10)).alias("momentum_vol_ratio"),
        ])
        
        # === SQUARED TERMS ===
        df = df.with_columns([
            (pl.col("ret_1") ** 2).alias("ret_1_sq"),
            (pl.col("ret_3") ** 2).alias("ret_3_sq"),
            (pl.col("vol_5") ** 2).alias("vol_5_sq"),
            (pl.col("vol_10") ** 2).alias("vol_10_sq"),
            (pl.col("momentum_3") ** 2).alias("momentum_3_sq"),
        ])
        
        # === CROSS TERMS ===
        df = df.with_columns([
            (pl.col("ret_3") * pl.col("vol_10")).alias("ret_x_vol"),
            (pl.col("ret_3") * pl.col("momentum_3")).alias("ret_x_mom"),
            (pl.col("vol_5") * pl.col("momentum_3")).alias("vol_x_mom"),
            (pl.col("vol_10") * (pl.col("rsi_14") - 50) / 50).alias("vol_x_rsi"),
        ])
        
        # === TRIPLE INTERACTIONS ===
        df = df.with_columns([
            (pl.col("ret_3") * pl.col("vol_5") * pl.col("momentum_3")).alias("ret_vol_mom"),
            (pl.col("vol_5") * pl.col("momentum_3") * (pl.col("rsi_14") - 50) / 50).alias("vol_mom_rsi"),
        ])
        
        # === TREND FEATURES ===
        df = df.with_columns([
            pl.col("close").rolling_mean(10).alias("sma_10"),
            pl.col("close").rolling_mean(30).alias("sma_30"),
        ])
        df = df.with_columns([
            ((pl.col("sma_10") / pl.col("sma_30")) - 1).alias("trend")
        ])
        
        df = df.with_columns([
            (pl.col("trend") * pl.col("vol_10")).alias("trend_x_vol"),
            (pl.col("trend") * pl.col("momentum_10")).alias("trend_x_mom"),
            (pl.col("trend") * pl.col("ret_3")).alias("trend_x_ret"),
        ])
        
        # === RANGE FEATURES ===
        df = df.with_columns([
            ((pl.col("high") - pl.col("low")) / (pl.col("close") + 1e-10)).alias("high_low_range"),
        ])
        df = df.with_columns([
            (pl.col("high_low_range") * pl.col("vol_10")).alias("range_x_vol"),
        ])
        
        df = df.with_columns([
            ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-10)).alias("close_position"),
        ])
        df = df.with_columns([
            (pl.col("close_position") * pl.col("vol_10")).alias("close_pos_x_vol"),
        ])
        
        # === LAG INTERACTIONS ===
        df = df.with_columns([
            (pl.col("ret_1") * pl.col("ret_3")).alias("ret1_x_ret3"),
            (pl.col("ret_3") * pl.col("ret_5")).alias("ret3_x_ret5"),
            (pl.col("vol_5") * pl.col("vol_10")).alias("vol5_x_vol10"),
            (pl.col("vol_10") * pl.col("vol_20")).alias("vol10_x_vol20"),
        ])
        
        # === SIGN FEATURES ===
        df = df.with_columns([
            pl.when(pl.col("ret_1") > 0).then(1).when(pl.col("ret_1") < 0).then(-1).otherwise(0).alias("ret_sign"),
            pl.when(pl.col("momentum_3") > 0).then(1).when(pl.col("momentum_3") < 0).then(-1).otherwise(0).alias("momentum_sign"),
        ])
        df = df.with_columns([
            (pl.col("ret_sign").cast(pl.Float64) * pl.col("vol_5")).alias("ret_sign_x_vol"),
            (pl.col("momentum_sign").cast(pl.Float64) * pl.col("vol_5")).alias("mom_sign_x_vol"),
        ])
        
        # === NORMALIZED/ZSCORE FEATURES ===
        df = df.with_columns([
            (pl.col("ret_3") / (pl.col("ret_3").rolling_std(20) + 1e-10)).alias("ret_zscore"),
            ((pl.col("vol_5") - pl.col("vol_5").rolling_mean(20)) / (pl.col("vol_5").rolling_std(20) + 1e-10)).alias("vol_zscore"),
            (pl.col("momentum_3") / (pl.col("momentum_3").rolling_std(20) + 1e-10)).alias("mom_zscore"),
        ])
        df = df.with_columns([
            (pl.col("ret_zscore") * pl.col("vol_zscore") * pl.col("mom_zscore")).alias("zscore_product"),
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
        """Train with additional polynomial features."""
        
        y_train_enc = self.label_encoder.fit_transform(y_train)
        
        # Scale base features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Add polynomial features (only degree 2 interactions, no bias)
        self.poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        
        # Only use first 10 features for polynomial to avoid explosion
        n_base = min(10, X_train_scaled.shape[1])
        X_train_poly = self.poly.fit_transform(X_train_scaled[:, :n_base])
        
        # Combine base + polynomial
        X_train_combined = np.hstack([X_train_scaled, X_train_poly[:, n_base:]])
        
        # Feature selection
        print(f"  Combined features: {X_train_combined.shape[1]} (base: {X_train_scaled.shape[1]}, poly: {X_train_poly.shape[1] - n_base})")
        
        if X_train_combined.shape[1] > self.n_poly_features:
            self.selector = SelectKBest(mutual_info_classif, k=self.n_poly_features)
            X_train_selected = self.selector.fit_transform(X_train_combined, y_train_enc)
        else:
            X_train_selected = X_train_combined
        
        print(f"  Selected features: {X_train_selected.shape[1]}")
        
        # Train model
        if HAS_LGB:
            model = lgb.LGBMClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.03,
                num_leaves=31,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
                importance_type='gain'
            )
        else:
            model = ExtraTreesClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        
        if sample_weights is not None:
            model.fit(X_train_selected, y_train_enc, sample_weight=sample_weights)
        else:
            model.fit(X_train_selected, y_train_enc)
        
        return model
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with polynomial features."""
        X_scaled = self.scaler.transform(X)
        
        n_base = min(10, X_scaled.shape[1])
        X_poly = self.poly.transform(X_scaled[:, :n_base])
        X_combined = np.hstack([X_scaled, X_poly[:, n_base:]])
        
        if self.selector is not None:
            X_selected = self.selector.transform(X_combined)
        else:
            X_selected = X_combined
        
        predictions = self.model.predict(X_selected)
        probas = self.model.predict_proba(X_selected)
        
        predictions = self.label_encoder.inverse_transform(predictions)
        return predictions, probas
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        importance = {}
        
        if hasattr(self.model, 'feature_importances_'):
            # Map importance back to original features
            n_base = len(self.feature_names)
            
            for i, imp in enumerate(self.model.feature_importances_):
                if i < n_base:
                    importance[self.feature_names[i]] = float(imp)
                else:
                    importance[f"poly_feature_{i}"] = float(imp)
        
        return importance


def run_strategy(timeframe: str = "15min", n_poly: int = 50) -> StrategyResult:
    strategy = FeatureInteractionsStrategy(
        symbol="BTCUSDT",
        period="2025_11",
        timeframe=timeframe,
        n_poly_features=n_poly,
    )
    return strategy.run(verbose=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Feature Interactions Strategy")
    parser.add_argument("--timeframe", type=str, default="15min", choices=["5min", "15min", "1hr"])
    parser.add_argument("--poly", type=int, default=50)
    args = parser.parse_args()
    result = run_strategy(args.timeframe, args.poly)
    print(f"\nFinal Sharpe Ratio: {result.sharpe_ratio:.4f}")
