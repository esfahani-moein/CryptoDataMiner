"""
Strategy 13: Optimized Binary Classification
=============================================
Uses binary classification (Long/Short only) with:
1. Focal loss-like weighting for minority class
2. Feature selection based on mutual information
3. Calibrated probability thresholds
4. Robust labeling with adaptive thresholds
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

from strategies.strategy_base import StrategyBase, StrategyResult


class OptimizedBinaryStrategy(StrategyBase):
    """
    Binary classification strategy with optimized feature selection.
    
    Key differences from other strategies:
    1. Binary classification (no hold class) for clearer signals
    2. Adaptive threshold based on volatility regime
    3. Feature selection based on information gain
    4. Out-of-fold probability calibration
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        n_features: int = 25,
        probability_threshold: float = 0.55,
        **kwargs
    ):
        super().__init__(symbol, period, timeframe, **kwargs)
        self.n_features = n_features
        self.probability_threshold = probability_threshold
        self.label_encoder = LabelEncoder()
        self.scaler = RobustScaler()
        self.selector = None
        self.selected_features = []
    
    def get_name(self) -> str:
        return f"Optimized_Binary_{self.n_features}feat_{self.timeframe}"
    
    def get_feature_columns(self) -> List[str]:
        return [
            # Core momentum
            'ret_1', 'ret_3', 'ret_5', 'ret_10',
            'momentum_5', 'momentum_10',
            
            # Volatility
            'vol_5', 'vol_10', 'vol_ratio',
            'atr_norm', 'vol_regime',
            
            # Mean reversion
            'zscore_10', 'zscore_20',
            'bb_position',
            
            # Trend
            'trend_strength', 'trend_direction',
            'ema_diff',
            
            # Volume
            'volume_ratio', 'volume_trend',
            
            # Order flow (if available)
            'taker_buy_ratio', 'volume_imbalance',
            
            # Microstructure
            'trade_intensity', 'avg_trade_size_norm',
            
            # Interactions
            'mom_vol_cross', 'trend_vol_cross',
            
            # Lagged signals
            'ret_1_lag1', 'ret_1_lag2',
            'vol_5_lag1',
        ]
    
    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create optimized feature set."""
        
        # === CORE MOMENTUM ===
        for period in [1, 3, 5, 10]:
            df = df.with_columns([
                pl.col("close").pct_change(period).alias(f"ret_{period}")
            ])
        
        for period in [5, 10]:
            df = df.with_columns([
                (pl.col("close") / pl.col("close").shift(period) - 1).alias(f"momentum_{period}")
            ])
        
        # === VOLATILITY ===
        for period in [5, 10]:
            df = df.with_columns([
                pl.col("ret_1").rolling_std(period).alias(f"vol_{period}")
            ])
        
        df = df.with_columns([
            (pl.col("vol_5") / (pl.col("vol_10") + 1e-10)).alias("vol_ratio"),
        ])
        
        # ATR normalized by price
        df = df.with_columns([
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs()
            ).alias("_tr")
        ])
        df = df.with_columns([
            (pl.col("_tr").rolling_mean(14) / pl.col("close")).alias("atr_norm")
        ])
        
        # Volatility regime (high/low)
        df = df.with_columns([
            pl.col("vol_10").rolling_mean(50).alias("_vol_ma")
        ])
        df = df.with_columns([
            pl.when(pl.col("vol_10") > pl.col("_vol_ma")).then(1).otherwise(0).alias("vol_regime")
        ])
        
        # === MEAN REVERSION ===
        for period in [10, 20]:
            df = df.with_columns([
                pl.col("close").rolling_mean(period).alias(f"_ma_{period}"),
                pl.col("close").rolling_std(period).alias(f"_std_{period}"),
            ])
            df = df.with_columns([
                ((pl.col("close") - pl.col(f"_ma_{period}")) / 
                 (pl.col(f"_std_{period}") + 1e-10)).alias(f"zscore_{period}")
            ])
        
        # Bollinger position
        df = df.with_columns([
            (pl.col("_ma_20") + 2 * pl.col("_std_20")).alias("bb_upper"),
            (pl.col("_ma_20") - 2 * pl.col("_std_20")).alias("bb_lower"),
        ])
        df = df.with_columns([
            ((pl.col("close") - pl.col("bb_lower")) / 
             (pl.col("bb_upper") - pl.col("bb_lower") + 1e-10)).alias("bb_position")
        ])
        
        # === TREND ===
        df = df.with_columns([
            pl.col("close").rolling_mean(10).alias("sma_10"),
            pl.col("close").rolling_mean(30).alias("sma_30"),
            pl.col("close").ewm_mean(span=10, adjust=False).alias("ema_10"),
            pl.col("close").ewm_mean(span=30, adjust=False).alias("ema_30"),
        ])
        df = df.with_columns([
            ((pl.col("sma_10") - pl.col("sma_30")) / (pl.col("sma_30") + 1e-10)).alias("trend_strength"),
            pl.when(pl.col("sma_10") > pl.col("sma_30")).then(1).otherwise(-1).alias("trend_direction"),
            ((pl.col("ema_10") - pl.col("ema_30")) / (pl.col("ema_30") + 1e-10)).alias("ema_diff"),
        ])
        
        # === VOLUME ===
        df = df.with_columns([
            (pl.col("volume") / (pl.col("volume").rolling_mean(20) + 1e-10)).alias("volume_ratio"),
            (pl.col("volume").rolling_mean(5) / (pl.col("volume").rolling_mean(20) + 1e-10)).alias("volume_trend"),
        ])
        
        # === ORDER FLOW (if available) ===
        if "taker_buy_volume" in df.columns:
            df = df.with_columns([
                (pl.col("taker_buy_volume") / (pl.col("volume") + 1e-10)).alias("taker_buy_ratio"),
                ((pl.col("taker_buy_volume") - (pl.col("volume") - pl.col("taker_buy_volume"))) / 
                 (pl.col("volume") + 1e-10)).alias("volume_imbalance"),
            ])
        else:
            df = df.with_columns([
                pl.lit(0.5).alias("taker_buy_ratio"),
                pl.lit(0.0).alias("volume_imbalance"),
            ])
        
        # === MICROSTRUCTURE ===
        if "count" in df.columns:
            df = df.with_columns([
                (pl.col("count") / (pl.col("count").rolling_mean(20) + 1)).alias("trade_intensity"),
                ((pl.col("volume") / (pl.col("count") + 1)) / 
                 ((pl.col("volume") / (pl.col("count") + 1)).rolling_mean(20) + 1e-10)).alias("avg_trade_size_norm"),
            ])
        else:
            df = df.with_columns([
                pl.lit(1.0).alias("trade_intensity"),
                pl.lit(1.0).alias("avg_trade_size_norm"),
            ])
        
        # === INTERACTIONS ===
        df = df.with_columns([
            (pl.col("momentum_5") * pl.col("vol_5")).alias("mom_vol_cross"),
            (pl.col("trend_strength") * pl.col("vol_ratio")).alias("trend_vol_cross"),
        ])
        
        # === LAGGED FEATURES ===
        df = df.with_columns([
            pl.col("ret_1").shift(1).alias("ret_1_lag1"),
            pl.col("ret_1").shift(2).alias("ret_1_lag2"),
            pl.col("vol_5").shift(1).alias("vol_5_lag1"),
        ])
        
        # Clip extreme values
        clip_cols = ['zscore_10', 'zscore_20', 'mom_vol_cross', 'trend_vol_cross']
        for col in clip_cols:
            if col in df.columns:
                df = df.with_columns([pl.col(col).clip(-5, 5).alias(col)])
        
        return df
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Any:
        """Train with feature selection and probability calibration."""
        from sklearn.utils.class_weight import compute_sample_weight
        
        # Binary encoding
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Feature selection
        if X_train_scaled.shape[1] > self.n_features:
            print(f"  Selecting {self.n_features} best features from {X_train_scaled.shape[1]}...")
            self.selector = SelectKBest(mutual_info_classif, k=self.n_features)
            X_train_sel = self.selector.fit_transform(X_train_scaled, y_train_enc)
            X_val_sel = self.selector.transform(X_val_scaled)
            
            # Track selected features
            mask = self.selector.get_support()
            self.selected_features = [self.feature_names[i] for i in range(len(mask)) if mask[i]]
            print(f"  Selected: {self.selected_features[:10]}...")
        else:
            X_train_sel = X_train_scaled
            X_val_sel = X_val_scaled
            self.selected_features = self.feature_names
        
        # Compute class weights
        class_weights = compute_sample_weight('balanced', y_train_enc)
        
        # Train model
        if HAS_LGB:
            model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=30,
                reg_alpha=0.2,
                reg_lambda=0.2,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42,
                verbose=-1,
                importance_type='gain'
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=300,
                max_depth=4,
                learning_rate=0.03,
                min_samples_leaf=30,
                l2_regularization=0.2,
                random_state=42
            )
        
        # Cross-validation score on training data
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_sel, y_train_enc, cv=cv, scoring='roc_auc')
        print(f"  CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Final training with sample weights
        try:
            model.fit(X_train_sel, y_train_enc, sample_weight=class_weights)
        except Exception:
            model.fit(X_train_sel, y_train_enc)
        
        # Calibrate probabilities
        try:
            calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
            calibrated.fit(X_val_sel, y_val_enc)
            print("  Probability calibration successful")
            return calibrated
        except Exception as e:
            print(f"  Calibration failed: {e}")
            return model
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with optional feature selection."""
        X_scaled = self.scaler.transform(X)
        
        if self.selector is not None:
            X_selected = self.selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        # Get probabilities
        probas = self.model.predict_proba(X_selected)
        
        # Apply probability threshold
        # probas[:, 1] is probability of positive class (encoded)
        predictions = np.zeros(len(X), dtype=int)
        
        if len(self.label_encoder.classes_) == 2:
            # Binary: use threshold
            predictions = np.where(probas[:, 1] > self.probability_threshold, 1, 0)
        else:
            # Multi-class: use argmax
            predictions = np.argmax(probas, axis=1)
        
        # Decode
        predictions = self.label_encoder.inverse_transform(predictions)
        return predictions, probas
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for selected features."""
        importance = {}
        
        model = self.model
        if hasattr(model, 'calibrated_classifiers_'):
            # CalibratedClassifierCV wraps the model
            model = model.calibrated_classifiers_[0].estimator
        
        if hasattr(model, 'feature_importances_'):
            for i, imp in enumerate(model.feature_importances_):
                if i < len(self.selected_features):
                    importance[self.selected_features[i]] = float(imp)
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def run_strategy(timeframe: str = "15min", n_features: int = 25) -> StrategyResult:
    strategy = OptimizedBinaryStrategy(
        symbol="BTCUSDT",
        period="2025_11",
        timeframe=timeframe,
        n_features=n_features,
    )
    return strategy.run(verbose=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimized Binary Classification Strategy")
    parser.add_argument("--timeframe", type=str, default="15min", choices=["5min", "15min", "1hr"])
    parser.add_argument("--features", type=int, default=25)
    args = parser.parse_args()
    result = run_strategy(args.timeframe, args.features)
    print(f"\nFinal Return: {result.total_return:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
