"""
Strategy 10: Regime Adaptive - Multiple Models per Market Regime
================================================================================
Trains separate models for each detected market regime (trending, mean-reverting,
high volatility, low volatility) and switches between them dynamically.
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from strategies.strategy_base import StrategyBase, StrategyResult


class RegimeAdaptiveStrategy(StrategyBase):
    """
    Regime-adaptive strategy with specialized models per market state.
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        n_regimes: int = 4,
        **kwargs
    ):
        super().__init__(symbol, period, timeframe, **kwargs)
        self.n_regimes = n_regimes
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.regime_detector = None
        self.regime_scalers = {}
        self.regime_models = {}
        self.global_model = None
    
    def get_name(self) -> str:
        return f"Regime_Adaptive_{self.n_regimes}reg_{self.timeframe}"
    
    def get_feature_columns(self) -> List[str]:
        return [
            # Momentum
            'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20',
            'momentum_5', 'momentum_10',
            
            # Volatility
            'vol_5', 'vol_10', 'vol_20',
            'vol_ratio', 'vol_percentile',
            
            # Trend
            'trend_strength', 'trend_direction',
            'sma_deviation', 'ema_deviation',
            
            # Mean reversion
            'mean_reversion_score', 'bb_position',
            'zscore_20', 'zscore_50',
            
            # Regime features
            'regime', 'regime_confidence',
            'regime_duration',
            'regime_0_prob', 'regime_1_prob', 'regime_2_prob', 'regime_3_prob',
            
            # Volume
            'volume_ratio', 'volume_ma_deviation',
            
            # Cross features
            'vol_trend_cross', 'mom_regime_cross',
        ]
    
    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create features with regime detection."""
        
        # === MOMENTUM FEATURES ===
        for period in [1, 3, 5, 10, 20]:
            df = df.with_columns([
                pl.col("close").pct_change(period).alias(f"ret_{period}")
            ])
        
        for period in [5, 10]:
            df = df.with_columns([
                (pl.col("close") / pl.col("close").shift(period) - 1).alias(f"momentum_{period}")
            ])
        
        # === VOLATILITY FEATURES ===
        for period in [5, 10, 20]:
            df = df.with_columns([
                pl.col("ret_1").rolling_std(period).alias(f"vol_{period}")
            ])
        
        df = df.with_columns([
            (pl.col("vol_5") / (pl.col("vol_20") + 1e-10)).alias("vol_ratio")
        ])
        
        df = df.with_columns([
            (pl.col("vol_10").rolling_quantile(0.5, window_size=50)).alias("_vol_median")
        ])
        df = df.with_columns([
            pl.when(pl.col("vol_10") > pl.col("_vol_median")).then(1).otherwise(0).alias("vol_percentile")
        ])
        
        # === TREND FEATURES ===
        df = df.with_columns([
            pl.col("close").rolling_mean(10).alias("sma_10"),
            pl.col("close").rolling_mean(20).alias("sma_20"),
            pl.col("close").rolling_mean(50).alias("sma_50"),
            pl.col("close").ewm_mean(span=10, adjust=False).alias("ema_10"),
            pl.col("close").ewm_mean(span=50, adjust=False).alias("ema_50"),
        ])
        
        df = df.with_columns([
            ((pl.col("sma_10") - pl.col("sma_50")) / (pl.col("sma_50") + 1e-10)).alias("trend_strength"),
            pl.when(pl.col("sma_10") > pl.col("sma_50")).then(1).otherwise(-1).alias("trend_direction"),
            ((pl.col("close") - pl.col("sma_20")) / (pl.col("sma_20") + 1e-10) * 100).alias("sma_deviation"),
            ((pl.col("close") - pl.col("ema_10")) / (pl.col("ema_10") + 1e-10) * 100).alias("ema_deviation"),
        ])
        
        # === MEAN REVERSION FEATURES ===
        for period in [20, 50]:
            df = df.with_columns([
                pl.col("close").rolling_mean(period).alias(f"_ma_{period}"),
                pl.col("close").rolling_std(period).alias(f"_std_{period}"),
            ])
            df = df.with_columns([
                ((pl.col("close") - pl.col(f"_ma_{period}")) / 
                 (pl.col(f"_std_{period}") + 1e-10)).alias(f"zscore_{period}")
            ])
        
        df = df.with_columns([
            pl.col("zscore_20").abs().alias("mean_reversion_score")
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
        
        # === VOLUME FEATURES ===
        df = df.with_columns([
            (pl.col("volume") / (pl.col("volume").rolling_mean(20) + 1e-10)).alias("volume_ratio"),
            ((pl.col("volume") - pl.col("volume").rolling_mean(20)) / 
             (pl.col("volume").rolling_std(20) + 1e-10)).alias("volume_ma_deviation"),
        ])
        
        # === REGIME DETECTION ===
        df_pd = df.select(["vol_10", "trend_strength", "momentum_5"]).to_pandas()
        df_pd = df_pd.fillna(0)
        regime_features = df_pd[["vol_10", "trend_strength", "momentum_5"]].values
        
        valid_mask = ~np.isnan(regime_features).any(axis=1) & ~np.isinf(regime_features).any(axis=1)
        
        if valid_mask.sum() > self.n_regimes * 10:
            self.regime_detector = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42,
                n_init=3
            )
            
            scaler_regime = StandardScaler()
            regime_features_scaled = scaler_regime.fit_transform(regime_features[valid_mask])
            self.regime_detector.fit(regime_features_scaled)
            
            all_scaled = scaler_regime.transform(np.nan_to_num(regime_features, 0))
            regimes = self.regime_detector.predict(all_scaled)
            regime_probs = self.regime_detector.predict_proba(all_scaled)
            
            df = df.with_columns([pl.Series("regime", regimes)])
            
            for i in range(self.n_regimes):
                df = df.with_columns([pl.Series(f"regime_{i}_prob", regime_probs[:, i])])
            
            df = df.with_columns([
                pl.Series("regime_confidence", np.max(regime_probs, axis=1))
            ])
        else:
            df = df.with_columns([
                pl.when(pl.col("vol_10") > pl.col("vol_10").rolling_quantile(0.75, window_size=50))
                .then(3)
                .when(pl.col("vol_10") < pl.col("vol_10").rolling_quantile(0.25, window_size=50))
                .then(0)
                .when(pl.col("trend_direction") > 0)
                .then(1)
                .otherwise(2)
                .alias("regime"),
                pl.lit(0.5).alias("regime_confidence"),
            ])
            for i in range(self.n_regimes):
                df = df.with_columns([pl.lit(0.25).alias(f"regime_{i}_prob")])
        
        # Regime duration
        df = df.with_columns([
            (pl.col("regime") == pl.col("regime").shift(1)).cast(pl.Int32).alias("_same_regime")
        ])
        df = df.with_columns([
            pl.col("_same_regime").cum_sum().over(
                (pl.col("regime") != pl.col("regime").shift(1)).cum_sum()
            ).alias("regime_duration")
        ])
        
        # === CROSS FEATURES ===
        df = df.with_columns([
            (pl.col("vol_10") * pl.col("trend_strength")).alias("vol_trend_cross"),
            (pl.col("momentum_5") * pl.col("regime")).alias("mom_regime_cross"),
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
        """Train regime-specific models."""
        
        y_train_enc = self.label_encoder.fit_transform(y_train)
        
        # Get regime column index
        regime_idx = None
        for i, col in enumerate(self.feature_names):
            if col == "regime":
                regime_idx = i
                break
        
        if regime_idx is None:
            return self._train_single_model(X_train, y_train_enc, sample_weights)
        
        train_regimes = X_train[:, regime_idx].astype(int)
        
        print(f"  Training {self.n_regimes} regime-specific models...")
        
        for regime in range(self.n_regimes):
            regime_mask = train_regimes == regime
            
            if regime_mask.sum() < 50:
                print(f"    Regime {regime}: insufficient data ({regime_mask.sum()} samples)")
                continue
            
            X_regime = X_train[regime_mask]
            y_regime = y_train_enc[regime_mask]
            weights_regime = sample_weights[regime_mask] if sample_weights is not None else None
            
            self.regime_scalers[regime] = StandardScaler()
            X_regime_scaled = self.regime_scalers[regime].fit_transform(X_regime)
            
            if HAS_LGB:
                model = lgb.LGBMClassifier(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                    verbose=-1
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    random_state=42
                )
            
            if weights_regime is not None:
                model.fit(X_regime_scaled, y_regime, sample_weight=weights_regime)
            else:
                model.fit(X_regime_scaled, y_regime)
            
            self.regime_models[regime] = model
            print(f"    Regime {regime}: trained on {regime_mask.sum()} samples")
        
        # Global fallback
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        if HAS_LGB:
            self.global_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                verbose=-1
            )
        else:
            self.global_model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                random_state=42
            )
        
        if sample_weights is not None:
            self.global_model.fit(X_train_scaled, y_train_enc, sample_weight=sample_weights)
        else:
            self.global_model.fit(X_train_scaled, y_train_enc)
        
        return {"regime_models": self.regime_models, "global_model": self.global_model}
    
    def _train_single_model(self, X_train, y_train, sample_weights):
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if HAS_LGB:
            model = lgb.LGBMClassifier(n_estimators=200, max_depth=5, random_state=42, verbose=-1)
        else:
            model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
        
        if sample_weights is not None:
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train_scaled, y_train)
        
        self.global_model = model
        return model
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using regime-specific models."""
        
        regime_idx = None
        for i, col in enumerate(self.feature_names):
            if col == "regime":
                regime_idx = i
                break
        
        if regime_idx is not None and isinstance(self.model, dict) and "regime_models" in self.model:
            predictions = np.zeros(len(X))
            probas = np.zeros((len(X), len(self.label_encoder.classes_)))
            
            regimes = X[:, regime_idx].astype(int)
            
            for regime in range(self.n_regimes):
                regime_mask = regimes == regime
                
                if regime_mask.sum() == 0:
                    continue
                
                X_regime = X[regime_mask]
                
                if regime in self.regime_models and regime in self.regime_scalers:
                    X_scaled = self.regime_scalers[regime].transform(X_regime)
                    predictions[regime_mask] = self.regime_models[regime].predict(X_scaled)
                    probas[regime_mask] = self.regime_models[regime].predict_proba(X_scaled)
                else:
                    X_scaled = self.scaler.transform(X_regime)
                    predictions[regime_mask] = self.global_model.predict(X_scaled)
                    probas[regime_mask] = self.global_model.predict_proba(X_scaled)
            
            predictions = self.label_encoder.inverse_transform(predictions.astype(int))
            return predictions, probas
        
        # Fallback
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(X_scaled)
            probas = self.model.predict_proba(X_scaled)
        else:
            predictions = self.global_model.predict(X_scaled)
            probas = self.global_model.predict_proba(X_scaled)
        
        predictions = self.label_encoder.inverse_transform(predictions)
        return predictions, probas
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Aggregate feature importance."""
        importance = {}
        
        if isinstance(self.model, dict) and "regime_models" in self.model:
            for regime, regime_model in self.model["regime_models"].items():
                if hasattr(regime_model, 'feature_importances_'):
                    for i, feat in enumerate(self.feature_names[:len(regime_model.feature_importances_)]):
                        if feat not in importance:
                            importance[feat] = 0
                        importance[feat] += float(regime_model.feature_importances_[i])
        
        if self.global_model is not None and hasattr(self.global_model, 'feature_importances_'):
            for i, feat in enumerate(self.feature_names[:len(self.global_model.feature_importances_)]):
                if feat not in importance:
                    importance[feat] = 0
                importance[feat] += float(self.global_model.feature_importances_[i])
        
        return importance


def run_strategy(timeframe: str = "15min", n_regimes: int = 4) -> StrategyResult:
    strategy = RegimeAdaptiveStrategy(
        symbol="BTCUSDT",
        period="2025_11",
        timeframe=timeframe,
        n_regimes=n_regimes,
    )
    return strategy.run(verbose=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Regime Adaptive Strategy")
    parser.add_argument("--timeframe", type=str, default="15min", choices=["5min", "15min", "1hr"])
    parser.add_argument("--regimes", type=int, default=4, choices=[2, 3, 4])
    args = parser.parse_args()
    result = run_strategy(args.timeframe, args.regimes)
    print(f"\nFinal Sharpe Ratio: {result.sharpe_ratio:.4f}")
