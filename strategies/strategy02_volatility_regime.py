"""
Strategy 02: Volatility Regime Detection
=========================================

Detects market volatility regimes and adapts trading strategy accordingly.
Uses Hidden Markov Model (HMM) or clustering for regime detection.

Key Concepts:
- Identify low/medium/high volatility regimes
- Use different feature sets per regime
- Adjust position sizing based on regime
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import xgboost as xgb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_base import StrategyBase


class VolatilityRegimeStrategy(StrategyBase):
    """
    Volatility regime-based strategy.
    
    Approach:
    1. Detect volatility regimes using clustering
    2. Train separate models for each regime OR use regime as feature
    3. Adapt predictions based on current regime
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        n_regimes: int = 3,  # Low, Medium, High volatility
        regime_method: str = "gmm",  # "gmm" or "kmeans"
        **kwargs
    ):
        super().__init__(symbol, period, timeframe, **kwargs)
        self.n_regimes = n_regimes
        self.regime_method = regime_method
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.regime_model = None
        self.regime_scalers = {}
        
    def get_name(self) -> str:
        return f"Volatility_Regime_{self.regime_method}_{self.timeframe}"
    
    def get_feature_columns(self) -> List[str]:
        """Volatility-focused features plus regime indicator."""
        features = []
        
        # Volatility measures at multiple windows
        for w in [5, 10, 20, 50, 100]:
            features.extend([
                f"vol_std_{w}",
                f"vol_parkinson_{w}",
                f"vol_garman_klass_{w}",
                f"vol_yang_zhang_{w}",
            ])
        
        # Volatility ratios and changes
        features.extend([
            "vol_ratio_5_20",
            "vol_ratio_20_50",
            "vol_change_5",
            "vol_change_20",
            "vol_percentile_20",
            "vol_zscore_20",
        ])
        
        # Range-based features
        for w in [5, 10, 20]:
            features.extend([
                f"atr_{w}",
                f"atr_pct_{w}",
                f"range_pct_{w}",
            ])
        
        # Regime indicator (added during create_features)
        features.append("volatility_regime")
        features.append("regime_duration")
        
        # Price features that work differently in each regime
        for w in [5, 10, 20]:
            features.extend([
                f"ret_{w}",
                f"ret_vol_adj_{w}",
                f"momentum_{w}",
            ])
        
        # Mean reversion vs trend features
        features.extend([
            "mean_reversion_score",
            "trend_strength_score",
            "regime_optimal_strategy",  # 1 for trend, -1 for mean-reversion
        ])
        
        # Bollinger Band features
        features.extend([
            "bb_position",
            "bb_width",
            "bb_squeeze",
        ])
        
        # Keltner Channel features
        features.extend([
            "kc_position",
            "kc_width",
            "squeeze_indicator",
        ])
        
        return features
    
    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create volatility and regime-based features."""
        
        # === Standard Deviation Volatility ===
        for w in [5, 10, 20, 50, 100]:
            df = df.with_columns([
                pl.col("close").pct_change().rolling_std(window_size=w).alias(f"vol_std_{w}")
            ])
        
        # === Parkinson Volatility ===
        log_hl = (pl.col("high") / pl.col("low")).log()
        for w in [5, 10, 20, 50, 100]:
            df = df.with_columns([
                ((log_hl ** 2).rolling_mean(window_size=w) / (4 * np.log(2))).sqrt().alias(f"vol_parkinson_{w}")
            ])
        
        # === Garman-Klass Volatility ===
        log_hl_sq = (pl.col("high") / pl.col("low")).log() ** 2
        log_co_sq = (pl.col("close") / pl.col("open")).log() ** 2
        
        for w in [5, 10, 20, 50, 100]:
            df = df.with_columns([
                (0.5 * log_hl_sq.rolling_mean(window_size=w) - 
                 (2 * np.log(2) - 1) * log_co_sq.rolling_mean(window_size=w)).sqrt().alias(f"vol_garman_klass_{w}")
            ])
        
        # === Yang-Zhang Volatility (simplified) ===
        log_oc = (pl.col("open") / pl.col("close").shift(1)).log()
        log_co = (pl.col("close") / pl.col("open")).log()
        
        for w in [5, 10, 20, 50, 100]:
            overnight_var = log_oc.rolling_var(window_size=w)
            close_var = log_co.rolling_var(window_size=w)
            df = df.with_columns([
                (overnight_var + 0.5 * close_var).sqrt().alias(f"vol_yang_zhang_{w}")
            ])
        
        # === Volatility Ratios ===
        df = df.with_columns([
            (pl.col("vol_std_5") / (pl.col("vol_std_20") + 1e-10)).alias("vol_ratio_5_20"),
            (pl.col("vol_std_20") / (pl.col("vol_std_50") + 1e-10)).alias("vol_ratio_20_50"),
            (pl.col("vol_std_5") / pl.col("vol_std_5").shift(5) - 1).alias("vol_change_5"),
            (pl.col("vol_std_20") / pl.col("vol_std_20").shift(20) - 1).alias("vol_change_20"),
        ])
        
        # === Volatility Percentile ===
        # Use rolling rank as percentile proxy
        df = df.with_columns([
            (pl.col("vol_std_20").rolling_mean(window_size=100)).alias("_vol_ma_100"),
            (pl.col("vol_std_20").rolling_std(window_size=100)).alias("_vol_std_100"),
        ])
        df = df.with_columns([
            ((pl.col("vol_std_20") - pl.col("_vol_ma_100")) / 
             (pl.col("_vol_std_100") + 1e-10)).alias("vol_zscore_20"),
        ])
        
        # Percentile approximation using zscore
        df = df.with_columns([
            pl.col("vol_zscore_20").tanh().alias("vol_percentile_20")
        ])
        
        # === ATR ===
        df = df.with_columns([
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs()
            ).alias("_tr")
        ])
        
        for w in [5, 10, 20]:
            df = df.with_columns([
                pl.col("_tr").ewm_mean(span=w, adjust=False).alias(f"atr_{w}"),
            ])
            df = df.with_columns([
                (pl.col(f"atr_{w}") / pl.col("close") * 100).alias(f"atr_pct_{w}"),
                ((pl.col("high").rolling_max(window_size=w) - 
                  pl.col("low").rolling_min(window_size=w)) / pl.col("close") * 100).alias(f"range_pct_{w}"),
            ])
        
        # === Returns and Volatility-Adjusted Returns ===
        for w in [5, 10, 20]:
            df = df.with_columns([
                (pl.col("close") / pl.col("close").shift(w) - 1).alias(f"ret_{w}"),
            ])
            df = df.with_columns([
                (pl.col(f"ret_{w}") / (pl.col(f"vol_std_{w}") + 1e-10)).alias(f"ret_vol_adj_{w}"),
                pl.col(f"ret_{w}").alias(f"momentum_{w}"),
            ])
        
        # === Detect Volatility Regime ===
        df = self._add_volatility_regime(df)
        
        # === Mean Reversion vs Trend Score ===
        # High volatility often means mean reversion, low volatility often means trend
        df = df.with_columns([
            # Mean reversion score: price distance from MA
            ((pl.col("close") - pl.col("close").rolling_mean(window_size=20)) / 
             (pl.col("close").rolling_std(window_size=20) + 1e-10)).alias("mean_reversion_score"),
            
            # Trend strength: directional movement ratio
            (pl.col("close").pct_change().rolling_sum(window_size=20).abs() /
             (pl.col("close").pct_change().abs().rolling_sum(window_size=20) + 1e-10)).alias("trend_strength_score"),
        ])
        
        # Regime optimal strategy: trend-following in low vol, mean-reversion in high vol
        df = df.with_columns([
            pl.when(pl.col("volatility_regime") == 0)
            .then(pl.lit(1))  # Low vol -> trend following
            .when(pl.col("volatility_regime") == 2)
            .then(pl.lit(-1))  # High vol -> mean reversion
            .otherwise(pl.lit(0)).alias("regime_optimal_strategy")
        ])
        
        # === Bollinger Bands ===
        bb_ma = pl.col("close").rolling_mean(window_size=20)
        bb_std = pl.col("close").rolling_std(window_size=20)
        
        df = df.with_columns([
            bb_ma.alias("_bb_ma"),
            (bb_ma + 2 * bb_std).alias("_bb_upper"),
            (bb_ma - 2 * bb_std).alias("_bb_lower"),
        ])
        
        df = df.with_columns([
            ((pl.col("close") - pl.col("_bb_lower")) / 
             (pl.col("_bb_upper") - pl.col("_bb_lower") + 1e-10)).alias("bb_position"),
            ((pl.col("_bb_upper") - pl.col("_bb_lower")) / pl.col("_bb_ma")).alias("bb_width"),
        ])
        
        # Squeeze indicator
        bb_width_ma = pl.col("bb_width").rolling_mean(window_size=50)
        df = df.with_columns([
            pl.when(pl.col("bb_width") < bb_width_ma * 0.8)
            .then(pl.lit(1))
            .otherwise(pl.lit(0)).alias("bb_squeeze")
        ])
        
        # === Keltner Channel ===
        kc_ma = pl.col("close").ewm_mean(span=20, adjust=False)
        
        df = df.with_columns([
            kc_ma.alias("_kc_ma"),
            (kc_ma + 2 * pl.col("atr_20")).alias("_kc_upper"),
            (kc_ma - 2 * pl.col("atr_20")).alias("_kc_lower"),
        ])
        
        df = df.with_columns([
            ((pl.col("close") - pl.col("_kc_lower")) / 
             (pl.col("_kc_upper") - pl.col("_kc_lower") + 1e-10)).alias("kc_position"),
            ((pl.col("_kc_upper") - pl.col("_kc_lower")) / pl.col("_kc_ma")).alias("kc_width"),
        ])
        
        # Squeeze: BB inside KC
        df = df.with_columns([
            pl.when(
                (pl.col("_bb_upper") < pl.col("_kc_upper")) &
                (pl.col("_bb_lower") > pl.col("_kc_lower"))
            ).then(pl.lit(1)).otherwise(pl.lit(0)).alias("squeeze_indicator")
        ])
        
        return df
    
    def _add_volatility_regime(self, df: pl.DataFrame) -> pl.DataFrame:
        """Detect volatility regime using clustering."""
        # Features for regime detection
        vol_features = ["vol_std_20", "vol_parkinson_20", "atr_pct_20"]
        available = [f for f in vol_features if f in df.columns]
        
        if not available:
            # Fallback: create simple vol feature
            df = df.with_columns([
                pl.col("close").pct_change().rolling_std(window_size=20).alias("vol_std_20")
            ])
            available = ["vol_std_20"]
        
        # Extract volatility data
        vol_data = df.select(available).to_numpy()
        vol_data = np.nan_to_num(vol_data, nan=0.0)
        
        # Fit regime model
        vol_scaler = StandardScaler()
        vol_scaled = vol_scaler.fit_transform(vol_data)
        
        if self.regime_method == "gmm":
            self.regime_model = GaussianMixture(
                n_components=self.n_regimes,
                random_state=self.random_state,
                covariance_type='full'
            )
        else:
            self.regime_model = KMeans(
                n_clusters=self.n_regimes,
                random_state=self.random_state,
                n_init=10
            )
        
        regimes = self.regime_model.fit_predict(vol_scaled)
        
        # Order regimes by mean volatility (0=low, 1=med, 2=high)
        regime_vols = []
        for r in range(self.n_regimes):
            mask = regimes == r
            if mask.sum() > 0:
                regime_vols.append((r, vol_data[mask, 0].mean()))
            else:
                regime_vols.append((r, 0))
        
        regime_order = [r for r, _ in sorted(regime_vols, key=lambda x: x[1])]
        regime_map = {old: new for new, old in enumerate(regime_order)}
        regimes_ordered = np.array([regime_map[r] for r in regimes])
        
        # Add regime to dataframe
        df = df.with_columns([
            pl.Series("volatility_regime", regimes_ordered.astype(np.int32))
        ])
        
        # Regime duration: how long in current regime
        regime_change = (pl.col("volatility_regime") != pl.col("volatility_regime").shift(1)).cast(pl.Int32)
        df = df.with_columns([
            regime_change.alias("_regime_change")
        ])
        
        # Cumsum groups for regime duration
        df = df.with_columns([
            pl.col("_regime_change").cum_sum().alias("_regime_group")
        ])
        
        df = df.with_columns([
            pl.lit(1).cum_sum().over("_regime_group").alias("regime_duration")
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
        """Train XGBoost with regime awareness."""
        
        # Encode labels
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        n_classes = len(self.label_encoder.classes_)
        
        # XGBoost with regime as feature (already included)
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.5,
            reg_lambda=2.0,
            gamma=0.1,
            random_state=self.random_state,
            objective='multi:softprob',
            num_class=n_classes,
            eval_metric='mlogloss',
            early_stopping_rounds=30,
            verbosity=0,
        )
        
        model.fit(
            X_train_scaled, y_train_enc,
            sample_weight=sample_weights,
            eval_set=[(X_val_scaled, y_val_enc)],
            verbose=False,
        )
        
        self.model = model
        return model
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        
        y_prob = self.model.predict_proba(X_scaled)
        y_pred_enc = np.argmax(y_prob, axis=1)
        y_pred = self.label_encoder.inverse_transform(y_pred_enc)
        
        return y_pred, y_prob


def run_strategy(timeframe: str = "15min", regime_method: str = "gmm"):
    """Run the volatility regime strategy."""
    strategy = VolatilityRegimeStrategy(
        timeframe=timeframe,
        regime_method=regime_method,
    )
    return strategy.run(verbose=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Volatility Regime Strategy")
    parser.add_argument("--timeframe", type=str, default="15min",
                        choices=["5min", "15min", "1hr"])
    parser.add_argument("--method", type=str, default="gmm",
                        choices=["gmm", "kmeans"])
    
    args = parser.parse_args()
    
    result = run_strategy(args.timeframe, args.method)
    print(f"\nFinal Sharpe Ratio: {result.sharpe_ratio:.4f}")
