"""
Strategy 17: Regime-Filtered Momentum
=====================================

Concept:
- Detect a higher-timeframe regime proxy ("1hr regime") from the same bar series
  using slow trend + volatility percentile.
- Trade momentum only when the regime indicates trending, and align the trade
  direction with the regime side.

Implementation constraints:
- Keeps compatibility with StrategyBase (labels via base create_labels).
- Uses a confidence filter + regime gate at prediction time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_base import StrategyBase


class RegimeFilteredMomentumStrategy(StrategyBase):
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        confidence_threshold: float = 0.52,
        trend_threshold: float = 0.0010,
        **kwargs,
    ):
        super().__init__(symbol=symbol, period=period, timeframe=timeframe, **kwargs)
        self.confidence_threshold = float(confidence_threshold)
        self.trend_threshold = float(trend_threshold)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def get_name(self) -> str:
        return f"Regime_Filtered_Momentum_ct{self.confidence_threshold:.2f}_{self.timeframe}"

    def get_label_params(self) -> Dict[str, Any]:
        # Horizon chosen to reflect "execution" on lower TFs and longer holds on 1hr.
        tf = self.timeframe
        if tf == "1min":
            horizon = 15
        elif tf == "15min":
            horizon = 4
        elif tf == "1hr":
            horizon = 4
        else:
            horizon = max(2, int(round(60 / max(1, self.timeframe_minutes))))

        return {
            "horizon": int(horizon),
            "threshold": 0.0001,
            "mode": "vol",
            "vol_col": "vol_slow",
            "vol_k": 0.75,
            "cost_bps": self.get_trading_cost_bps(),
        }

    def get_feature_columns(self) -> List[str]:
        return [
            "ret_1",
            "mom_exec",
            "mom_regime",
            "trend_strength",
            "vol_fast",
            "vol_slow",
            "vol_percentile",
            "regime_trend",
            "regime_side",
            "volume_ratio",
        ]

    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        tfm = max(1, self.timeframe_minutes)
        bars_15m = max(2, int(round(15 / tfm)))
        bars_1h = max(2, int(round(60 / tfm)))
        bars_slow = max(10, 4 * bars_1h)
        pct_window = max(30, 24 * bars_1h)

        df = df.with_columns([
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("ret_1"),
        ])

        df = df.with_columns([
            (pl.col("close") / pl.col("close").shift(bars_15m) - 1).alias("mom_exec"),
            (pl.col("close") / pl.col("close").shift(bars_1h) - 1).alias("mom_regime"),
        ])

        df = df.with_columns([
            pl.col("ret_1").rolling_std(bars_15m).alias("vol_fast"),
            pl.col("ret_1").rolling_std(bars_slow).alias("vol_slow"),
        ])

        df = df.with_columns([
            pl.col("close").ewm_mean(span=bars_1h, adjust=False).alias("ema_fast"),
            pl.col("close").ewm_mean(span=bars_slow, adjust=False).alias("ema_slow"),
        ])

        df = df.with_columns([
            ((pl.col("ema_fast") - pl.col("ema_slow")) / (pl.col("ema_slow") + 1e-12)).alias("trend_strength"),
        ])

        df = df.with_columns([
            (pl.col("vol_slow").rolling_rank(pct_window) / pct_window).alias("vol_percentile"),
        ])

        df = df.with_columns([
            (
                (pl.col("trend_strength").abs() > float(self.trend_threshold))
                & (pl.col("vol_percentile") > 0.20)
                & (pl.col("vol_percentile") < 0.95)
            ).cast(pl.Int8).alias("regime_trend"),
            pl.when(pl.col("trend_strength") > 0).then(pl.lit(1))
            .when(pl.col("trend_strength") < 0).then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("regime_side"),
            (pl.col("volume") / (pl.col("volume").rolling_mean(20) + 1e-12)).alias("volume_ratio"),
        ])

        return df

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Any:
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)

        if HAS_LGB:
            model = lgb.LGBMClassifier(
                n_estimators=700,
                learning_rate=0.03,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multiclass",
                num_class=len(self.label_encoder.classes_),
                random_state=42,
                n_jobs=-1,
            )
            model.fit(
                X_train_s,
                y_train_enc,
                sample_weight=sample_weights,
                eval_set=[(X_val_s, y_val_enc)],
                eval_metric="multi_logloss",
            )
            return model

        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            random_state=42,
        )
        model.fit(X_train_s, y_train_enc, sample_weight=sample_weights)
        return model

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_s = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(X_s)
        else:
            n = X.shape[0]
            k = len(self.label_encoder.classes_)
            prob = np.full((n, k), 1.0 / max(1, k), dtype=float)

        pred_enc = np.argmax(prob, axis=1)
        pred = self.label_encoder.inverse_transform(pred_enc)

        max_prob = np.max(prob, axis=1)
        conf_mask = max_prob >= float(self.confidence_threshold)

        idx_regime_trend = None
        idx_regime_side = None
        for i, name in enumerate(self.feature_names):
            if name == "regime_trend":
                idx_regime_trend = i
            elif name == "regime_side":
                idx_regime_side = i

        if idx_regime_trend is None or idx_regime_side is None:
            pred = np.where(conf_mask, pred, 0)
            return pred.astype(int), prob

        regime_trend = (X[:, idx_regime_trend] > 0.5)
        regime_side = np.sign(X[:, idx_regime_side]).astype(int)
        pred_sign = np.sign(pred).astype(int)

        keep = conf_mask & regime_trend & (regime_side != 0) & (pred_sign == regime_side)
        pred = np.where(keep, pred, 0)

        return pred.astype(int), prob
