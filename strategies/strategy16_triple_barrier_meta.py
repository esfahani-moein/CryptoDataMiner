"""
Strategy 16: Triple-Barrier + Meta-Label (AFML-style)
=====================================================

Goal:
- Use triple-barrier outcomes as a more robust notion of "tradable" events.
- Use an initial side signal (momentum/trend) and meta-label it (trade or hold).
- Train a model to predict the meta-labeled side (-1/0/+1), then apply a
  confidence filter and enforce consistency with the initial side.

Notes:
- Triple-barrier is computed with the iterative implementation for correctness
  (still feasible on aggregated OHLCV bar counts).
- Evaluation remains compatible with StrategyBase: we align the label horizon
  with the triple-barrier max holding period and use `fwd_ret_{horizon}` for PnL.
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


class TripleBarrierMetaStrategy(StrategyBase):
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        confidence_threshold: float = 0.55,
        **kwargs
    ):
        super().__init__(symbol=symbol, period=period, timeframe=timeframe, **kwargs)
        self.confidence_threshold = float(confidence_threshold)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # Derived per-timeframe settings
        self._tb_horizon = self._choose_tb_horizon()
        self._tb_profit, self._tb_stop = self._choose_tb_barriers()

    def _choose_tb_horizon(self) -> int:
        # Hold horizon is the vertical barrier; keep reasonable for speed.
        # Target ~2 hours, with a minimum to keep signal meaningful.
        hold_minutes = 120
        horizon = max(4, int(round(hold_minutes / max(1, self.timeframe_minutes))))
        return horizon

    def _choose_tb_barriers(self) -> Tuple[float, float]:
        # Conservative defaults by bar size.
        tfm = self.timeframe_minutes
        if tfm <= 1:
            return 0.0035, 0.0030
        if tfm <= 15:
            return 0.0100, 0.0080
        return 0.0200, 0.0150

    def get_name(self) -> str:
        return (
            f"TB_Meta_h{self._tb_horizon}_pt{self._tb_profit:.4f}_sl{self._tb_stop:.4f}_"
            f"ct{self.confidence_threshold:.2f}_{self.timeframe}"
        )

    def get_label_params(self) -> Dict[str, Any]:
        # Align the evaluation horizon with the triple-barrier vertical barrier.
        return {
            "horizon": int(self._tb_horizon),
            "threshold": 0.0001,
            "mode": "fixed",
            "vol_col": "vol_slow",
            "vol_k": 1.0,
            "cost_bps": self.get_trading_cost_bps(),
        }

    def get_feature_columns(self) -> List[str]:
        return [
            "ret_1",
            "mom_fast",
            "mom_slow",
            "trend_strength",
            "vol_fast",
            "vol_slow",
            "volume_ratio",
            "base_side",
        ]

    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        tfm = max(1, self.timeframe_minutes)
        bars_15m = max(2, int(round(15 / tfm)))
        bars_1h = max(2, int(round(60 / tfm)))
        bars_slow = max(10, 4 * bars_1h)

        df = df.with_columns([
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("ret_1"),
        ])

        df = df.with_columns([
            (pl.col("close") / pl.col("close").shift(bars_15m) - 1).alias("mom_fast"),
            (pl.col("close") / pl.col("close").shift(bars_1h) - 1).alias("mom_slow"),
        ])

        df = df.with_columns([
            pl.col("ret_1").rolling_std(bars_15m).alias("vol_fast"),
            pl.col("ret_1").rolling_std(bars_slow).alias("vol_slow"),
        ])

        # Trend proxy: EMA cross
        df = df.with_columns([
            pl.col("close").ewm_mean(span=bars_1h, adjust=False).alias("ema_fast"),
            pl.col("close").ewm_mean(span=bars_slow, adjust=False).alias("ema_slow"),
        ])

        df = df.with_columns([
            ((pl.col("ema_fast") - pl.col("ema_slow")) / (pl.col("ema_slow") + 1e-12)).alias("trend_strength"),
            (pl.col("volume") / (pl.col("volume").rolling_mean(20) + 1e-12)).alias("volume_ratio"),
        ])

        return df

    def create_labels(
        self,
        df: pl.DataFrame,
        horizon: int = None,
        threshold: float = 0.001,
        mode: str = "fixed",
        vol_col: str = "vol_10",
        vol_k: float = 1.0,
        cost_bps: float = 0.0,
    ) -> pl.DataFrame:
        from quant_features.labeling import add_forward_returns, add_triple_barrier_labels, add_sample_weights

        if horizon is None:
            horizon = int(self._tb_horizon)

        # Ensure forward returns exist for evaluation.
        df = add_forward_returns(df, periods=[int(horizon)])

        # Initial side (the "primary" model/heuristic).
        # Use a deadband so we don't trade noise.
        deadband = (
            pl.when(pl.col("vol_slow").is_not_null())
            .then(pl.col("vol_slow") * 0.5)
            .otherwise(pl.lit(0.0))
        )

        df = df.with_columns([
            pl.when(pl.col("mom_slow") > deadband).then(pl.lit(1))
            .when(pl.col("mom_slow") < -deadband).then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("base_side")
        ])

        # Triple barrier outcome for a LONG (tb_label: +1 good long, -1 bad long).
        df = add_triple_barrier_labels(
            df,
            max_holding_period=int(horizon),
            profit_taking=float(self._tb_profit),
            stop_loss=float(self._tb_stop),
            price_col="close",
            use_vertical_barrier=True,
        )

        # Meta-label: is the base side correct according to triple-barrier?
        df = df.with_columns([
            (
                (pl.col("base_side") != 0)
                & (pl.col("tb_label") != 0)
                & (pl.col("base_side") * pl.col("tb_label") > 0)
            ).cast(pl.Int8).alias("meta_label")
        ])

        # Meta-labeled side target: -1/0/+1.
        df = df.with_columns([
            (pl.col("base_side") * pl.col("meta_label")).cast(pl.Int8).alias("label")
        ])

        # Sample weights from triple-barrier return magnitude.
        df = add_sample_weights(df, return_col="tb_return")

        return df

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Any:
        # Scale
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        # Encode labels (-1,0,1) -> (0,1,2)
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)

        if HAS_LGB:
            model = lgb.LGBMClassifier(
                n_estimators=600,
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
            # Fallback: uniform probabilities
            n = X.shape[0]
            k = len(self.label_encoder.classes_)
            prob = np.full((n, k), 1.0 / max(1, k), dtype=float)

        pred_enc = np.argmax(prob, axis=1)
        pred = self.label_encoder.inverse_transform(pred_enc)

        # Confidence + consistency with base side.
        max_prob = np.max(prob, axis=1)
        conf_mask = max_prob >= float(self.confidence_threshold)

        base_side_idx = None
        for i, name in enumerate(self.feature_names):
            if name == "base_side":
                base_side_idx = i
                break

        if base_side_idx is None:
            # Should not happen; if it does, just apply confidence.
            pred = np.where(conf_mask, pred, 0)
            return pred.astype(int), prob

        base_side = np.sign(X[:, base_side_idx]).astype(int)
        pred_sign = np.sign(pred).astype(int)

        keep = conf_mask & (base_side != 0) & (pred_sign == base_side)
        pred = np.where(keep, pred, 0)

        return pred.astype(int), prob
