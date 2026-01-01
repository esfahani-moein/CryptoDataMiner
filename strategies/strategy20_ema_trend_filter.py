"""\
Strategy 20: EMA Trend Filter + Momentum (Rule-Based)
=====================================================

Non-ML trend strategy designed to be relatively low-turnover:
- Direction from EMA fast/slow.
- Only trade when trend_strength exceeds a small threshold.
- Momentum confirmation on a shorter window.

This is intentionally simple and robust across timeframes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_base import StrategyBase


class EmaTrendFilterStrategy(StrategyBase):
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        trend_threshold: float = 0.0010,
        **kwargs,
    ):
        super().__init__(symbol=symbol, period=period, timeframe=timeframe, **kwargs)
        self.trend_threshold = float(trend_threshold)

        self._fast = self._choose_fast_span()
        self._slow = self._choose_slow_span()
        self._mom = self._choose_mom_window()

    def _choose_fast_span(self) -> int:
        # ~1 hour
        return max(8, int(round(60 / max(1, self.timeframe_minutes))))

    def _choose_slow_span(self) -> int:
        # ~6 hours
        return max(30, int(round(360 / max(1, self.timeframe_minutes))))

    def _choose_mom_window(self) -> int:
        # ~15 minutes
        return max(3, int(round(15 / max(1, self.timeframe_minutes))))

    def get_name(self) -> str:
        return f"EMA_Trend_Filter_{self.timeframe}"

    def get_label_params(self) -> Dict[str, Any]:
        if self.timeframe == "1min":
            horizon = 30
        elif self.timeframe == "15min":
            horizon = 4
        elif self.timeframe == "1hr":
            horizon = 2
        else:
            horizon = max(2, int(round(60 / max(1, self.timeframe_minutes))))

        return {
            "horizon": int(horizon),
            "threshold": 0.0001,
            "mode": "vol",
            "vol_col": "vol_slow",
            "vol_k": 0.70,
            "cost_bps": self.get_trading_cost_bps(),
        }

    def get_feature_columns(self) -> List[str]:
        return [
            "trend_strength",
            "trend_dir",
            "mom",
            "vol_slow",
            "vol_percentile",
            "volume_ratio",
        ]

    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        fast = int(self._fast)
        slow = int(self._slow)
        mom_w = int(self._mom)
        pct_window = max(30, 24 * max(2, int(round(60 / max(1, self.timeframe_minutes)))))

        df = df.with_columns([
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("ret_1"),
            pl.col("close").ewm_mean(span=fast, adjust=False).alias("ema_fast"),
            pl.col("close").ewm_mean(span=slow, adjust=False).alias("ema_slow"),
        ])

        df = df.with_columns([
            ((pl.col("ema_fast") - pl.col("ema_slow")) / (pl.col("ema_slow") + 1e-12)).alias("trend_strength"),
            pl.when(pl.col("ema_fast") > pl.col("ema_slow")).then(1).otherwise(-1).cast(pl.Int8).alias("trend_dir"),
            (pl.col("close") / pl.col("close").shift(mom_w) - 1).alias("mom"),
        ])

        df = df.with_columns([
            pl.col("ret_1").rolling_std(max(20, slow)).alias("vol_slow"),
        ])
        df = df.with_columns([
            (pl.col("vol_slow").rolling_rank(pct_window) / pct_window).alias("vol_percentile"),
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
        return {
            "fast": self._fast,
            "slow": self._slow,
            "mom": self._mom,
            "trend_threshold": self.trend_threshold,
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = {name: i for i, name in enumerate(self.feature_names)}

        strength = X[:, idx["trend_strength"]]
        direction = np.sign(X[:, idx["trend_dir"]]).astype(int)
        mom = X[:, idx["mom"]]
        vol_pct = X[:, idx["vol_percentile"]]

        # Avoid ultra-dead and ultra-chaotic periods
        gate = (vol_pct >= 0.15) & (vol_pct <= 0.95)

        # Hysteresis by requiring strength
        strong = np.abs(strength) >= self.trend_threshold

        # Momentum confirmation: sign matches direction
        confirm = (direction * mom) > 0

        pred = np.zeros(X.shape[0], dtype=int)
        keep = gate & strong & confirm
        pred[keep] = direction[keep]

        conf = np.clip((np.abs(strength) / max(1e-12, self.trend_threshold) - 1.0), 0.0, 3.0) / 3.0

        prob = np.zeros((X.shape[0], 3), dtype=float)
        prob[:, 0] = 0.10
        prob[:, 1] = 0.80
        prob[:, 2] = 0.10

        m = pred == 1
        if np.any(m):
            p = 0.55 + 0.40 * conf[m]
            prob[m, 2] = p
            prob[m, 1] = 1.0 - p
            prob[m, 0] = 0.0

        m = pred == -1
        if np.any(m):
            p = 0.55 + 0.40 * conf[m]
            prob[m, 0] = p
            prob[m, 1] = 1.0 - p
            prob[m, 2] = 0.0

        return pred, prob
