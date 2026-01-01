"""\
Strategy 19: Bollinger + RSI Mean Reversion (Rule-Based)
========================================================

Non-ML strategy that trades statistically extreme deviations:
- Long when price is far below mean (zscore low) AND RSI is oversold.
- Short when price is far above mean (zscore high) AND RSI is overbought.

Includes a volatility gate favoring mean-reversion regimes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_base import StrategyBase


class BollingerReversionStrategy(StrategyBase):
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        z_entry: float = 2.0,
        rsi_low: float = 30.0,
        rsi_high: float = 70.0,
        vol_gate_max: float = 0.60,
        **kwargs,
    ):
        super().__init__(symbol=symbol, period=period, timeframe=timeframe, **kwargs)
        self.z_entry = float(z_entry)
        self.rsi_low = float(rsi_low)
        self.rsi_high = float(rsi_high)
        self.vol_gate_max = float(vol_gate_max)

        self._bb_window = self._choose_bb_window()
        self._rsi_window = self._choose_rsi_window()

    def _choose_bb_window(self) -> int:
        # Target ~3 hours lookback
        target_minutes = 180
        return max(20, int(round(target_minutes / max(1, self.timeframe_minutes))))

    def _choose_rsi_window(self) -> int:
        # Target ~1.5 hours lookback
        target_minutes = 90
        return max(14, int(round(target_minutes / max(1, self.timeframe_minutes))))

    def get_name(self) -> str:
        return f"BB_Reversion_z{self.z_entry:.1f}_{self.timeframe}"

    def get_label_params(self) -> Dict[str, Any]:
        # Mean-reversion tends to be quicker.
        if self.timeframe == "1min":
            horizon = 10
        elif self.timeframe == "15min":
            horizon = 2
        elif self.timeframe == "1hr":
            horizon = 2
        else:
            horizon = max(1, int(round(30 / max(1, self.timeframe_minutes))))

        return {
            "horizon": int(horizon),
            "threshold": 0.0001,
            "mode": "vol",
            "vol_col": "vol_slow",
            "vol_k": 0.60,
            "cost_bps": self.get_trading_cost_bps(),
        }

    def get_feature_columns(self) -> List[str]:
        return [
            "zscore",
            "rsi",
            "bb_pos",
            "vol_slow",
            "vol_percentile",
            "volume_ratio",
        ]

    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        bb_w = int(self._bb_window)
        rsi_w = int(self._rsi_window)
        pct_window = max(30, 24 * max(2, int(round(60 / max(1, self.timeframe_minutes)))))

        # Returns + slow vol
        df = df.with_columns([
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("ret_1"),
        ])
        df = df.with_columns([
            pl.col("ret_1").rolling_std(max(20, bb_w)).alias("vol_slow"),
        ])
        df = df.with_columns([
            (pl.col("vol_slow").rolling_rank(pct_window) / pct_window).alias("vol_percentile"),
        ])

        # Bollinger stats
        df = df.with_columns([
            pl.col("close").rolling_mean(bb_w).alias("bb_mid"),
            pl.col("close").rolling_std(bb_w).alias("bb_std"),
        ])
        df = df.with_columns([
            ((pl.col("close") - pl.col("bb_mid")) / (pl.col("bb_std") + 1e-12)).alias("zscore"),
            (
                (pl.col("close") - (pl.col("bb_mid") - 2.0 * pl.col("bb_std")))
                / (4.0 * pl.col("bb_std") + 1e-12)
            ).alias("bb_pos"),
        ])

        # RSI
        df = df.with_columns([
            (pl.col("close") - pl.col("close").shift(1)).alias("_delta"),
        ])
        df = df.with_columns([
            pl.when(pl.col("_delta") > 0).then(pl.col("_delta")).otherwise(0.0).alias("_gain"),
            pl.when(pl.col("_delta") < 0).then(-pl.col("_delta")).otherwise(0.0).alias("_loss"),
        ])
        df = df.with_columns([
            pl.col("_gain").rolling_mean(rsi_w).alias("_avg_gain"),
            pl.col("_loss").rolling_mean(rsi_w).alias("_avg_loss"),
        ])
        df = df.with_columns([
            (100.0 - (100.0 / (1.0 + (pl.col("_avg_gain") / (pl.col("_avg_loss") + 1e-12))))).alias("rsi"),
        ])

        df = df.with_columns([
            (pl.col("volume") / (pl.col("volume").rolling_mean(20) + 1e-12)).alias("volume_ratio"),
        ])

        # Cleanup temp cols
        df = df.drop([c for c in ["_delta", "_gain", "_loss", "_avg_gain", "_avg_loss"] if c in df.columns])

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
            "z_entry": self.z_entry,
            "rsi_low": self.rsi_low,
            "rsi_high": self.rsi_high,
            "vol_gate_max": self.vol_gate_max,
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = {name: i for i, name in enumerate(self.feature_names)}

        z = X[:, idx["zscore"]]
        rsi = X[:, idx["rsi"]]
        vol_pct = X[:, idx["vol_percentile"]]

        # Prefer lower-vol regimes for reversion
        gate = vol_pct <= self.vol_gate_max

        long_sig = gate & (z <= -self.z_entry) & (rsi <= self.rsi_low)
        short_sig = gate & (z >= self.z_entry) & (rsi >= self.rsi_high)

        pred = np.zeros(X.shape[0], dtype=int)
        pred[long_sig] = 1
        pred[short_sig] = -1

        # Confidence from how extreme zscore is (clipped)
        conf = np.clip((np.abs(z) - self.z_entry) / max(1e-12, self.z_entry), 0.0, 2.0) / 2.0

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
