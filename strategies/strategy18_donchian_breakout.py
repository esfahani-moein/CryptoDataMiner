"""\
Strategy 18: Donchian Breakout (Rule-Based)
==========================================

Non-ML strategy focused on capturing directional breakouts using only past data.

Signal:
- Long when close breaks above the prior Donchian high.
- Short when close breaks below the prior Donchian low.
- Optional volatility gate to avoid dead/ultra-chaotic periods.

This strategy outputs positions directly in `predict()`; `train_model()` is a
no-op to keep compatibility with the shared StrategyBase pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_base import StrategyBase


class DonchianBreakoutStrategy(StrategyBase):
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        vol_gate_low: float = 0.20,
        vol_gate_high: float = 0.95,
        **kwargs,
    ):
        super().__init__(symbol=symbol, period=period, timeframe=timeframe, **kwargs)
        self.vol_gate_low = float(vol_gate_low)
        self.vol_gate_high = float(vol_gate_high)

        self._lookback = self._choose_lookback()

    def _choose_lookback(self) -> int:
        # Target ~6 hours of lookback for channel; enforce minimum.
        target_minutes = 6 * 60
        bars = max(20, int(round(target_minutes / max(1, self.timeframe_minutes))))
        return bars

    def get_name(self) -> str:
        return f"Donchian_Breakout_L{self._lookback}_{self.timeframe}"

    def get_label_params(self) -> Dict[str, Any]:
        # Align horizon with a modest holding period so the PnL proxy makes sense.
        # For 1m: 30m, for 15m: 2h, for 1h: 4h
        if self.timeframe == "1min":
            horizon = 30
        elif self.timeframe == "15min":
            horizon = 8
        elif self.timeframe == "1hr":
            horizon = 4
        else:
            horizon = max(2, int(round(120 / max(1, self.timeframe_minutes))))

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
            "close",
            "dc_high",
            "dc_low",
            "dc_width",
            "break_strength",
            "vol_slow",
            "vol_percentile",
            "volume_ratio",
        ]

    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        lookback = int(self._lookback)
        pct_window = max(30, 24 * max(2, int(round(60 / max(1, self.timeframe_minutes)))))

        # Log return
        df = df.with_columns([
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("ret_1"),
        ])

        # Slow vol (uses past data only)
        df = df.with_columns([
            pl.col("ret_1").rolling_std(max(20, lookback)).alias("vol_slow"),
        ])

        # Vol percentile gate (past-only rolling rank)
        df = df.with_columns([
            (pl.col("vol_slow").rolling_rank(pct_window) / pct_window).alias("vol_percentile"),
        ])

        # Donchian channel on PRIOR highs/lows only
        df = df.with_columns([
            pl.col("high").shift(1).rolling_max(lookback).alias("dc_high"),
            pl.col("low").shift(1).rolling_min(lookback).alias("dc_low"),
        ])

        df = df.with_columns([
            ((pl.col("dc_high") - pl.col("dc_low")) / (pl.col("close") + 1e-12)).alias("dc_width"),
            (
                (pl.col("close") - pl.col("dc_high")) / (pl.col("vol_slow") + 1e-12)
            ).alias("break_strength"),
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
        # No ML; return parameters to satisfy pipeline.
        return {
            "lookback": self._lookback,
            "vol_gate_low": self.vol_gate_low,
            "vol_gate_high": self.vol_gate_high,
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Map feature names to indices
        idx = {name: i for i, name in enumerate(self.feature_names)}

        close = X[:, idx["close"]]
        dc_high = X[:, idx["dc_high"]]
        dc_low = X[:, idx["dc_low"]]
        vol_pct = X[:, idx["vol_percentile"]]
        strength = X[:, idx["break_strength"]]

        gate = (vol_pct >= self.vol_gate_low) & (vol_pct <= self.vol_gate_high)

        long_sig = gate & (close > dc_high)
        short_sig = gate & (close < dc_low)

        pred = np.zeros(X.shape[0], dtype=int)
        pred[long_sig] = 1
        pred[short_sig] = -1

        # Confidence from breakout strength (clipped)
        conf = np.clip(np.abs(strength), 0.0, 3.0) / 3.0

        # Build 3-class probabilities in label order [-1, 0, 1]
        prob = np.zeros((X.shape[0], 3), dtype=float)

        # Default: mostly hold
        prob[:, 0] = 0.10
        prob[:, 1] = 0.80
        prob[:, 2] = 0.10

        # Long
        m = pred == 1
        if np.any(m):
            p = 0.55 + 0.40 * conf[m]
            prob[m, 2] = p
            prob[m, 1] = 1.0 - p
            prob[m, 0] = 0.0

        # Short
        m = pred == -1
        if np.any(m):
            p = 0.55 + 0.40 * conf[m]
            prob[m, 0] = p
            prob[m, 1] = 1.0 - p
            prob[m, 2] = 0.0

        return pred, prob
