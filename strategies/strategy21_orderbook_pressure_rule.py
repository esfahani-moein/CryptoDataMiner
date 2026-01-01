"""\
Strategy 21: Orderbook Pressure (Rule-Based, Tick+L2+Aux Fusion)
===============================================================

Uses:
- Tick-derived bar stats from trades aggregation (taker buy ratio, count if present).
- L2 book depth snapshots (bookDepth) -> depth imbalance + liquidity features.
- Aux data (metrics + funding + mark price already merged in StrategyBase).
- Optional index/premium klines for basis features.

This is intentionally rule-based (no ML) to provide a strong non-ML baseline
that *actually* uses the orderbook dataset.

Design goals:
- Trade only when orderbook imbalance is extreme and liquidity is healthy.
- Use a confirm filter (taker flow + liquidity) to reduce churn.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_base import StrategyBase


class OrderbookPressureRuleStrategy(StrategyBase):
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        imbalance_threshold: float = 0.12,
        min_confidence: float = 0.00,
        **kwargs,
    ):
        super().__init__(symbol=symbol, period=period, timeframe=timeframe, **kwargs)
        self.imbalance_threshold = float(imbalance_threshold)
        self.min_confidence = float(min_confidence)

    def get_name(self) -> str:
        return f"Orderbook_Pressure_Rule_it{self.imbalance_threshold:.2f}_ct{self.min_confidence:.2f}_{self.timeframe}"

    def get_label_params(self) -> Dict[str, Any]:
        # Label horizon is used only for evaluation/PnL proxy.
        # Keep moderate to avoid over-penalizing slower signals.
        if self.timeframe == "1min":
            horizon = 15
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
            "vol_k": 0.75,
            "cost_bps": self.get_trading_cost_bps(),
        }

    def get_feature_columns(self) -> List[str]:
        return [
            # Orderbook
            "depth_imbalance",
            "notional_imbalance",
            "depth_imbalance_change",
            "total_liquidity",
            "liquidity_ratio_12",
            "bid_ask_ratio_log",
            # Tick-derived bar fields (if present)
            "taker_buy_ratio",
            "trade_intensity",
            # Price/vol context
            "ret_1",
            "vol_fast",
            "vol_slow",
            # Aux
            "last_funding_rate",
            "oi_change",
            "mark_index_basis",
        ]

    def _maybe_merge_index_premium(self, df: pl.DataFrame) -> pl.DataFrame:
        from quant_features.data_loader import load_klines, merge_features_to_ohlcv

        # Index price
        try:
            index_df = load_klines(
                self.data_path,
                symbol=self.symbol,
                kline_type="indexPriceKlines",
                start_year=self.start_year,
                start_month=self.start_month,
                end_year=self.end_year,
                end_month=self.end_month,
            )
            if "open_time" in index_df.columns and "close" in index_df.columns:
                index_df = index_df.select(["open_time", "close"]).rename({"open_time": "timestamp", "close": "index_price"})
                df = merge_features_to_ohlcv(df, index_df, ohlcv_time_col="timestamp", features_time_col="timestamp")
        except Exception:
            pass

        # Premium index
        try:
            prem_df = load_klines(
                self.data_path,
                symbol=self.symbol,
                kline_type="premiumIndexKlines",
                start_year=self.start_year,
                start_month=self.start_month,
                end_year=self.end_year,
                end_month=self.end_month,
            )
            if "open_time" in prem_df.columns and "close" in prem_df.columns:
                prem_df = prem_df.select(["open_time", "close"]).rename({"open_time": "timestamp", "close": "premium_index"})
                df = merge_features_to_ohlcv(df, prem_df, ohlcv_time_col="timestamp", features_time_col="timestamp")
        except Exception:
            pass

        if "mark_price" in df.columns and "index_price" in df.columns:
            df = df.with_columns([
                ((pl.col("mark_price") - pl.col("index_price")) / (pl.col("index_price") + 1e-12)).alias("mark_index_basis")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("mark_index_basis")])

        return df

    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        from quant_features.data_loader import load_book_depth, merge_features_to_ohlcv
        from quant_features.orderbook_features import add_all_orderbook_features

        tfm = max(1, self.timeframe_minutes)
        bars_15m = max(2, int(round(15 / tfm)))
        bars_1h = max(2, int(round(60 / tfm)))

        # Price context
        df = df.with_columns([
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("ret_1"),
        ])

        df = df.with_columns([
            pl.col("ret_1").rolling_std(bars_15m).alias("vol_fast"),
            pl.col("ret_1").rolling_std(max(20, 4 * bars_1h)).alias("vol_slow"),
        ])

        # Tick-derived bar features (from aggregated trades)
        if "taker_buy_volume" in df.columns:
            df = df.with_columns([
                (pl.col("taker_buy_volume") / (pl.col("volume") + 1e-12)).alias("taker_buy_ratio")
            ])
        else:
            df = df.with_columns([pl.lit(0.5).alias("taker_buy_ratio")])

        if "count" in df.columns:
            df = df.with_columns([
                (pl.col("count") / (pl.col("count").rolling_mean(20) + 1e-12)).alias("trade_intensity")
            ])
        else:
            df = df.with_columns([pl.lit(1.0).alias("trade_intensity")])

        # Metrics proxy: open interest change
        if "sum_open_interest" in df.columns:
            df = df.with_columns([
                ((pl.col("sum_open_interest") - pl.col("sum_open_interest").shift(1)) / (pl.col("sum_open_interest").shift(1) + 1e-12)).alias("oi_change")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("oi_change")])

        # Merge orderbook features (point-in-time)
        try:
            book = load_book_depth(
                self.data_path,
                symbol=self.symbol,
                start_year=self.start_year,
                start_month=self.start_month,
                end_year=self.end_year,
                end_month=self.end_month,
            )
            # Windows in *snapshots*; use small/medium/large.
            windows = [max(6, bars_15m), max(12, 2 * bars_15m), max(24, 4 * bars_15m)]
            book_feat = add_all_orderbook_features(book, windows=windows)
            df = merge_features_to_ohlcv(df, book_feat, ohlcv_time_col="timestamp", features_time_col="time")
        except Exception:
            # If anything fails, create placeholders.
            df = df.with_columns([
                pl.lit(0.0).alias("depth_imbalance"),
                pl.lit(0.0).alias("notional_imbalance"),
                pl.lit(0.0).alias("depth_imbalance_change"),
                pl.lit(0.0).alias("total_liquidity"),
                pl.lit(1.0).alias("liquidity_ratio_12"),
                pl.lit(0.0).alias("bid_ask_ratio_log"),
            ])

        # Basis features (index/premium)
        df = self._maybe_merge_index_premium(df)

        # Clean up likely duplicated time column from asof join
        if "time" in df.columns:
            df = df.drop("time")

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
            "imbalance_threshold": self.imbalance_threshold,
            "min_confidence": self.min_confidence,
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = {name: i for i, name in enumerate(self.feature_names)}

        di = np.nan_to_num(X[:, idx.get("depth_imbalance", 0)], nan=0.0)
        ni = np.nan_to_num(X[:, idx.get("notional_imbalance", 0)], nan=0.0)
        di_chg = np.nan_to_num(X[:, idx.get("depth_imbalance_change", 0)], nan=0.0)
        liq_ratio = np.nan_to_num(X[:, idx.get("liquidity_ratio_12", 0)], nan=1.0)
        taker = X[:, idx.get("taker_buy_ratio", 0)]
        vol_fast = X[:, idx.get("vol_fast", 0)]
        vol_slow = X[:, idx.get("vol_slow", 0)]

        # Gate: avoid ultra-chaotic microstructure (fast vol spikes)
        gate_vol = (np.nan_to_num(vol_fast) <= (np.nan_to_num(vol_slow) * 2.5 + 1e-12))
        gate_liq = liq_ratio >= 0.9

        score = 0.55 * di + 0.35 * ni + 0.10 * np.sign(di_chg) * np.abs(di)

        # Flow confirmation: if taker flow isn't available (defaults to ~0.5),
        # don't block trades; otherwise require agreement.
        has_flow = np.abs(np.nan_to_num(taker, nan=0.5) - 0.5) > 1e-6
        flow_ok_long = (~has_flow) | (taker >= 0.52)
        flow_ok_short = (~has_flow) | (taker <= 0.48)

        pred = np.zeros(X.shape[0], dtype=int)

        long_sig = gate_vol & gate_liq & (score >= self.imbalance_threshold) & flow_ok_long
        short_sig = gate_vol & gate_liq & (score <= -self.imbalance_threshold) & flow_ok_short

        pred[long_sig] = 1
        pred[short_sig] = -1

        # Confidence based on how far beyond threshold we are.
        # At |score| == threshold => 0.0, at |score| == 2*threshold => 1.0
        conf = np.clip((np.abs(score) - self.imbalance_threshold) / max(1e-12, self.imbalance_threshold), 0.0, 1.0)

        # Optional global min-confidence filter
        if self.min_confidence > 0:
            pred = np.where(conf >= self.min_confidence, pred, 0)

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
