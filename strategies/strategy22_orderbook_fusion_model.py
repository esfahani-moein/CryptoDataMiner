"""\
Strategy 22: Orderbook Fusion Model (L2 + Tick + Aux, ML)
=========================================================

Model-based strategy that fuses:
- Orderbook depth features from `bookDepth` (imbalance, liquidity, momentum).
- Tick-derived bar microstructure (taker flow, trade intensity, VPIN proxy).
- Aux data: open interest + long/short ratios (metrics), funding, mark price.
- Basis features from index/premium klines.

Key design choices:
- Strict point-in-time merge for orderbook + klines (asof join).
- Cost-aware, volatility-scaled labels (via StrategyBase.create_labels).
- Confidence + margin filter to reduce churn.

This is a stronger, "real" strategy that uses the datasets you have.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_base import StrategyBase


class OrderbookFusionModelStrategy(StrategyBase):
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        confidence_threshold: float = 0.52,
        margin_threshold: float = 0.10,
        **kwargs,
    ):
        super().__init__(symbol=symbol, period=period, timeframe=timeframe, **kwargs)
        self.confidence_threshold = float(confidence_threshold)
        self.margin_threshold = float(margin_threshold)
        self.label_encoder = LabelEncoder()

    def get_name(self) -> str:
        return f"Orderbook_Fusion_ct{self.confidence_threshold:.2f}_mt{self.margin_threshold:.2f}_{self.timeframe}"

    def get_label_params(self) -> Dict[str, Any]:
        # Align horizon with the speed of L2 signals.
        if self.timeframe == "1min":
            horizon = 10
        elif self.timeframe == "15min":
            horizon = 2
        elif self.timeframe == "1hr":
            horizon = 2
        else:
            horizon = max(2, int(round(30 / max(1, self.timeframe_minutes))))

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
            "depth_imbalance_ma_12",
            "depth_imbalance_std_12",
            "depth_imbalance_change",
            "total_liquidity",
            "liquidity_ratio_12",
            "bid_ask_ratio_log",
            "depth_concentration",
            "liquidity_change_pct",
            # Tick/microstructure (from bars)
            "taker_buy_ratio",
            "trade_intensity",
            "avg_trade_size",
            "vpin",
            "kyle_lambda",
            "amihud",
            # Price context
            "ret_1",
            "ret_3",
            "vol_fast",
            "vol_slow",
            # Aux (if present)
            "last_funding_rate",
            "oi_change",
            "ls_ratio_change",
            "mark_index_basis",
            "premium_index",
        ]

    def _merge_book_and_klines(self, df: pl.DataFrame) -> pl.DataFrame:
        from quant_features.data_loader import load_book_depth, load_klines, merge_features_to_ohlcv
        from quant_features.orderbook_features import add_all_orderbook_features

        tfm = max(1, self.timeframe_minutes)
        bars_15m = max(2, int(round(15 / tfm)))
        windows = [max(6, bars_15m), max(12, 2 * bars_15m), max(24, 4 * bars_15m)]

        # Book depth
        try:
            book = load_book_depth(
                self.data_path,
                symbol=self.symbol,
                start_year=self.start_year,
                start_month=self.start_month,
                end_year=self.end_year,
                end_month=self.end_month,
            )
            book_feat = add_all_orderbook_features(book, windows=windows)
            df = merge_features_to_ohlcv(df, book_feat, ohlcv_time_col="timestamp", features_time_col="time")
        except Exception:
            pass

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

        if "time" in df.columns:
            df = df.drop("time")

        return df

    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        tfm = max(1, self.timeframe_minutes)
        bars_15m = max(2, int(round(15 / tfm)))
        bars_1h = max(2, int(round(60 / tfm)))

        # Returns
        df = df.with_columns([
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("ret_1"),
            (pl.col("close").log() - pl.col("close").shift(3).log()).alias("ret_3"),
        ])

        df = df.with_columns([
            pl.col("ret_1").rolling_std(bars_15m).alias("vol_fast"),
            pl.col("ret_1").rolling_std(max(20, 4 * bars_1h)).alias("vol_slow"),
        ])

        # Tick-derived bar stats
        if "count" in df.columns:
            df = df.with_columns([
                (pl.col("count") / (pl.col("count").rolling_mean(20) + 1e-12)).alias("trade_intensity"),
                (pl.col("volume") / (pl.col("count") + 1e-12)).alias("avg_trade_size"),
            ])
        else:
            df = df.with_columns([
                pl.lit(1.0).alias("trade_intensity"),
                pl.lit(0.0).alias("avg_trade_size"),
            ])

        if "taker_buy_volume" in df.columns:
            df = df.with_columns([
                (pl.col("taker_buy_volume") / (pl.col("volume") + 1e-12)).alias("taker_buy_ratio")
            ])
        else:
            df = df.with_columns([pl.lit(0.5).alias("taker_buy_ratio")])

        # VPIN proxy from volume imbalance
        if "taker_buy_ratio" in df.columns:
            df = df.with_columns([
                (2.0 * (pl.col("taker_buy_ratio") - 0.5)).alias("flow_imbalance"),
            ])
            df = df.with_columns([
                pl.col("flow_imbalance").abs().rolling_mean(max(20, 4 * bars_15m)).alias("vpin"),
            ])
        else:
            df = df.with_columns([
                pl.lit(0.0).alias("flow_imbalance"),
                pl.lit(0.0).alias("vpin"),
            ])

        # Price impact proxies
        df = df.with_columns([
            (pl.col("ret_1").abs() / (pl.col("volume") + 1e-12) * 1e6).alias("kyle_lambda"),
            (pl.col("ret_1").abs() / (pl.col("volume") * pl.col("close") + 1e-12) * 1e9).alias("amihud"),
        ])

        # Metrics
        if "sum_open_interest" in df.columns:
            df = df.with_columns([
                ((pl.col("sum_open_interest") - pl.col("sum_open_interest").shift(1)) / (pl.col("sum_open_interest").shift(1) + 1e-12)).alias("oi_change")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("oi_change")])

        if "count_long_short_ratio" in df.columns:
            df = df.with_columns([
                (pl.col("count_long_short_ratio") - pl.col("count_long_short_ratio").shift(1)).alias("ls_ratio_change")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("ls_ratio_change")])

        # Merge orderbook + klines
        df = self._merge_book_and_klines(df)

        # Basis
        if "mark_price" in df.columns and "index_price" in df.columns:
            df = df.with_columns([
                ((pl.col("mark_price") - pl.col("index_price")) / (pl.col("index_price") + 1e-12)).alias("mark_index_basis")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("mark_index_basis")])

        # Ensure premium exists
        if "premium_index" not in df.columns:
            df = df.with_columns([pl.lit(0.0).alias("premium_index")])

        # Fill missing orderbook fields with zeros if not merged
        for col in [
            "depth_imbalance",
            "notional_imbalance",
            "depth_imbalance_change",
            "total_liquidity",
            "bid_ask_ratio_log",
            "depth_concentration",
            "liquidity_change_pct",
            "depth_imbalance_ma_12",
            "depth_imbalance_std_12",
            "liquidity_ratio_12",
        ]:
            if col not in df.columns:
                df = df.with_columns([pl.lit(0.0).alias(col)])

        return df

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Any:
        y_train_enc = self.label_encoder.fit_transform(y_train)

        if HAS_LGB:
            model = lgb.LGBMClassifier(
                n_estimators=600,
                learning_rate=0.03,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_samples=50,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="multiclass",
                num_class=len(self.label_encoder.classes_),
                random_state=42,
                n_jobs=-1,
            )
            model.fit(
                X_train,
                y_train_enc,
                sample_weight=sample_weights,
            )
            return model

        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=400,
            random_state=42,
        )
        model.fit(X_train, y_train_enc, sample_weight=sample_weights)
        return model

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(X)
        else:
            n = X.shape[0]
            k = len(self.label_encoder.classes_)
            prob = np.full((n, k), 1.0 / max(1, k), dtype=float)

        pred_enc = np.argmax(prob, axis=1)
        pred = self.label_encoder.inverse_transform(pred_enc)

        # Confidence filter
        max_prob = np.max(prob, axis=1)

        # Margin filter: require separation between best and second best
        prob_sorted = np.sort(prob, axis=1)
        margin = prob_sorted[:, -1] - prob_sorted[:, -2] if prob.shape[1] >= 2 else np.zeros(len(max_prob))

        keep = (max_prob >= self.confidence_threshold) & (margin >= self.margin_threshold)
        pred = np.where(keep, pred, 0)

        return pred.astype(int), prob
