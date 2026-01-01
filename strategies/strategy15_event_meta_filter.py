"""\
Strategy 15: Event Meta Filter (Selective Trading)
================================================================================
Purpose: A pragmatic, firm-style intraday strategy that tends to generalize
better than always-in classifiers.

Core idea:
- Labels are 3-class (-1/0/+1) with volatility-scaled + cost-aware thresholds.
- Model predicts class probabilities.
- Trade only when the model is confident AND the long-vs-short margin is large.

This typically reduces turnover and makes 1min evaluation more realistic.
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Any, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.preprocessing import LabelEncoder

from strategies.strategy_base import StrategyBase, StrategyResult


class EventMetaFilterStrategy(StrategyBase):
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        min_conf: float = 0.55,
        min_margin: float = 0.10,
        vol_k: float = 0.6,
        **kwargs,
    ):
        super().__init__(symbol, period, timeframe, **kwargs)
        self.min_conf = float(min_conf)
        self.min_margin = float(min_margin)
        self.vol_k = float(vol_k)
        self.label_encoder = LabelEncoder()

    def get_name(self) -> str:
        return f"Event_Meta_Filter_{self.timeframe}_c{self.min_conf:.2f}_m{self.min_margin:.2f}"

    def get_label_params(self) -> Dict[str, Any]:
        tfm = self.timeframe_minutes
        # Holding horizon in minutes mapped to bars
        horizon_minutes = 30 if tfm <= 5 else (60 if tfm <= 15 else 120)
        horizon = max(1, int(round(horizon_minutes / tfm)))

        return {
            "horizon": horizon,
            "mode": "vol",
            "vol_col": "vol_slow",
            "vol_k": self.vol_k,
            "threshold": 0.0,
            "cost_bps": self.get_trading_cost_bps(),
        }

    def get_feature_columns(self) -> List[str]:
        return [
            "ret_1",
            "mom_fast",
            "mom_slow",
            "vol_fast",
            "vol_slow",
            "vol_ratio",
            "atr_norm",
            "range_norm",
            "volume_ratio",
            "trade_intensity",
            "avg_trade_size_norm",
            "taker_buy_ratio",
            "volume_imbalance",
            "oi_change",
            "ls_ratio",
            "funding_rate",
        ]

    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        tfm = self.timeframe_minutes
        bars_15m = max(1, int(round(15 / tfm)))
        bars_1h = max(1, int(round(60 / tfm)))

        df = df.with_columns([
            pl.col("close").pct_change(1).alias("ret_1"),
            (pl.col("close") / pl.col("close").shift(bars_15m) - 1).alias("mom_fast"),
            (pl.col("close") / pl.col("close").shift(bars_1h) - 1).alias("mom_slow"),
        ])

        slow_win = max(30, 5 * bars_1h)
        df = df.with_columns([
            pl.col("ret_1").rolling_std(20).alias("vol_fast"),
            pl.col("ret_1").rolling_std(slow_win).alias("vol_slow"),
        ])
        df = df.with_columns([
            (pl.col("vol_fast") / (pl.col("vol_slow") + 1e-10)).alias("vol_ratio"),
        ])

        df = df.with_columns([
            (pl.col("high") - pl.col("low")).alias("_hl"),
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs(),
            ).alias("_tr"),
        ])
        df = df.with_columns([
            (pl.col("_hl") / (pl.col("close") + 1e-10)).alias("range_norm"),
            (pl.col("_tr").rolling_mean(14) / (pl.col("close") + 1e-10)).alias("atr_norm"),
        ])

        df = df.with_columns([
            (pl.col("volume") / (pl.col("volume").rolling_mean(50) + 1e-10)).alias("volume_ratio"),
        ])

        if "count" in df.columns:
            avg_trade = pl.col("volume") / (pl.col("count") + 1)
            df = df.with_columns([
                (pl.col("count") / (pl.col("count").rolling_mean(50) + 1)).alias("trade_intensity"),
                (avg_trade / (avg_trade.rolling_mean(50) + 1e-10)).alias("avg_trade_size_norm"),
            ])
        else:
            df = df.with_columns([
                pl.lit(1.0).alias("trade_intensity"),
                pl.lit(1.0).alias("avg_trade_size_norm"),
            ])

        if "taker_buy_volume" in df.columns:
            taker_sell = pl.col("volume") - pl.col("taker_buy_volume")
            df = df.with_columns([
                (pl.col("taker_buy_volume") / (pl.col("volume") + 1e-10)).alias("taker_buy_ratio"),
                ((pl.col("taker_buy_volume") - taker_sell) / (pl.col("volume") + 1e-10)).alias("volume_imbalance"),
            ])
        else:
            df = df.with_columns([
                pl.lit(0.5).alias("taker_buy_ratio"),
                pl.lit(0.0).alias("volume_imbalance"),
            ])

        if "sum_open_interest" in df.columns:
            df = df.with_columns([
                (pl.col("sum_open_interest") / (pl.col("sum_open_interest").shift(1) + 1e-10) - 1).alias("oi_change")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("oi_change")])

        if "count_long_short_ratio" in df.columns:
            df = df.with_columns([
                (pl.col("count_long_short_ratio") - pl.col("count_long_short_ratio").rolling_mean(200)).alias("ls_ratio")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("ls_ratio")])

        if "last_funding_rate" in df.columns:
            df = df.with_columns([pl.col("last_funding_rate").alias("funding_rate")])
        else:
            df = df.with_columns([pl.lit(0.0).alias("funding_rate")])

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
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                objective="multiclass",
                class_weight="balanced",
                verbose=-1,
            )
            try:
                model.fit(X_train, y_train_enc, sample_weight=sample_weights)
            except Exception:
                model.fit(X_train, y_train_enc)
            return model

        model = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=400,
            random_state=42,
        )
        model.fit(X_train, y_train_enc)
        return model

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probas = self.model.predict_proba(X)

        # Map probabilities to class labels via the encoder order
        classes = list(self.label_encoder.classes_)
        idx_long = classes.index(1) if 1 in classes else None
        idx_short = classes.index(-1) if -1 in classes else None

        conf = np.max(probas, axis=1)

        # Default to hold
        pred = np.zeros(len(X), dtype=int)

        if idx_long is None or idx_short is None:
            pred_enc = np.argmax(probas, axis=1)
            pred = self.label_encoder.inverse_transform(pred_enc)
            return pred, probas

        margin = probas[:, idx_long] - probas[:, idx_short]
        go_long = (conf >= self.min_conf) & (margin >= self.min_margin)
        go_short = (conf >= self.min_conf) & (margin <= -self.min_margin)

        pred = np.where(go_long, 1, pred)
        pred = np.where(go_short, -1, pred)

        return pred, probas


def run_strategy(timeframe: str = "15min") -> StrategyResult:
    strategy = EventMetaFilterStrategy(symbol="BTCUSDT", period="2025_11", timeframe=timeframe)
    return strategy.run(verbose=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Strategy 15: Event Meta Filter")
    parser.add_argument("--timeframe", type=str, default="15min", choices=["1min", "5min", "15min", "1hr"])
    parser.add_argument("--min-conf", type=float, default=0.55)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--vol-k", type=float, default=0.6)
    args = parser.parse_args()

    result = EventMetaFilterStrategy(
        symbol="BTCUSDT",
        period="2025_11",
        timeframe=args.timeframe,
        min_conf=args.min_conf,
        min_margin=args.min_margin,
        vol_k=args.vol_k,
    ).run(verbose=True)

    print(f"\nFinal Return: {result.total_return:.2%}")
    print(f"Sharpe: {result.sharpe_ratio:.2f}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
