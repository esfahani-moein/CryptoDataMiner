"""
Strategy 14: Multiscale Confidence Filter
================================================================================
Goal: Robust intraday strategy across 1min / 15min / 1hr.

Design choices (firm-style, leakage-safe):
- Use 3-class labels (-1/0/+1) with volatility-scaled + cost-aware thresholds.
- Train a multiclass classifier, but only trade when model confidence is high.
- Use multiscale ("higher-timeframe") features via longer lookbacks.

Notes:
- This strategy intentionally prioritizes realistic trade selectivity.
- Profitability is not guaranteed; evaluation includes simple turnover costs.
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import LabelEncoder

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

from strategies.strategy_base import StrategyBase, StrategyResult


class MultiscaleConfidenceStrategy(StrategyBase):
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        confidence_threshold: float = 0.52,
        vol_k: float = 0.6,
        **kwargs,
    ):
        super().__init__(symbol, period, timeframe, **kwargs)
        self.confidence_threshold = confidence_threshold
        self.vol_k = vol_k
        self.label_encoder = LabelEncoder()

    def get_name(self) -> str:
        return f"Multiscale_Confidence_{self.timeframe}_ct{self.confidence_threshold:.2f}"

    def get_label_params(self) -> Dict[str, Any]:
        # Choose horizon in minutes, then convert to bars.
        tfm = self.timeframe_minutes
        if tfm <= 1:
            horizon_minutes = 30
        elif tfm <= 5:
            horizon_minutes = 60
        elif tfm <= 15:
            horizon_minutes = 60
        else:
            horizon_minutes = 120

        horizon = max(1, int(round(horizon_minutes / tfm)))

        return {
            "horizon": horizon,
            "mode": "vol",
            "vol_col": "vol_slow",
            "vol_k": float(self.vol_k),
            "threshold": 0.0,
            "cost_bps": self.get_trading_cost_bps(),
        }

    def get_feature_columns(self) -> List[str]:
        return [
            # Short-horizon returns/momentum
            "ret_1",
            "ret_3",
            "ret_5",
            "mom_fast",
            "mom_slow",

            # Volatility (fast/slow)
            "vol_fast",
            "vol_slow",
            "vol_ratio",
            "atr_norm",
            "range_norm",

            # Volume/liquidity normalized
            "volume_ratio",
            "trade_intensity",
            "avg_trade_size_norm",

            # Orderflow
            "taker_buy_ratio",
            "volume_imbalance",

            # Derivatives sentiment (if present)
            "oi_change",
            "ls_ratio",
            "funding_rate",
        ]

    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        tfm = self.timeframe_minutes

        # "higher timeframe" lookbacks expressed in bars
        bars_15m = max(1, int(round(15 / tfm)))
        bars_1h = max(1, int(round(60 / tfm)))

        # Returns
        df = df.with_columns([
            pl.col("close").pct_change(1).alias("ret_1"),
            pl.col("close").pct_change(3).alias("ret_3"),
            pl.col("close").pct_change(5).alias("ret_5"),
        ])

        df = df.with_columns([
            (pl.col("close") / pl.col("close").shift(bars_15m) - 1).alias("mom_fast"),
            (pl.col("close") / pl.col("close").shift(bars_1h) - 1).alias("mom_slow"),
        ])

        # Volatility
        df = df.with_columns([
            pl.col("ret_1").rolling_std(20).alias("vol_fast"),
            pl.col("ret_1").rolling_std(5 * bars_1h).alias("vol_slow"),
        ])

        df = df.with_columns([
            (pl.col("vol_fast") / (pl.col("vol_slow") + 1e-10)).alias("vol_ratio"),
        ])

        # Range and ATR normalized
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

        # Volume normalization
        df = df.with_columns([
            (pl.col("volume") / (pl.col("volume").rolling_mean(50) + 1e-10)).alias("volume_ratio"),
        ])

        # Trade intensity / avg trade size normalization
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

        # Order flow
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

        # Derivatives / sentiment (optional)
        # Metrics loader provides columns like sum_open_interest, count_long_short_ratio etc.
        if "sum_open_interest" in df.columns:
            df = df.with_columns([
                (pl.col("sum_open_interest") / (pl.col("sum_open_interest").shift(1) + 1e-10) - 1).alias("oi_change")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("oi_change")])

        if "count_long_short_ratio" in df.columns:
            df = df.with_columns([
                (pl.col("count_long_short_ratio") - pl.col("count_long_short_ratio").rolling_mean(200))
                .alias("ls_ratio")
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
        # Encode labels to 0..K-1
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)

        if HAS_LGB:
            model = lgb.LGBMClassifier(
                n_estimators=400,
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

            # Pick a confidence threshold on validation to reduce overtrading.
            try:
                probas_val = model.predict_proba(X_val)
                conf = np.max(probas_val, axis=1)
                pred_enc = np.argmax(probas_val, axis=1)
                pred = self.label_encoder.inverse_transform(pred_enc)

                # Approximate validation returns: use 1-bar forward return proxy.
                # In this framework, we don't pass returns into train_model.
                # We instead tune for *trade count stability* + confidence.
                # Keep it simple: choose threshold that keeps turnover moderate.
                # (PnL tuning should be done in run() where forward returns are available.)
                candidates = np.linspace(0.50, 0.70, 11)
                best_thr = self.confidence_threshold
                best_score = -1e18
                for thr in candidates:
                    pred_thr = np.where(conf >= thr, pred, 0)
                    pos = np.sign(pred_thr.astype(float))
                    changes = np.diff(np.concatenate([[0.0], pos]))
                    n_trades = int(np.sum(np.abs(changes) > 0))
                    # Prefer: not too many trades, not too few.
                    # Target ~2-10% of bars switching position.
                    target = 0.05 * len(pos)
                    score = -abs(n_trades - target)
                    if score > best_score:
                        best_score = score
                        best_thr = float(thr)

                self.confidence_threshold = best_thr
            except Exception:
                pass

            return model

        # Fallback: HistGradientBoosting multiclass
        model = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        )
        model.fit(X_train, y_train_enc)
        return model

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probas = self.model.predict_proba(X)
        # Confidence filter: trade only when max prob is high.
        conf = np.max(probas, axis=1)
        pred_enc = np.argmax(probas, axis=1)

        # Decode to original labels (-1/0/+1)
        pred = self.label_encoder.inverse_transform(pred_enc)

        # Force hold when confidence is low
        pred = np.where(conf >= self.confidence_threshold, pred, 0)

        return pred, probas


def run_strategy(timeframe: str = "15min") -> StrategyResult:
    strategy = MultiscaleConfidenceStrategy(
        symbol="BTCUSDT",
        period="2025_11",
        timeframe=timeframe,
    )
    return strategy.run(verbose=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Strategy 14: Multiscale Confidence")
    parser.add_argument("--timeframe", type=str, default="15min", choices=["1min", "5min", "15min", "1hr"])
    parser.add_argument("--confidence", type=float, default=0.52)
    parser.add_argument("--vol-k", type=float, default=0.6)
    args = parser.parse_args()

    result = MultiscaleConfidenceStrategy(
        symbol="BTCUSDT",
        period="2025_11",
        timeframe=args.timeframe,
        confidence_threshold=args.confidence,
        vol_k=args.vol_k,
    ).run(verbose=True)

    print(f"\nFinal Return: {result.total_return:.2%}")
    print(f"Sharpe: {result.sharpe_ratio:.2f}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
