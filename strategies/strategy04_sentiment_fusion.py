"""
Strategy 04: Sentiment Fusion
=============================

Combines multiple sentiment indicators from Binance Futures:
- Funding Rate (+ premium index)
- Open Interest
- Long/Short Ratios (top traders, accounts)
- Liquidation signals

Focus on crypto-specific sentiment that drives price movements.
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_base import StrategyBase


class SentimentFusionStrategy(StrategyBase):
    """
    Sentiment-based strategy using crypto-specific indicators.
    
    Key Insights:
    - High funding rate -> longs pay shorts -> potential reversal down
    - Extreme long/short ratios -> crowded trade -> potential reversal
    - OI increasing + price increasing -> strong trend
    - OI increasing + price decreasing -> forced liquidations possible
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        **kwargs
    ):
        super().__init__(symbol, period, timeframe, **kwargs)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def get_name(self) -> str:
        return f"Sentiment_Fusion_{self.timeframe}"
    
    def get_feature_columns(self) -> List[str]:
        """Sentiment-focused features."""
        features = []
        
        # === Funding Rate Features ===
        features.extend([
            "funding_rate",
            "funding_rate_ma_8",  # 8-hour funding
            "funding_rate_ma_24",  # 24-hour funding
            "funding_rate_zscore",
            "funding_rate_extreme",  # Binary: extreme funding
            "funding_rate_direction",  # Sign change detection
            "cumulative_funding_8",
            "cumulative_funding_24",
        ])
        
        # === Premium Index Features ===
        features.extend([
            "premium_index",
            "premium_index_ma",
            "premium_vs_funding",
            "basis",
            "basis_ma",
            "basis_zscore",
        ])
        
        # === Open Interest Features ===
        features.extend([
            "sum_open_interest",
            "oi_change",
            "oi_change_pct",
            "oi_ma_24",
            "oi_vs_ma",
            "oi_zscore",
            "oi_momentum",
        ])
        
        # OI + Price divergence
        features.extend([
            "oi_price_divergence",
            "oi_price_trend",  # Same direction = strong, different = weak
        ])
        
        # === Long/Short Ratio Features ===
        # Top Trader Position Ratio
        features.extend([
            "sum_toptrader_long_short_ratio",
            "top_ratio_ma_8",
            "top_ratio_zscore",
            "top_ratio_extreme",
            "top_ratio_momentum",
        ])
        
        # Top Trader Account Ratio  
        features.extend([
            "count_toptrader_long_short_ratio",
            "account_ratio_ma_8",
            "account_ratio_zscore",
            "account_ratio_extreme",
        ])
        
        # Long/Short difference
        features.extend([
            "ratio_divergence",  # Position vs Account ratio
            "crowd_sentiment",  # Aggregate sentiment score
        ])
        
        # === Composite Sentiment Indicators ===
        features.extend([
            "sentiment_score",  # Combined weighted score
            "sentiment_extreme",  # Extreme readings
            "sentiment_momentum",  # Change in sentiment
            "contrarian_signal",  # Contrarian opportunity
        ])
        
        # === Price Context Features ===
        for w in [5, 10, 20]:
            features.extend([
                f"ret_{w}",
                f"vol_{w}",
            ])
        
        # Sentiment-price alignment
        features.extend([
            "sentiment_price_alignment",
            "funding_price_alignment",
            "oi_price_alignment",
        ])
        
        return features
    
    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create sentiment and derived features."""
        
        # === Funding Rate Features ===
        if "last_funding_rate" in df.columns:
            df = df.rename({"last_funding_rate": "funding_rate"})
        
        if "funding_rate" in df.columns:
            # Funding rate rolling averages (8h = funding period)
            bars_per_8h = 8 * 60 // self.timeframe_minutes
            bars_per_24h = 24 * 60 // self.timeframe_minutes
            
            df = df.with_columns([
                pl.col("funding_rate").rolling_mean(window_size=max(1, bars_per_8h)).alias("funding_rate_ma_8"),
                pl.col("funding_rate").rolling_mean(window_size=max(1, bars_per_24h)).alias("funding_rate_ma_24"),
            ])
            
            # Z-score
            df = df.with_columns([
                ((pl.col("funding_rate") - pl.col("funding_rate").rolling_mean(window_size=max(1, bars_per_24h))) /
                 (pl.col("funding_rate").rolling_std(window_size=max(1, bars_per_24h)) + 1e-10)).alias("funding_rate_zscore")
            ])
            
            # Extreme funding (> 2 std)
            df = df.with_columns([
                pl.when(pl.col("funding_rate_zscore").abs() > 2)
                .then(pl.col("funding_rate_zscore").sign())
                .otherwise(pl.lit(0)).alias("funding_rate_extreme")
            ])
            
            # Direction change
            df = df.with_columns([
                pl.when(
                    (pl.col("funding_rate") > 0) & (pl.col("funding_rate").shift(1) < 0)
                ).then(pl.lit(1))
                .when(
                    (pl.col("funding_rate") < 0) & (pl.col("funding_rate").shift(1) > 0)
                ).then(pl.lit(-1))
                .otherwise(pl.lit(0)).alias("funding_rate_direction")
            ])
            
            # Cumulative funding
            df = df.with_columns([
                pl.col("funding_rate").rolling_sum(window_size=max(1, bars_per_8h)).alias("cumulative_funding_8"),
                pl.col("funding_rate").rolling_sum(window_size=max(1, bars_per_24h)).alias("cumulative_funding_24"),
            ])
        else:
            # Fill with zeros if not available
            for col in ["funding_rate", "funding_rate_ma_8", "funding_rate_ma_24",
                       "funding_rate_zscore", "funding_rate_extreme", "funding_rate_direction",
                       "cumulative_funding_8", "cumulative_funding_24"]:
                df = df.with_columns([pl.lit(0.0).alias(col)])
        
        # === Premium Index Features ===
        if "mark_price" in df.columns:
            df = df.with_columns([
                ((pl.col("close") - pl.col("mark_price")) / pl.col("mark_price")).alias("premium_index")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("premium_index")])
        
        bars_per_4h = max(1, 4 * 60 // self.timeframe_minutes)
        df = df.with_columns([
            pl.col("premium_index").rolling_mean(window_size=bars_per_4h).alias("premium_index_ma"),
        ])
        
        if "funding_rate" in df.columns:
            df = df.with_columns([
                (pl.col("premium_index") - pl.col("funding_rate") * 3).alias("premium_vs_funding")
            ])
        else:
            df = df.with_columns([pl.col("premium_index").alias("premium_vs_funding")])
        
        # Basis
        df = df.with_columns([
            pl.col("premium_index").alias("basis"),
            pl.col("premium_index").rolling_mean(window_size=bars_per_4h).alias("basis_ma"),
        ])
        
        df = df.with_columns([
            ((pl.col("basis") - pl.col("basis").rolling_mean(window_size=max(1, 24 * 60 // self.timeframe_minutes))) /
             (pl.col("basis").rolling_std(window_size=max(1, 24 * 60 // self.timeframe_minutes)) + 1e-10)).alias("basis_zscore")
        ])
        
        # === Open Interest Features ===
        if "sum_open_interest" in df.columns:
            bars_per_24h = max(1, 24 * 60 // self.timeframe_minutes)
            
            df = df.with_columns([
                (pl.col("sum_open_interest") - pl.col("sum_open_interest").shift(1)).alias("oi_change"),
                (pl.col("sum_open_interest") / pl.col("sum_open_interest").shift(1) - 1).alias("oi_change_pct"),
                pl.col("sum_open_interest").rolling_mean(window_size=bars_per_24h).alias("oi_ma_24"),
            ])
            
            df = df.with_columns([
                (pl.col("sum_open_interest") / pl.col("oi_ma_24") - 1).alias("oi_vs_ma"),
                ((pl.col("sum_open_interest") - pl.col("oi_ma_24")) /
                 (pl.col("sum_open_interest").rolling_std(window_size=bars_per_24h) + 1e-10)).alias("oi_zscore"),
                (pl.col("sum_open_interest") / pl.col("sum_open_interest").shift(12) - 1).alias("oi_momentum"),
            ])
            
            # OI + Price divergence
            price_change = pl.col("close") / pl.col("close").shift(12) - 1
            oi_change = pl.col("sum_open_interest") / pl.col("sum_open_interest").shift(12) - 1
            
            df = df.with_columns([
                (price_change - oi_change).alias("oi_price_divergence"),
                pl.when(
                    ((price_change > 0) & (oi_change > 0)) |
                    ((price_change < 0) & (oi_change < 0))
                ).then(pl.lit(1))  # Same direction = strong
                .otherwise(pl.lit(-1)).alias("oi_price_trend")
            ])
        else:
            for col in ["oi_change", "oi_change_pct", "oi_ma_24", "oi_vs_ma", 
                       "oi_zscore", "oi_momentum", "oi_price_divergence", "oi_price_trend"]:
                df = df.with_columns([pl.lit(0.0).alias(col)])
        
        # === Long/Short Ratio Features ===
        bars_per_8h = max(1, 8 * 60 // self.timeframe_minutes)
        
        if "sum_toptrader_long_short_ratio" in df.columns:
            df = df.with_columns([
                pl.col("sum_toptrader_long_short_ratio").rolling_mean(window_size=bars_per_8h).alias("top_ratio_ma_8"),
            ])
            
            df = df.with_columns([
                ((pl.col("sum_toptrader_long_short_ratio") - pl.col("top_ratio_ma_8")) /
                 (pl.col("sum_toptrader_long_short_ratio").rolling_std(window_size=bars_per_8h) + 1e-10)).alias("top_ratio_zscore"),
            ])
            
            df = df.with_columns([
                pl.when(pl.col("top_ratio_zscore").abs() > 1.5)
                .then(pl.col("top_ratio_zscore").sign())
                .otherwise(pl.lit(0)).alias("top_ratio_extreme"),
                (pl.col("sum_toptrader_long_short_ratio") / 
                 pl.col("sum_toptrader_long_short_ratio").shift(12) - 1).alias("top_ratio_momentum"),
            ])
        else:
            for col in ["top_ratio_ma_8", "top_ratio_zscore", "top_ratio_extreme", "top_ratio_momentum"]:
                df = df.with_columns([pl.lit(0.0).alias(col)])
        
        if "count_toptrader_long_short_ratio" in df.columns:
            df = df.with_columns([
                pl.col("count_toptrader_long_short_ratio").rolling_mean(window_size=bars_per_8h).alias("account_ratio_ma_8"),
            ])
            
            df = df.with_columns([
                ((pl.col("count_toptrader_long_short_ratio") - pl.col("account_ratio_ma_8")) /
                 (pl.col("count_toptrader_long_short_ratio").rolling_std(window_size=bars_per_8h) + 1e-10)).alias("account_ratio_zscore"),
            ])
            
            df = df.with_columns([
                pl.when(pl.col("account_ratio_zscore").abs() > 1.5)
                .then(pl.col("account_ratio_zscore").sign())
                .otherwise(pl.lit(0)).alias("account_ratio_extreme"),
            ])
        else:
            for col in ["account_ratio_ma_8", "account_ratio_zscore", "account_ratio_extreme"]:
                df = df.with_columns([pl.lit(0.0).alias(col)])
        
        # Ratio divergence
        if "sum_toptrader_long_short_ratio" in df.columns and "count_toptrader_long_short_ratio" in df.columns:
            df = df.with_columns([
                (pl.col("sum_toptrader_long_short_ratio") - 
                 pl.col("count_toptrader_long_short_ratio")).alias("ratio_divergence")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("ratio_divergence")])
        
        # === Composite Sentiment Indicators ===
        # Crowd sentiment: average of various sentiment z-scores
        sentiment_cols = ["funding_rate_zscore", "top_ratio_zscore", "account_ratio_zscore"]
        available_sentiment = [c for c in sentiment_cols if c in df.columns]
        
        if available_sentiment:
            # Simple average
            df = df.with_columns([
                sum([pl.col(c) for c in available_sentiment]).truediv(len(available_sentiment)).alias("crowd_sentiment")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("crowd_sentiment")])
        
        # Overall sentiment score
        df = df.with_columns([
            pl.col("crowd_sentiment").alias("sentiment_score"),
            pl.when(pl.col("crowd_sentiment").abs() > 1.5)
            .then(pl.lit(1))
            .otherwise(pl.lit(0)).alias("sentiment_extreme"),
            (pl.col("crowd_sentiment") - pl.col("crowd_sentiment").shift(6)).alias("sentiment_momentum"),
        ])
        
        # Contrarian signal: extreme sentiment suggests reversal
        df = df.with_columns([
            pl.when(pl.col("crowd_sentiment") > 1.5)
            .then(pl.lit(-1))  # Extreme bullish sentiment -> sell signal
            .when(pl.col("crowd_sentiment") < -1.5)
            .then(pl.lit(1))  # Extreme bearish sentiment -> buy signal
            .otherwise(pl.lit(0)).alias("contrarian_signal")
        ])
        
        # === Price Context Features ===
        for w in [5, 10, 20]:
            df = df.with_columns([
                (pl.col("close") / pl.col("close").shift(w) - 1).alias(f"ret_{w}"),
                pl.col("close").pct_change().rolling_std(window_size=w).alias(f"vol_{w}"),
            ])
        
        # Sentiment-price alignment
        ret_20 = pl.col("ret_20")
        
        if "crowd_sentiment" in df.columns:
            df = df.with_columns([
                pl.when(
                    ((ret_20 > 0) & (pl.col("crowd_sentiment") > 0)) |
                    ((ret_20 < 0) & (pl.col("crowd_sentiment") < 0))
                ).then(pl.lit(1))
                .otherwise(pl.lit(-1)).alias("sentiment_price_alignment")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("sentiment_price_alignment")])
        
        if "funding_rate" in df.columns:
            df = df.with_columns([
                pl.when(
                    ((ret_20 > 0) & (pl.col("funding_rate") > 0)) |
                    ((ret_20 < 0) & (pl.col("funding_rate") < 0))
                ).then(pl.lit(1))
                .otherwise(pl.lit(-1)).alias("funding_price_alignment")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("funding_price_alignment")])
        
        if "oi_change" in df.columns:
            df = df.with_columns([
                pl.when(
                    ((ret_20 > 0) & (pl.col("oi_change") > 0)) |
                    ((ret_20 < 0) & (pl.col("oi_change") < 0))
                ).then(pl.lit(1))
                .otherwise(pl.lit(-1)).alias("oi_price_alignment")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("oi_price_alignment")])
        
        return df
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Any:
        """Train XGBoost with focus on sentiment features."""
        
        # Encode labels
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        n_classes = len(self.label_encoder.classes_)
        
        model = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.3,
            reg_lambda=1.5,
            gamma=0.05,
            random_state=self.random_state,
            objective='multi:softprob',
            num_class=n_classes,
            eval_metric='mlogloss',
            early_stopping_rounds=25,
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


def run_strategy(timeframe: str = "15min"):
    """Run the sentiment fusion strategy."""
    strategy = SentimentFusionStrategy(timeframe=timeframe)
    return strategy.run(verbose=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sentiment Fusion Strategy")
    parser.add_argument("--timeframe", type=str, default="15min",
                        choices=["5min", "15min", "1hr"])
    
    args = parser.parse_args()
    
    result = run_strategy(args.timeframe)
    print(f"\nFinal Sharpe Ratio: {result.sharpe_ratio:.4f}")
