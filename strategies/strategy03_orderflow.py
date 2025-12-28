"""
Strategy 03: Order Flow & Microstructure
=========================================

Focus on order flow imbalance and market microstructure features.
Uses trade-level data to extract:
- Buy/Sell pressure
- Volume imbalance
- Trade intensity
- Price impact
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_base import StrategyBase


class OrderFlowStrategy(StrategyBase):
    """
    Order flow and microstructure-based strategy.
    
    Features:
    - Trade imbalance (buy vs sell volume)
    - Volume-weighted order flow
    - Trade intensity and clustering
    - Kyle's Lambda (price impact)
    - VPIN (Volume-Synchronized Probability of Informed Trading)
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
        self.trades_df = None  # Store raw trades for flow analysis
        
    def get_name(self) -> str:
        return f"OrderFlow_Microstructure_{self.timeframe}"
    
    def get_feature_columns(self) -> List[str]:
        """Order flow and microstructure features."""
        features = []
        
        # Basic trade imbalance
        features.extend([
            "buy_volume",
            "sell_volume",
            "volume_imbalance",
            "volume_imbalance_pct",
            "trade_imbalance",
            "trade_imbalance_pct",
        ])
        
        # Rolling imbalances
        for w in [3, 5, 10, 20]:
            features.extend([
                f"volume_imbalance_ma_{w}",
                f"trade_imbalance_ma_{w}",
                f"cumulative_volume_delta_{w}",
            ])
        
        # Trade intensity
        features.extend([
            "trade_count",
            "avg_trade_size",
            "trade_size_std",
            "large_trade_ratio",
            "small_trade_ratio",
        ])
        
        for w in [5, 10, 20]:
            features.extend([
                f"trade_intensity_{w}",
                f"trade_intensity_change_{w}",
            ])
        
        # VWAP features
        features.extend([
            "vwap",
            "price_vs_vwap",
            "vwap_slope",
        ])
        
        for w in [5, 10, 20]:
            features.append(f"vwap_distance_{w}")
        
        # Price impact (Kyle's Lambda approximation)
        features.extend([
            "price_impact",
            "price_impact_buy",
            "price_impact_sell",
            "kyle_lambda",
        ])
        
        for w in [5, 10]:
            features.append(f"kyle_lambda_ma_{w}")
        
        # VPIN (simplified)
        features.extend([
            "vpin",
            "vpin_ma_5",
            "vpin_ma_10",
        ])
        
        # Order flow toxicity
        features.extend([
            "toxicity_score",
            "informed_trading_prob",
        ])
        
        # Tick features
        features.extend([
            "tick_direction",
            "tick_run_length",
            "uptick_ratio",
        ])
        
        for w in [5, 10, 20]:
            features.append(f"tick_imbalance_{w}")
        
        # Aggression metrics
        features.extend([
            "buy_aggression",
            "sell_aggression",
            "net_aggression",
        ])
        
        # Price pressure
        for w in [5, 10, 20]:
            features.extend([
                f"buy_pressure_{w}",
                f"sell_pressure_{w}",
                f"net_pressure_{w}",
            ])
        
        return features
    
    def load_data(self) -> pl.DataFrame:
        """Load trades and compute order flow features before aggregation."""
        from quant_features.data_loader import load_trades
        
        print(f"  Loading trades data...")
        trades_df = load_trades(
            self.data_path, 
            symbol=self.symbol,
            start_year=self.start_year,
            start_month=self.start_month,
            end_year=self.end_year,
            end_month=self.end_month
        )
        print(f"    Loaded {len(trades_df):,} trades")
        
        # Store for later use
        self.trades_df = trades_df
        
        # Add trade direction features
        trades_df = self._add_trade_features(trades_df)
        
        # Aggregate to bars with order flow features
        print(f"  Aggregating to {self.timeframe} bars with order flow...")
        ohlcv = self._aggregate_with_orderflow(trades_df)
        print(f"    Created {len(ohlcv):,} bars")
        
        return ohlcv
    
    def _add_trade_features(self, trades_df: pl.DataFrame) -> pl.DataFrame:
        """Add features to individual trades."""
        # Ensure we have is_buyer_maker column
        if "is_buyer_maker" not in trades_df.columns:
            # Infer from price movement
            trades_df = trades_df.with_columns([
                (pl.col("price") >= pl.col("price").shift(1)).alias("is_buyer_maker")
            ])
        
        # Trade direction: 1 for buy (taker), -1 for sell (taker)
        # is_buyer_maker=True means seller was taker (initiated trade)
        trades_df = trades_df.with_columns([
            pl.when(pl.col("is_buyer_maker"))
            .then(pl.lit(-1))  # Sell initiated
            .otherwise(pl.lit(1))  # Buy initiated
            .alias("trade_direction")
        ])
        
        # Signed volume
        trades_df = trades_df.with_columns([
            (pl.col("qty") * pl.col("trade_direction")).alias("signed_volume")
        ])
        
        return trades_df
    
    def _aggregate_with_orderflow(self, trades_df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate trades to bars with order flow metrics."""
        # Create time bucket
        interval_ms = self.timeframe_minutes * 60 * 1000
        
        trades_df = trades_df.with_columns([
            (pl.col("time") // interval_ms * interval_ms).alias("timestamp")
        ])
        
        # Aggregate
        ohlcv = trades_df.group_by("timestamp").agg([
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("qty").sum().alias("volume"),
            pl.col("quote_qty").sum().alias("quote_volume"),
            pl.col("id").count().alias("trade_count"),
            
            # Buy/Sell volume
            pl.when(pl.col("trade_direction") == 1)
            .then(pl.col("qty"))
            .otherwise(pl.lit(0.0))
            .sum()
            .alias("buy_volume"),
            
            pl.when(pl.col("trade_direction") == -1)
            .then(pl.col("qty"))
            .otherwise(pl.lit(0.0))
            .sum()
            .alias("sell_volume"),
            
            # Buy/Sell count
            (pl.col("trade_direction") == 1).sum().alias("buy_count"),
            (pl.col("trade_direction") == -1).sum().alias("sell_count"),
            
            # Signed volume (CVD)
            pl.col("signed_volume").sum().alias("signed_volume"),
            
            # Trade size stats
            pl.col("qty").mean().alias("avg_trade_size"),
            pl.col("qty").std().alias("trade_size_std"),
            pl.col("qty").max().alias("max_trade_size"),
            
            # Large trade detection (> 2x average)
            (pl.col("qty") > pl.col("qty").mean() * 2).sum().alias("large_trade_count"),
            (pl.col("qty") < pl.col("qty").mean() * 0.5).sum().alias("small_trade_count"),
            
            # VWAP
            (pl.col("price") * pl.col("qty")).sum().alias("_price_volume"),
            
            # Price range for impact calculation
            (pl.col("price").last() - pl.col("price").first()).alias("price_change"),
            
            # Tick direction
            (pl.col("price") > pl.col("price").shift(1)).sum().alias("uptick_count"),
            (pl.col("price") < pl.col("price").shift(1)).sum().alias("downtick_count"),
        ]).sort("timestamp")
        
        # Calculate VWAP
        ohlcv = ohlcv.with_columns([
            (pl.col("_price_volume") / (pl.col("volume") + 1e-10)).alias("vwap")
        ])
        
        # Volume imbalance
        ohlcv = ohlcv.with_columns([
            (pl.col("buy_volume") - pl.col("sell_volume")).alias("volume_imbalance"),
            ((pl.col("buy_volume") - pl.col("sell_volume")) / 
             (pl.col("volume") + 1e-10)).alias("volume_imbalance_pct"),
            (pl.col("buy_count") - pl.col("sell_count")).alias("trade_imbalance"),
            ((pl.col("buy_count") - pl.col("sell_count")).cast(pl.Float64) / 
             (pl.col("trade_count") + 1e-10)).alias("trade_imbalance_pct"),
        ])
        
        # Large/small trade ratios
        ohlcv = ohlcv.with_columns([
            (pl.col("large_trade_count").cast(pl.Float64) / 
             (pl.col("trade_count") + 1e-10)).alias("large_trade_ratio"),
            (pl.col("small_trade_count").cast(pl.Float64) / 
             (pl.col("trade_count") + 1e-10)).alias("small_trade_ratio"),
        ])
        
        # Tick direction metrics
        ohlcv = ohlcv.with_columns([
            pl.when(pl.col("uptick_count") > pl.col("downtick_count"))
            .then(pl.lit(1))
            .when(pl.col("uptick_count") < pl.col("downtick_count"))
            .then(pl.lit(-1))
            .otherwise(pl.lit(0)).alias("tick_direction"),
            
            (pl.col("uptick_count").cast(pl.Float64) / 
             (pl.col("uptick_count") + pl.col("downtick_count") + 1e-10)).alias("uptick_ratio"),
        ])
        
        return ohlcv
    
    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create order flow and microstructure features."""
        
        # === Rolling Imbalances ===
        for w in [3, 5, 10, 20]:
            df = df.with_columns([
                pl.col("volume_imbalance").rolling_mean(window_size=w).alias(f"volume_imbalance_ma_{w}"),
                pl.col("trade_imbalance").rolling_mean(window_size=w).alias(f"trade_imbalance_ma_{w}"),
                pl.col("signed_volume").rolling_sum(window_size=w).alias(f"cumulative_volume_delta_{w}"),
            ])
        
        # === Trade Intensity ===
        for w in [5, 10, 20]:
            df = df.with_columns([
                pl.col("trade_count").rolling_mean(window_size=w).alias(f"trade_intensity_{w}"),
            ])
            df = df.with_columns([
                (pl.col("trade_count") / (pl.col(f"trade_intensity_{w}") + 1e-10) - 1).alias(f"trade_intensity_change_{w}")
            ])
        
        # === VWAP Features ===
        df = df.with_columns([
            (pl.col("close") / pl.col("vwap") - 1).alias("price_vs_vwap"),
            (pl.col("vwap") - pl.col("vwap").shift(1)).alias("vwap_slope"),
        ])
        
        for w in [5, 10, 20]:
            vwap_rolling = (pl.col("quote_volume").rolling_sum(window_size=w) / 
                           (pl.col("volume").rolling_sum(window_size=w) + 1e-10))
            df = df.with_columns([
                (pl.col("close") / vwap_rolling - 1).alias(f"vwap_distance_{w}")
            ])
        
        # === Price Impact (Kyle's Lambda) ===
        # Lambda = price_change / signed_volume (simplified)
        df = df.with_columns([
            (pl.col("price_change").abs() / (pl.col("volume") + 1e-10)).alias("price_impact"),
            
            # Separate buy/sell impact
            pl.when(pl.col("signed_volume") > 0)
            .then(pl.col("price_change") / (pl.col("signed_volume").abs() + 1e-10))
            .otherwise(pl.lit(0.0)).alias("price_impact_buy"),
            
            pl.when(pl.col("signed_volume") < 0)
            .then(-pl.col("price_change") / (pl.col("signed_volume").abs() + 1e-10))
            .otherwise(pl.lit(0.0)).alias("price_impact_sell"),
        ])
        
        # Kyle's Lambda: regression coefficient of price on volume
        df = df.with_columns([
            (pl.col("price_change") / (pl.col("signed_volume").abs() + 1e-10)).alias("kyle_lambda")
        ])
        
        for w in [5, 10]:
            df = df.with_columns([
                pl.col("kyle_lambda").rolling_mean(window_size=w).alias(f"kyle_lambda_ma_{w}")
            ])
        
        # === VPIN (Simplified) ===
        # VPIN = |Buy - Sell| / Total over volume buckets
        df = df.with_columns([
            (pl.col("volume_imbalance").abs() / (pl.col("volume") + 1e-10)).alias("vpin")
        ])
        
        df = df.with_columns([
            pl.col("vpin").rolling_mean(window_size=5).alias("vpin_ma_5"),
            pl.col("vpin").rolling_mean(window_size=10).alias("vpin_ma_10"),
        ])
        
        # === Order Flow Toxicity ===
        df = df.with_columns([
            # Toxicity: high imbalance + high price impact
            (pl.col("vpin") * pl.col("price_impact")).alias("toxicity_score"),
            
            # Informed trading probability (higher VPIN = more informed)
            pl.col("vpin").rolling_quantile(quantile=0.5, window_size=50).alias("_vpin_median"),
        ])
        
        df = df.with_columns([
            pl.when(pl.col("vpin") > pl.col("_vpin_median"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0)).alias("informed_trading_prob")
        ])
        
        # === Tick Imbalance ===
        for w in [5, 10, 20]:
            df = df.with_columns([
                (pl.col("uptick_count") - pl.col("downtick_count"))
                .rolling_sum(window_size=w).alias(f"tick_imbalance_{w}")
            ])
        
        # === Tick Run Length ===
        # How many consecutive bars with same tick direction
        tick_change = (pl.col("tick_direction") != pl.col("tick_direction").shift(1)).cast(pl.Int32)
        df = df.with_columns([
            tick_change.alias("_tick_change")
        ])
        df = df.with_columns([
            pl.col("_tick_change").cum_sum().alias("_tick_group")
        ])
        df = df.with_columns([
            pl.lit(1).cum_sum().over("_tick_group").alias("tick_run_length")
        ])
        
        # === Aggression Metrics ===
        # Buy aggression: buy volume when price goes up
        df = df.with_columns([
            pl.when(pl.col("price_change") > 0)
            .then(pl.col("buy_volume") / (pl.col("volume") + 1e-10))
            .otherwise(pl.lit(0.0)).alias("buy_aggression"),
            
            pl.when(pl.col("price_change") < 0)
            .then(pl.col("sell_volume") / (pl.col("volume") + 1e-10))
            .otherwise(pl.lit(0.0)).alias("sell_aggression"),
        ])
        
        df = df.with_columns([
            (pl.col("buy_aggression") - pl.col("sell_aggression")).alias("net_aggression")
        ])
        
        # === Price Pressure ===
        for w in [5, 10, 20]:
            df = df.with_columns([
                (pl.col("buy_volume") * pl.col("price_change").clip(lower_bound=0))
                .rolling_sum(window_size=w).alias(f"buy_pressure_{w}"),
                
                (pl.col("sell_volume") * (-pl.col("price_change")).clip(lower_bound=0))
                .rolling_sum(window_size=w).alias(f"sell_pressure_{w}"),
            ])
            
            df = df.with_columns([
                (pl.col(f"buy_pressure_{w}") - pl.col(f"sell_pressure_{w}")).alias(f"net_pressure_{w}")
            ])
        
        return df
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Any:
        """Train LightGBM model."""
        
        # Encode labels
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        n_classes = len(self.label_encoder.classes_)
        
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=30,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=self.random_state,
            objective='multiclass',
            num_class=n_classes,
            verbosity=-1,
            importance_type='gain',
        )
        
        model.fit(
            X_train_scaled, y_train_enc,
            sample_weight=sample_weights,
            eval_set=[(X_val_scaled, y_val_enc)],
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
    """Run the order flow strategy."""
    strategy = OrderFlowStrategy(timeframe=timeframe)
    return strategy.run(verbose=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Order Flow Strategy")
    parser.add_argument("--timeframe", type=str, default="15min",
                        choices=["5min", "15min", "1hr"])
    
    args = parser.parse_args()
    
    result = run_strategy(args.timeframe)
    print(f"\nFinal Sharpe Ratio: {result.sharpe_ratio:.4f}")
