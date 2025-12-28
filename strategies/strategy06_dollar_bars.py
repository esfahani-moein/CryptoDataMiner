"""
Strategy 06: Dollar Bars
========================

Uses dollar bars (volume-based sampling) instead of time bars.
Dollar bars sample based on cumulative dollar volume, which:
- Normalizes for varying market activity
- Provides more bars during high-activity periods
- Results in more stable statistical properties

Reference: Lopez de Prado, "Advances in Financial Machine Learning"
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_base import StrategyBase, StrategyResult


class DollarBarsStrategy(StrategyBase):
    """
    Strategy using dollar bars instead of time bars.
    
    Dollar bars are created when cumulative dollar volume exceeds a threshold.
    This results in:
    - More bars during high volatility/volume periods
    - Fewer bars during quiet periods
    - More homogeneous statistical properties
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "dollar",  # Ignored, always uses dollar bars
        dollar_threshold: Optional[float] = None,  # Auto-calculated if None
        target_bars_per_day: int = 50,  # Used to calculate threshold
        **kwargs
    ):
        # Don't pass timeframe to parent, we use dollar bars
        super().__init__(symbol, period, timeframe="dollar", **kwargs)
        self.dollar_threshold = dollar_threshold
        self.target_bars_per_day = target_bars_per_day
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def get_name(self) -> str:
        return f"Dollar_Bars_{self.target_bars_per_day}pd"
    
    @property
    def bars_per_day(self) -> int:
        return self.target_bars_per_day
    
    def get_feature_columns(self) -> List[str]:
        """Features for dollar bar strategy."""
        features = []
        
        # Price features
        for w in [1, 3, 5, 10, 20, 50]:
            features.append(f"ret_{w}")
        
        # Bar-specific features
        features.extend([
            "dollar_volume",
            "bar_duration_seconds",
            "bar_duration_normalized",
            "trades_per_bar",
            "avg_trade_size",
            "bar_range_pct",
        ])
        
        # Rolling bar statistics
        for w in [5, 10, 20]:
            features.extend([
                f"dollar_volume_ma_{w}",
                f"bar_duration_ma_{w}",
                f"trades_ma_{w}",
            ])
        
        # Volatility (normalized by bar count)
        for w in [5, 10, 20, 50]:
            features.extend([
                f"vol_std_{w}",
                f"vol_parkinson_{w}",
            ])
        
        # Momentum
        for p in [7, 14, 21]:
            features.append(f"rsi_{p}")
        
        # Trend
        for w in [10, 20, 50]:
            features.extend([f"sma_{w}", f"ema_{w}", f"price_vs_sma_{w}"])
        
        # MACD
        features.extend(["macd", "macd_signal", "macd_hist"])
        
        # Order flow
        features.extend([
            "buy_volume_pct",
            "sell_volume_pct",
            "volume_imbalance",
            "trade_imbalance",
        ])
        
        # Time-of-day encoding (for crypto, 24h market)
        features.extend([
            "hour_sin",
            "hour_cos",
            "is_asia_session",
            "is_europe_session",
            "is_us_session",
        ])
        
        return features
    
    def load_data(self) -> pl.DataFrame:
        """Load trades and create dollar bars."""
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
        
        # Calculate dollar threshold if not specified
        if self.dollar_threshold is None:
            self.dollar_threshold = self._calculate_threshold(trades_df)
            print(f"    Calculated dollar threshold: ${self.dollar_threshold:,.2f}")
        
        # Create dollar bars
        print(f"  Creating dollar bars...")
        dollar_bars = self._create_dollar_bars(trades_df)
        print(f"    Created {len(dollar_bars):,} dollar bars")
        
        return dollar_bars
    
    def _calculate_threshold(self, trades_df: pl.DataFrame) -> float:
        """Calculate dollar threshold to achieve target bars per day."""
        # Total dollar volume
        total_dollar_volume = trades_df["quote_qty"].sum()
        
        # Number of days in data
        time_range_ms = trades_df["time"].max() - trades_df["time"].min()
        n_days = time_range_ms / (24 * 60 * 60 * 1000)
        
        # Target total bars
        target_total_bars = n_days * self.target_bars_per_day
        
        # Threshold
        threshold = total_dollar_volume / target_total_bars
        
        return threshold
    
    def _create_dollar_bars(self, trades_df: pl.DataFrame) -> pl.DataFrame:
        """Create dollar bars from tick data."""
        # Ensure we have is_buyer_maker
        if "is_buyer_maker" not in trades_df.columns:
            trades_df = trades_df.with_columns([
                (pl.col("price") >= pl.col("price").shift(1)).alias("is_buyer_maker")
            ])
        
        # Calculate signed volume
        trades_df = trades_df.with_columns([
            pl.when(pl.col("is_buyer_maker"))
            .then(-pl.col("qty"))  # Seller initiated
            .otherwise(pl.col("qty"))  # Buyer initiated
            .alias("signed_qty")
        ])
        
        # Convert to numpy for efficient processing
        times = trades_df["time"].to_numpy()
        prices = trades_df["price"].to_numpy()
        quantities = trades_df["qty"].to_numpy()
        quote_qty = trades_df["quote_qty"].to_numpy()
        signed_qty = trades_df["signed_qty"].to_numpy()
        
        # Create dollar bars
        bars = []
        cumulative_dollar = 0.0
        bar_start_idx = 0
        
        for i in range(len(trades_df)):
            cumulative_dollar += quote_qty[i]
            
            if cumulative_dollar >= self.dollar_threshold:
                # Create bar
                bar_prices = prices[bar_start_idx:i+1]
                bar_qty = quantities[bar_start_idx:i+1]
                bar_signed_qty = signed_qty[bar_start_idx:i+1]
                bar_quote = quote_qty[bar_start_idx:i+1]
                
                bar = {
                    'timestamp': times[bar_start_idx],
                    'end_time': times[i],
                    'open': bar_prices[0],
                    'high': bar_prices.max(),
                    'low': bar_prices.min(),
                    'close': bar_prices[-1],
                    'volume': bar_qty.sum(),
                    'dollar_volume': bar_quote.sum(),
                    'trade_count': len(bar_prices),
                    'buy_volume': bar_signed_qty[bar_signed_qty > 0].sum(),
                    'sell_volume': -bar_signed_qty[bar_signed_qty < 0].sum(),
                    'bar_duration_ms': times[i] - times[bar_start_idx],
                }
                bars.append(bar)
                
                # Reset
                cumulative_dollar = 0.0
                bar_start_idx = i + 1
        
        # Create DataFrame
        if not bars:
            raise ValueError("No dollar bars created. Try lowering threshold.")
        
        dollar_bars = pl.DataFrame(bars)
        
        # Add derived features
        dollar_bars = dollar_bars.with_columns([
            (pl.col("bar_duration_ms") / 1000).alias("bar_duration_seconds"),
            (pl.col("volume") / pl.col("trade_count")).alias("avg_trade_size"),
            ((pl.col("high") - pl.col("low")) / pl.col("open") * 100).alias("bar_range_pct"),
            (pl.col("buy_volume") / (pl.col("volume") + 1e-10)).alias("buy_volume_pct"),
            (pl.col("sell_volume") / (pl.col("volume") + 1e-10)).alias("sell_volume_pct"),
            ((pl.col("buy_volume") - pl.col("sell_volume")) / 
             (pl.col("volume") + 1e-10)).alias("volume_imbalance"),
        ])
        
        # Normalize bar duration
        mean_duration = dollar_bars["bar_duration_seconds"].mean()
        dollar_bars = dollar_bars.with_columns([
            (pl.col("bar_duration_seconds") / mean_duration).alias("bar_duration_normalized")
        ])
        
        # Trade imbalance
        mean_trades = dollar_bars["trade_count"].mean()
        dollar_bars = dollar_bars.with_columns([
            (pl.col("trade_count").cast(pl.Float64) / mean_trades - 1).alias("trade_imbalance")
        ])
        
        return dollar_bars
    
    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create features for dollar bars."""
        
        # === Returns ===
        for w in [1, 3, 5, 10, 20, 50]:
            df = df.with_columns([
                (pl.col("close") / pl.col("close").shift(w) - 1).alias(f"ret_{w}")
            ])
        
        # === Rolling bar statistics ===
        for w in [5, 10, 20]:
            df = df.with_columns([
                pl.col("dollar_volume").rolling_mean(window_size=w).alias(f"dollar_volume_ma_{w}"),
                pl.col("bar_duration_seconds").rolling_mean(window_size=w).alias(f"bar_duration_ma_{w}"),
                pl.col("trade_count").cast(pl.Float64).rolling_mean(window_size=w).alias(f"trades_ma_{w}"),
            ])
        
        # === Volatility ===
        for w in [5, 10, 20, 50]:
            df = df.with_columns([
                pl.col("close").pct_change().rolling_std(window_size=w).alias(f"vol_std_{w}"),
                ((pl.col("high") / pl.col("low")).log() ** 2 / (4 * np.log(2)))
                .rolling_mean(window_size=w).sqrt().alias(f"vol_parkinson_{w}"),
            ])
        
        # === RSI ===
        for period in [7, 14, 21]:
            delta = pl.col("close") - pl.col("close").shift(1)
            gain = pl.when(delta > 0).then(delta).otherwise(pl.lit(0.0))
            loss = pl.when(delta < 0).then(-delta).otherwise(pl.lit(0.0))
            
            df = df.with_columns([
                gain.rolling_mean(window_size=period).alias(f"_avg_gain_{period}"),
                loss.rolling_mean(window_size=period).alias(f"_avg_loss_{period}"),
            ])
            df = df.with_columns([
                (100 - 100 / (1 + pl.col(f"_avg_gain_{period}") / 
                              (pl.col(f"_avg_loss_{period}") + 1e-10))).alias(f"rsi_{period}")
            ])
        
        # === Moving Averages ===
        for w in [10, 20, 50]:
            df = df.with_columns([
                pl.col("close").rolling_mean(window_size=w).alias(f"sma_{w}"),
                pl.col("close").ewm_mean(span=w, adjust=False).alias(f"ema_{w}"),
            ])
            df = df.with_columns([
                (pl.col("close") / pl.col(f"sma_{w}") - 1).alias(f"price_vs_sma_{w}"),
            ])
        
        # === MACD ===
        ema_12 = pl.col("close").ewm_mean(span=12, adjust=False)
        ema_26 = pl.col("close").ewm_mean(span=26, adjust=False)
        
        df = df.with_columns([
            (ema_12 - ema_26).alias("macd")
        ])
        df = df.with_columns([
            pl.col("macd").ewm_mean(span=9, adjust=False).alias("macd_signal")
        ])
        df = df.with_columns([
            (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist")
        ])
        
        # === Time-of-day features ===
        # Convert timestamp to hour
        df = df.with_columns([
            (pl.col("timestamp") % (24 * 60 * 60 * 1000) / (60 * 60 * 1000)).alias("hour_float")
        ])
        
        df = df.with_columns([
            (pl.col("hour_float") * 2 * np.pi / 24).sin().alias("hour_sin"),
            (pl.col("hour_float") * 2 * np.pi / 24).cos().alias("hour_cos"),
        ])
        
        # Trading sessions (UTC times)
        df = df.with_columns([
            pl.when((pl.col("hour_float") >= 0) & (pl.col("hour_float") < 8))
            .then(pl.lit(1))
            .otherwise(pl.lit(0)).alias("is_asia_session"),
            
            pl.when((pl.col("hour_float") >= 7) & (pl.col("hour_float") < 16))
            .then(pl.lit(1))
            .otherwise(pl.lit(0)).alias("is_europe_session"),
            
            pl.when((pl.col("hour_float") >= 13) & (pl.col("hour_float") < 22))
            .then(pl.lit(1))
            .otherwise(pl.lit(0)).alias("is_us_session"),
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
        """Train XGBoost on dollar bar features."""
        
        # Encode labels
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        n_classes = len(self.label_encoder.classes_)
        
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.5,
            reg_lambda=2.0,
            gamma=0.1,
            random_state=self.random_state,
            objective='multi:softprob',
            num_class=n_classes,
            eval_metric='mlogloss',
            early_stopping_rounds=30,
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
    
    def run(self, verbose: bool = True) -> StrategyResult:
        """Execute strategy with dollar bars."""
        result = super().run(verbose=verbose)
        result.bar_type = "dollar"
        return result


def run_strategy(target_bars_per_day: int = 50):
    """Run the dollar bars strategy."""
    strategy = DollarBarsStrategy(target_bars_per_day=target_bars_per_day)
    return strategy.run(verbose=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dollar Bars Strategy")
    parser.add_argument("--bars-per-day", type=int, default=50,
                        help="Target number of bars per day")
    
    args = parser.parse_args()
    
    result = run_strategy(args.bars_per_day)
    print(f"\nFinal Sharpe Ratio: {result.sharpe_ratio:.4f}")
