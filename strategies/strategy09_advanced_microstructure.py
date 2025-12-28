"""
Strategy 09: Advanced Microstructure - VPIN, Toxicity, Kyle's Lambda
================================================================================
Deep microstructure analysis with information-based metrics that capture
informed trading activity and market toxicity.

Key features:
- VPIN (Volume-synchronized Probability of Informed Trading)
- Kyle's Lambda (price impact measure)
- Amihud illiquidity ratio
- Trade toxicity indicators
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from strategies.strategy_base import StrategyBase, StrategyResult


class AdvancedMicrostructureStrategy(StrategyBase):
    """
    Advanced microstructure strategy with VPIN and toxicity metrics.
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        vpin_buckets: int = 50,
        **kwargs
    ):
        super().__init__(symbol, period, timeframe, **kwargs)
        self.vpin_buckets = vpin_buckets
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
    
    def get_name(self) -> str:
        return f"Advanced_Microstructure_{self.timeframe}"
    
    def get_feature_columns(self) -> List[str]:
        return [
            # VPIN features
            'vpin', 'vpin_ma_5', 'vpin_ma_10', 'vpin_zscore',
            'vpin_change', 'vpin_regime',
            
            # Volume imbalance
            'volume_imbalance', 'volume_imbalance_ma_5', 'volume_imbalance_ma_10',
            'cumulative_imbalance_5', 'cumulative_imbalance_10',
            'imbalance_momentum',
            
            # Trade features
            'trade_intensity', 'trade_intensity_ma_5',
            'avg_trade_size', 'avg_trade_size_zscore',
            
            # Kyle's Lambda
            'kyle_lambda', 'kyle_lambda_ma_5', 'kyle_lambda_zscore',
            'price_impact_ratio',
            
            # Amihud
            'amihud', 'amihud_ma_5', 'amihud_zscore',
            
            # Toxicity
            'toxicity_score', 'adverse_selection',
            'informed_intensity',
            
            # Spread proxies
            'range_volume_ratio', 'realized_spread_proxy',
            
            # Basic momentum
            'ret_1', 'ret_3', 'ret_5',
            'vol_5', 'vol_10',
            
            # Cross features
            'vpin_vol_interaction', 'imbalance_momentum_interaction',
        ]
    
    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create advanced microstructure features."""
        
        # === BASIC PRICE FEATURES ===
        for period in [1, 3, 5, 10]:
            df = df.with_columns([
                pl.col("close").pct_change(period).alias(f"ret_{period}")
            ])
        
        for period in [5, 10]:
            df = df.with_columns([
                pl.col("ret_1").rolling_std(period).alias(f"vol_{period}")
            ])
        
        # === VOLUME IMBALANCE ===
        if "taker_buy_volume" in df.columns:
            taker_sell = pl.col("volume") - pl.col("taker_buy_volume")
            df = df.with_columns([
                ((pl.col("taker_buy_volume") - taker_sell) / 
                 (pl.col("volume") + 1e-10)).alias("volume_imbalance")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("volume_imbalance")])
        
        # Volume imbalance moving averages
        for period in [5, 10]:
            df = df.with_columns([
                pl.col("volume_imbalance").rolling_mean(period).alias(f"volume_imbalance_ma_{period}")
            ])
        
        # Cumulative imbalance
        for period in [5, 10]:
            df = df.with_columns([
                pl.col("volume_imbalance").rolling_sum(period).alias(f"cumulative_imbalance_{period}")
            ])
        
        df = df.with_columns([
            (pl.col("volume_imbalance") - pl.col("volume_imbalance").shift(3)).alias("imbalance_momentum")
        ])
        
        # === VPIN ===
        df = df.with_columns([
            pl.col("volume_imbalance").abs().rolling_mean(self.vpin_buckets).alias("vpin")
        ])
        
        for period in [5, 10]:
            df = df.with_columns([
                pl.col("vpin").rolling_mean(period).alias(f"vpin_ma_{period}")
            ])
        
        df = df.with_columns([
            ((pl.col("vpin") - pl.col("vpin").rolling_mean(20)) / 
             (pl.col("vpin").rolling_std(20) + 1e-10)).alias("vpin_zscore"),
            pl.col("vpin").pct_change(1).alias("vpin_change"),
        ])
        
        df = df.with_columns([
            pl.when(pl.col("vpin") > pl.col("vpin").rolling_quantile(0.8, window_size=50))
            .then(1).otherwise(0).alias("vpin_regime")
        ])
        
        # === TRADE FEATURES ===
        if "count" in df.columns:
            trade_col = "count"
        else:
            df = df.with_columns([pl.lit(1.0).alias("_trade_count")])
            trade_col = "_trade_count"
        
        df = df.with_columns([
            (pl.col(trade_col) / (pl.col(trade_col).rolling_mean(20) + 1e-10)).alias("trade_intensity")
        ])
        df = df.with_columns([
            pl.col("trade_intensity").rolling_mean(5).alias("trade_intensity_ma_5")
        ])
        
        df = df.with_columns([
            (pl.col("volume") / (pl.col(trade_col) + 1e-10)).alias("avg_trade_size")
        ])
        df = df.with_columns([
            ((pl.col("avg_trade_size") - pl.col("avg_trade_size").rolling_mean(20)) /
             (pl.col("avg_trade_size").rolling_std(20) + 1e-10)).alias("avg_trade_size_zscore")
        ])
        
        # === KYLE'S LAMBDA ===
        df = df.with_columns([
            (pl.col("ret_1").abs() / (pl.col("volume") + 1e-10) * 1e6).alias("kyle_lambda")
        ])
        df = df.with_columns([
            pl.col("kyle_lambda").rolling_mean(5).alias("kyle_lambda_ma_5"),
            ((pl.col("kyle_lambda") - pl.col("kyle_lambda").rolling_mean(20)) /
             (pl.col("kyle_lambda").rolling_std(20) + 1e-10)).alias("kyle_lambda_zscore"),
            (pl.col("kyle_lambda") / (pl.col("kyle_lambda").rolling_mean(20) + 1e-10)).alias("price_impact_ratio")
        ])
        
        # === AMIHUD ===
        df = df.with_columns([
            (pl.col("ret_1").abs() / (pl.col("volume") * pl.col("close") + 1e-10) * 1e9).alias("amihud")
        ])
        df = df.with_columns([
            pl.col("amihud").rolling_mean(5).alias("amihud_ma_5"),
            ((pl.col("amihud") - pl.col("amihud").rolling_mean(20)) /
             (pl.col("amihud").rolling_std(20) + 1e-10)).alias("amihud_zscore")
        ])
        
        # === TOXICITY ===
        df = df.with_columns([
            (pl.col("vpin") * pl.col("kyle_lambda") / 
             (pl.col("kyle_lambda").rolling_mean(20) + 1e-10)).alias("toxicity_score"),
            (pl.col("volume_imbalance") * pl.col("ret_1")).alias("adverse_selection"),
            (pl.col("vpin") * pl.col("trade_intensity")).alias("informed_intensity"),
        ])
        
        # === SPREAD PROXIES ===
        df = df.with_columns([
            ((pl.col("high") - pl.col("low")) / (pl.col("volume") + 1e-10) * 1e6).alias("range_volume_ratio"),
            ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-10) - 0.5).alias("realized_spread_proxy")
        ])
        
        # === CROSS FEATURES ===
        df = df.with_columns([
            (pl.col("vpin") * pl.col("vol_5")).alias("vpin_vol_interaction"),
            (pl.col("volume_imbalance") * pl.col("ret_3")).alias("imbalance_momentum_interaction"),
        ])
        
        # Clip extreme z-scores
        for col in ['vpin_zscore', 'kyle_lambda_zscore', 'amihud_zscore', 'avg_trade_size_zscore']:
            if col in df.columns:
                df = df.with_columns([pl.col(col).clip(-5, 5).alias(col)])
        
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
        
        y_train_enc = self.label_encoder.fit_transform(y_train)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if HAS_LGB:
            model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.2,
                reg_lambda=1.0,
                min_child_samples=20,
                random_state=42,
                verbose=-1,
                importance_type='gain',
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        
        if sample_weights is not None:
            model.fit(X_train_scaled, y_train_enc, sample_weight=sample_weights)
        else:
            model.fit(X_train_scaled, y_train_enc)
        
        return model
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with the trained model."""
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred)
        return y_pred, y_prob
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        importance = {}
        if hasattr(self.model, 'feature_importances_'):
            for i, feat in enumerate(self.feature_names[:len(self.model.feature_importances_)]):
                importance[feat] = float(self.model.feature_importances_[i])
        return importance


def run_strategy(timeframe: str = "15min") -> StrategyResult:
    """Run the advanced microstructure strategy."""
    strategy = AdvancedMicrostructureStrategy(
        symbol="BTCUSDT",
        period="2025_11",
        timeframe=timeframe,
    )
    return strategy.run(verbose=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Microstructure Strategy")
    parser.add_argument("--timeframe", type=str, default="15min", 
                        choices=["5min", "15min", "1hr"])
    args = parser.parse_args()
    result = run_strategy(args.timeframe)
    print(f"\nFinal Sharpe Ratio: {result.sharpe_ratio:.4f}")
