"""
Strategy 08: Hybrid Ensemble - Best Features from All Strategies
================================================================================
Combines the most predictive features from order flow, momentum, and sentiment
with a sophisticated ensemble approach using binary classification.

Key innovations:
1. Binary classification (Long/Short) - no Hold class
2. Probability-based position sizing  
3. Feature interaction terms
4. Dynamic threshold optimization
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import xgboost as xgb
    import lightgbm as lgb
    HAS_BOOSTING = True
except ImportError:
    HAS_BOOSTING = False

from strategies.strategy_base import StrategyBase, StrategyResult


class HybridEnsembleStrategy(StrategyBase):
    """
    Hybrid ensemble combining best features from all strategies.
    Uses binary classification with probability-based position sizing.
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        ensemble_type: str = "stacking",  # voting, stacking
        use_calibration: bool = True,
        probability_threshold: float = 0.55,
        **kwargs
    ):
        super().__init__(symbol, period, timeframe, **kwargs)
        self.ensemble_type = ensemble_type
        self.use_calibration = use_calibration
        self.probability_threshold = probability_threshold
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.calibrated_model = None
    
    def get_name(self) -> str:
        return f"Hybrid_Ensemble_{self.ensemble_type}_{self.timeframe}"
    
    def get_feature_columns(self) -> List[str]:
        """All features we want to use."""
        features = [
            # Momentum (best from Strategy 01)
            'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20',
            'rsi_7', 'rsi_14', 'rsi_21',
            'macd', 'macd_signal', 'macd_hist',
            
            # Volatility (best from Strategy 02)
            'vol_std_5', 'vol_std_10', 'vol_std_20',
            'vol_ratio_5_20', 'vol_zscore',
            'atr_14', 'atr_ratio',
            
            # Order flow (best from Strategy 03)
            'volume_imbalance', 'trade_imbalance',
            'buy_pressure', 'sell_pressure',
            'trade_intensity', 'vwap_distance',
            
            # Microstructure
            'avg_trade_size', 'large_trade_ratio',
            'taker_buy_ratio', 'quote_volume_ratio',
            
            # Sentiment (best from Strategy 04)
            'sum_open_interest', 'oi_change',
            'sum_toptrader_long_short_ratio',
            
            # Technical
            'sma_cross', 'ema_cross',
            'bb_position', 'bb_width',
            'stoch_k', 'stoch_d',
            
            # Interaction features
            'mom_vol_interaction', 'flow_mom_interaction',
            'rsi_macd_interaction', 'vol_trend_interaction',
            'pressure_imbalance_ratio',
        ]
        return features
    
    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create comprehensive feature set from multiple domains."""
        
        # === MOMENTUM FEATURES ===
        for period in [1, 3, 5, 10, 20, 50]:
            df = df.with_columns([
                (pl.col("close").pct_change(period)).alias(f"ret_{period}")
            ])
        
        # RSI with multiple periods
        for period in [7, 14, 21]:
            delta = pl.col("close").diff()
            gain = pl.when(delta > 0).then(delta).otherwise(0)
            loss = pl.when(delta < 0).then(-delta).otherwise(0)
            
            df = df.with_columns([
                gain.rolling_mean(period).alias(f"_avg_gain_{period}"),
                loss.rolling_mean(period).alias(f"_avg_loss_{period}"),
            ])
            df = df.with_columns([
                (100 - (100 / (1 + pl.col(f"_avg_gain_{period}") / 
                              (pl.col(f"_avg_loss_{period}") + 1e-10)))).alias(f"rsi_{period}")
            ])
        
        # MACD
        df = df.with_columns([
            pl.col("close").ewm_mean(span=12, adjust=False).alias("_ema12"),
            pl.col("close").ewm_mean(span=26, adjust=False).alias("_ema26"),
        ])
        df = df.with_columns([
            (pl.col("_ema12") - pl.col("_ema26")).alias("macd")
        ])
        df = df.with_columns([
            pl.col("macd").ewm_mean(span=9, adjust=False).alias("macd_signal")
        ])
        df = df.with_columns([
            (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist")
        ])
        
        # === VOLATILITY FEATURES ===
        for period in [5, 10, 20]:
            df = df.with_columns([
                pl.col("ret_1").rolling_std(period).alias(f"vol_std_{period}")
            ])
        
        # Volatility ratio and z-score
        df = df.with_columns([
            (pl.col("vol_std_5") / (pl.col("vol_std_20") + 1e-10)).alias("vol_ratio_5_20"),
            ((pl.col("vol_std_5") - pl.col("vol_std_20")) / 
             (pl.col("vol_std_20") + 1e-10)).alias("vol_zscore"),
        ])
        
        # ATR
        df = df.with_columns([
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs()
            ).alias("_tr")
        ])
        df = df.with_columns([
            pl.col("_tr").rolling_mean(14).alias("atr_14"),
            pl.col("_tr").rolling_mean(7).alias("_atr_7"),
        ])
        df = df.with_columns([
            (pl.col("_atr_7") / (pl.col("atr_14") + 1e-10)).alias("atr_ratio")
        ])
        
        # === ORDER FLOW FEATURES ===
        if "taker_buy_volume" in df.columns:
            taker_sell = pl.col("volume") - pl.col("taker_buy_volume")
            df = df.with_columns([
                ((pl.col("taker_buy_volume") - taker_sell) / 
                 (pl.col("volume") + 1e-10)).alias("volume_imbalance"),
                (pl.col("taker_buy_volume") / (pl.col("volume") + 1e-10)).alias("taker_buy_ratio"),
            ])
        else:
            df = df.with_columns([
                pl.lit(0.0).alias("volume_imbalance"),
                pl.lit(0.5).alias("taker_buy_ratio"),
            ])
        
        # Trade imbalance
        if "buy_count" in df.columns and "sell_count" in df.columns:
            df = df.with_columns([
                ((pl.col("buy_count") - pl.col("sell_count")) / 
                 (pl.col("buy_count") + pl.col("sell_count") + 1e-10)).alias("trade_imbalance"),
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("trade_imbalance")])
        
        # Pressure indicators
        df = df.with_columns([
            (pl.col("volume_imbalance").clip(-1, 1) * pl.col("volume")).alias("signed_volume")
        ])
        for period in [5, 10]:
            df = df.with_columns([
                pl.col("signed_volume").rolling_sum(period).alias(f"_sv_{period}"),
                pl.col("volume").rolling_sum(period).alias(f"_vol_{period}"),
            ])
            df = df.with_columns([
                (pl.col(f"_sv_{period}") / (pl.col(f"_vol_{period}") + 1e-10)).alias(f"pressure_{period}")
            ])
        
        df = df.with_columns([
            pl.col("pressure_5").alias("buy_pressure"),
            (-pl.col("pressure_5")).alias("sell_pressure"),
        ])
        
        # Trade intensity
        if "count" in df.columns:
            df = df.with_columns([
                (pl.col("count") / (pl.col("count").rolling_mean(20) + 1e-10)).alias("trade_intensity")
            ])
        else:
            df = df.with_columns([pl.lit(1.0).alias("trade_intensity")])
        
        # VWAP distance
        df = df.with_columns([
            (pl.col("close") * pl.col("volume")).rolling_sum(20).alias("_pv_sum"),
            pl.col("volume").rolling_sum(20).alias("_v_sum"),
        ])
        df = df.with_columns([
            (pl.col("_pv_sum") / (pl.col("_v_sum") + 1e-10)).alias("_vwap_20")
        ])
        df = df.with_columns([
            ((pl.col("close") - pl.col("_vwap_20")) / (pl.col("_vwap_20") + 1e-10) * 100).alias("vwap_distance")
        ])
        
        # === MICROSTRUCTURE FEATURES ===
        if "count" in df.columns:
            df = df.with_columns([
                (pl.col("volume") / (pl.col("count") + 1e-10)).alias("avg_trade_size")
            ])
        else:
            df = df.with_columns([pl.lit(1.0).alias("avg_trade_size")])
        
        df = df.with_columns([pl.lit(0.0).alias("large_trade_ratio")])
        
        if "quote_volume" in df.columns:
            df = df.with_columns([
                (pl.col("quote_volume") / (pl.col("quote_volume").rolling_mean(20) + 1e-10)).alias("quote_volume_ratio")
            ])
        else:
            df = df.with_columns([pl.lit(1.0).alias("quote_volume_ratio")])
        
        # === SENTIMENT FEATURES ===
        if "sum_open_interest" in df.columns:
            df = df.with_columns([
                pl.col("sum_open_interest").pct_change(1).alias("oi_change")
            ])
        else:
            df = df.with_columns([
                pl.lit(0.0).alias("sum_open_interest"),
                pl.lit(0.0).alias("oi_change"),
            ])
        
        if 'sum_toptrader_long_short_ratio' not in df.columns:
            df = df.with_columns([pl.lit(1.0).alias("sum_toptrader_long_short_ratio")])
        
        # === TECHNICAL FEATURES ===
        df = df.with_columns([
            pl.col("close").rolling_mean(10).alias("sma_10"),
            pl.col("close").rolling_mean(50).alias("sma_50"),
            pl.col("close").ewm_mean(span=10, adjust=False).alias("ema_10"),
            pl.col("close").ewm_mean(span=50, adjust=False).alias("ema_50"),
        ])
        df = df.with_columns([
            pl.when(pl.col("sma_10") > pl.col("sma_50")).then(1).otherwise(-1).alias("sma_cross"),
            pl.when(pl.col("ema_10") > pl.col("ema_50")).then(1).otherwise(-1).alias("ema_cross"),
        ])
        
        # Bollinger Bands
        df = df.with_columns([
            pl.col("close").rolling_mean(20).alias("bb_mid"),
            pl.col("close").rolling_std(20).alias("bb_std"),
        ])
        df = df.with_columns([
            (pl.col("bb_mid") + 2 * pl.col("bb_std")).alias("bb_upper"),
            (pl.col("bb_mid") - 2 * pl.col("bb_std")).alias("bb_lower"),
        ])
        df = df.with_columns([
            ((pl.col("close") - pl.col("bb_lower")) / 
             (pl.col("bb_upper") - pl.col("bb_lower") + 1e-10)).alias("bb_position"),
            ((pl.col("bb_upper") - pl.col("bb_lower")) / 
             (pl.col("bb_mid") + 1e-10)).alias("bb_width"),
        ])
        
        # Stochastic
        df = df.with_columns([
            pl.col("high").rolling_max(14).alias("_high_14"),
            pl.col("low").rolling_min(14).alias("_low_14"),
        ])
        df = df.with_columns([
            ((pl.col("close") - pl.col("_low_14")) / 
             (pl.col("_high_14") - pl.col("_low_14") + 1e-10) * 100).alias("stoch_k")
        ])
        df = df.with_columns([
            pl.col("stoch_k").rolling_mean(3).alias("stoch_d")
        ])
        
        # === INTERACTION FEATURES ===
        df = df.with_columns([
            (pl.col("ret_5") * pl.col("vol_std_5")).alias("mom_vol_interaction"),
            (pl.col("volume_imbalance") * pl.col("ret_3")).alias("flow_mom_interaction"),
            ((pl.col("rsi_14") - 50) / 50 * pl.col("macd_hist")).alias("rsi_macd_interaction"),
            (pl.col("vol_ratio_5_20") * pl.col("sma_cross")).alias("vol_trend_interaction"),
            (pl.col("buy_pressure") / (pl.col("sell_pressure").abs() + 1e-10)).alias("pressure_imbalance_ratio"),
        ])
        
        # Clip extreme values
        for col in ['vol_zscore', 'vwap_distance', 'mom_vol_interaction', 'flow_mom_interaction',
                    'rsi_macd_interaction', 'vol_trend_interaction', 'pressure_imbalance_ratio']:
            if col in df.columns:
                df = df.with_columns([pl.col(col).clip(-10, 10).alias(col)])
        
        return df
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Any:
        """Train hybrid ensemble with calibration."""
        from sklearn.utils.class_weight import compute_sample_weight
        
        # Encode labels
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        
        # Compute class weights for balanced training
        class_weights = compute_sample_weight('balanced', y_train_enc)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Base models
        base_models = []
        
        if HAS_BOOSTING:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            base_models.append(('xgb', xgb_model))
            
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )
            base_models.append(('lgb', lgb_model))
        
        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        base_models.append(('rf', rf_model))
        
        et_model = ExtraTreesClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        base_models.append(('et', et_model))
        
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        base_models.append(('gb', gb_model))
        
        # Create ensemble
        if self.ensemble_type == "voting":
            ensemble = VotingClassifier(
                estimators=base_models,
                voting='soft',
                n_jobs=-1
            )
        else:  # stacking
            ensemble = StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced', random_state=42),
                cv=3,
                n_jobs=-1,
                passthrough=False
            )
        
        # Train with class weights
        try:
            ensemble.fit(X_train_scaled, y_train_enc, sample_weight=class_weights)
        except Exception as e:
            print(f"  Warning: sample_weight not supported: {e}")
            ensemble.fit(X_train_scaled, y_train_enc)
        
        # Calibrate if requested
        if self.use_calibration:
            try:
                self.calibrated_model = CalibratedClassifierCV(
                    ensemble,
                    method='sigmoid',
                    cv='prefit'
                )
                self.calibrated_model.fit(X_val_scaled, y_val_enc)
                return self.calibrated_model
            except Exception as e:
                print(f"  Calibration failed: {e}")
                return ensemble
        
        return ensemble
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with the trained model."""
        X_scaled = self.scaler.transform(X)
        
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)
        
        # Decode labels back
        y_pred = self.label_encoder.inverse_transform(y_pred)
        
        return y_pred, y_prob
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from ensemble."""
        importance = {}
        
        base_model = self.model
        if hasattr(self.model, 'estimator'):
            base_model = self.model.estimator
        
        if hasattr(base_model, 'feature_importances_'):
            for i, feat in enumerate(self.feature_names[:len(base_model.feature_importances_)]):
                importance[feat] = float(base_model.feature_importances_[i])
        elif hasattr(base_model, 'estimators_'):
            for name, est in base_model.named_estimators_.items():
                if hasattr(est, 'feature_importances_'):
                    for i, feat in enumerate(self.feature_names[:len(est.feature_importances_)]):
                        if feat not in importance:
                            importance[feat] = 0
                        importance[feat] += float(est.feature_importances_[i])
        
        return importance


def run_strategy(timeframe: str = "15min", ensemble_type: str = "stacking") -> StrategyResult:
    """Run the hybrid ensemble strategy."""
    
    strategy = HybridEnsembleStrategy(
        symbol="BTCUSDT",
        period="2025_11",
        timeframe=timeframe,
        ensemble_type=ensemble_type,
        use_calibration=True,
        probability_threshold=0.55,
    )
    
    return strategy.run(verbose=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid Ensemble Strategy")
    parser.add_argument("--timeframe", type=str, default="15min", 
                        choices=["5min", "15min", "1hr"])
    parser.add_argument("--ensemble", type=str, default="stacking",
                        choices=["voting", "stacking"])
    
    args = parser.parse_args()
    result = run_strategy(args.timeframe, args.ensemble)
    print(f"\nFinal Sharpe Ratio: {result.sharpe_ratio:.4f}")
