"""
Strategy 01: Momentum Ensemble
==============================

Focus on momentum and trend-following features with an ensemble of:
- XGBoost
- LightGBM  
- Random Forest

Uses voting or stacking to combine predictions.
Optimized for 5min, 15min, and 1hr timeframes.
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_base import StrategyBase, StrategyResult


class MomentumEnsembleStrategy(StrategyBase):
    """
    Momentum-based strategy using ensemble of gradient boosting and random forest.
    
    Feature Focus:
    - Price momentum (ROC, RSI, Stochastic)
    - Trend indicators (MA crossovers, MACD, ADX)
    - Volatility-adjusted momentum
    - Multi-timeframe momentum alignment
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        ensemble_method: str = "voting",  # "voting" or "stacking"
        **kwargs
    ):
        super().__init__(symbol, period, timeframe, **kwargs)
        self.ensemble_method = ensemble_method
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def get_name(self) -> str:
        return f"Momentum_Ensemble_{self.ensemble_method}_{self.timeframe}"
    
    def get_feature_columns(self) -> List[str]:
        """Momentum and trend focused features."""
        features = []
        
        # Price returns at multiple windows
        for w in [1, 3, 5, 10, 20, 50]:
            features.append(f"ret_{w}")
        
        # RSI at multiple periods
        for p in [7, 14, 21]:
            features.append(f"rsi_{p}")
        
        # Rate of change
        for w in [5, 10, 20]:
            features.append(f"roc_{w}")
        
        # Stochastic oscillator
        for w in [14, 21]:
            features.extend([f"stoch_k_{w}", f"stoch_d_{w}"])
        
        # Moving average features
        for w in [10, 20, 50, 100]:
            features.extend([
                f"sma_{w}",
                f"ema_{w}",
                f"price_vs_sma_{w}",
                f"price_vs_ema_{w}",
            ])
        
        # MA crossover signals
        features.extend([
            "sma_cross_10_50",
            "sma_cross_20_100",
            "ema_cross_10_50",
        ])
        
        # MACD
        features.extend([
            "macd", "macd_signal", "macd_hist",
            "macd_cross", "macd_hist_change",
        ])
        
        # ADX (trend strength)
        features.extend([
            "adx_14", "plus_di_14", "minus_di_14",
            "adx_trend_strength", "di_cross",
        ])
        
        # Volatility-adjusted momentum
        features.extend([
            "momentum_vol_adj_5",
            "momentum_vol_adj_20",
            "momentum_vol_ratio",
        ])
        
        # Trend consistency
        features.extend([
            "trend_consistency_10",
            "trend_consistency_20",
            "higher_highs_10",
            "lower_lows_10",
        ])
        
        # Multi-timeframe alignment (for intraday)
        features.extend([
            "mtf_momentum_alignment",
            "mtf_trend_alignment",
        ])
        
        return features
    
    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create momentum and trend features."""
        
        # === Price Returns ===
        for w in [1, 3, 5, 10, 20, 50]:
            df = df.with_columns([
                (pl.col("close") / pl.col("close").shift(w) - 1).alias(f"ret_{w}")
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
        
        # === Rate of Change ===
        for w in [5, 10, 20]:
            df = df.with_columns([
                ((pl.col("close") - pl.col("close").shift(w)) / 
                 (pl.col("close").shift(w) + 1e-10) * 100).alias(f"roc_{w}")
            ])
        
        # === Stochastic Oscillator ===
        for w in [14, 21]:
            lowest_low = pl.col("low").rolling_min(window_size=w)
            highest_high = pl.col("high").rolling_max(window_size=w)
            
            df = df.with_columns([
                ((pl.col("close") - lowest_low) / 
                 (highest_high - lowest_low + 1e-10) * 100).alias(f"stoch_k_{w}")
            ])
            df = df.with_columns([
                pl.col(f"stoch_k_{w}").rolling_mean(window_size=3).alias(f"stoch_d_{w}")
            ])
        
        # === Moving Averages ===
        for w in [10, 20, 50, 100]:
            df = df.with_columns([
                pl.col("close").rolling_mean(window_size=w).alias(f"sma_{w}"),
                pl.col("close").ewm_mean(span=w, adjust=False).alias(f"ema_{w}"),
            ])
            df = df.with_columns([
                (pl.col("close") / pl.col(f"sma_{w}") - 1).alias(f"price_vs_sma_{w}"),
                (pl.col("close") / pl.col(f"ema_{w}") - 1).alias(f"price_vs_ema_{w}"),
            ])
        
        # === MA Crossover Signals ===
        df = df.with_columns([
            pl.when(pl.col("sma_10") > pl.col("sma_50"))
            .then(pl.lit(1))
            .otherwise(pl.lit(-1)).alias("sma_cross_10_50"),
            
            pl.when(pl.col("sma_20") > pl.col("sma_100"))
            .then(pl.lit(1))
            .otherwise(pl.lit(-1)).alias("sma_cross_20_100"),
            
            pl.when(pl.col("ema_10") > pl.col("ema_50"))
            .then(pl.lit(1))
            .otherwise(pl.lit(-1)).alias("ema_cross_10_50"),
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
            (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist"),
            pl.when(pl.col("macd") > pl.col("macd_signal"))
            .then(pl.lit(1))
            .otherwise(pl.lit(-1)).alias("macd_cross"),
            (pl.col("macd") - pl.col("macd").shift(1)).alias("macd_hist_change"),
        ])
        
        # === ADX ===
        df = self._add_adx(df, period=14)
        
        # === Volatility-Adjusted Momentum ===
        for w in [5, 20]:
            vol = pl.col("close").pct_change().rolling_std(window_size=w)
            mom = pl.col("close") / pl.col("close").shift(w) - 1
            
            df = df.with_columns([
                (mom / (vol + 1e-10)).alias(f"momentum_vol_adj_{w}")
            ])
        
        df = df.with_columns([
            (pl.col("momentum_vol_adj_5") / 
             (pl.col("momentum_vol_adj_20").abs() + 1e-10)).alias("momentum_vol_ratio")
        ])
        
        # === Trend Consistency ===
        for w in [10, 20]:
            # Count positive returns in window
            df = df.with_columns([
                (pl.col("ret_1") > 0).cast(pl.Int32).rolling_sum(window_size=w)
                .truediv(w).alias(f"trend_consistency_{w}")
            ])
        
        # Higher highs / Lower lows
        df = df.with_columns([
            (pl.col("high") > pl.col("high").shift(1)).cast(pl.Int32)
            .rolling_sum(window_size=10).alias("higher_highs_10"),
            (pl.col("low") < pl.col("low").shift(1)).cast(pl.Int32)
            .rolling_sum(window_size=10).alias("lower_lows_10"),
        ])
        
        # === Multi-Timeframe Alignment ===
        # Use longer lookbacks to simulate higher timeframe
        df = df.with_columns([
            # Momentum alignment: short and long momentum agree
            pl.when(
                ((pl.col("ret_5") > 0) & (pl.col("ret_20") > 0)) |
                ((pl.col("ret_5") < 0) & (pl.col("ret_20") < 0))
            ).then(pl.lit(1)).otherwise(pl.lit(-1)).alias("mtf_momentum_alignment"),
            
            # Trend alignment: multiple MAs aligned
            pl.when(
                (pl.col("ema_10") > pl.col("ema_20")) &
                (pl.col("ema_20") > pl.col("ema_50"))
            ).then(pl.lit(1))
            .when(
                (pl.col("ema_10") < pl.col("ema_20")) &
                (pl.col("ema_20") < pl.col("ema_50"))
            ).then(pl.lit(-1))
            .otherwise(pl.lit(0)).alias("mtf_trend_alignment"),
        ])
        
        return df
    
    def _add_adx(self, df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Add ADX indicator."""
        # True Range
        df = df.with_columns([
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs()
            ).alias("_tr")
        ])
        
        # Directional Movement
        df = df.with_columns([
            pl.when(
                (pl.col("high") - pl.col("high").shift(1)) > 
                (pl.col("low").shift(1) - pl.col("low"))
            ).then(
                pl.max_horizontal(pl.col("high") - pl.col("high").shift(1), pl.lit(0.0))
            ).otherwise(pl.lit(0.0)).alias("_plus_dm"),
            
            pl.when(
                (pl.col("low").shift(1) - pl.col("low")) > 
                (pl.col("high") - pl.col("high").shift(1))
            ).then(
                pl.max_horizontal(pl.col("low").shift(1) - pl.col("low"), pl.lit(0.0))
            ).otherwise(pl.lit(0.0)).alias("_minus_dm"),
        ])
        
        # Smoothed averages
        df = df.with_columns([
            pl.col("_tr").ewm_mean(span=period, adjust=False).alias("_atr"),
            pl.col("_plus_dm").ewm_mean(span=period, adjust=False).alias("_plus_dm_smooth"),
            pl.col("_minus_dm").ewm_mean(span=period, adjust=False).alias("_minus_dm_smooth"),
        ])
        
        # DI+ and DI-
        df = df.with_columns([
            (pl.col("_plus_dm_smooth") / (pl.col("_atr") + 1e-10) * 100).alias(f"plus_di_{period}"),
            (pl.col("_minus_dm_smooth") / (pl.col("_atr") + 1e-10) * 100).alias(f"minus_di_{period}"),
        ])
        
        # DX and ADX
        df = df.with_columns([
            ((pl.col(f"plus_di_{period}") - pl.col(f"minus_di_{period}")).abs() /
             (pl.col(f"plus_di_{period}") + pl.col(f"minus_di_{period}") + 1e-10) * 100).alias("_dx")
        ])
        
        # Create ADX
        df = df.with_columns([
            pl.col("_dx").ewm_mean(span=period, adjust=False).alias(f"adx_{period}"),
        ])
        
        # Create ADX-derived signals
        df = df.with_columns([
            pl.when(pl.col(f"adx_{period}") > 25).then(pl.lit(1)).otherwise(pl.lit(0)).alias("adx_trend_strength"),
            pl.when(pl.col(f"plus_di_{period}") > pl.col(f"minus_di_{period}"))
            .then(pl.lit(1)).otherwise(pl.lit(-1)).alias("di_cross"),
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
        """Train ensemble of models."""
        
        # Encode labels
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        n_classes = len(self.label_encoder.classes_)
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            objective='multi:softprob',
            num_class=n_classes,
            eval_metric='mlogloss',
            early_stopping_rounds=20,
            verbosity=0,
        )
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            objective='multiclass',
            num_class=n_classes,
            verbosity=-1,
        )
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1,
        )
        
        if self.ensemble_method == "voting":
            # Soft voting ensemble
            ensemble = VotingClassifier(
                estimators=[
                    ('xgb', xgb_model),
                    ('lgb', lgb_model),
                    ('rf', rf_model),
                ],
                voting='soft',
                n_jobs=-1,
            )
            
            # Fit ensemble (can't use early stopping with VotingClassifier directly)
            # Train individual models first for early stopping
            xgb_model.fit(
                X_train_scaled, y_train_enc,
                sample_weight=sample_weights,
                eval_set=[(X_val_scaled, y_val_enc)],
                verbose=False,
            )
            
            lgb_model.fit(
                X_train_scaled, y_train_enc,
                sample_weight=sample_weights,
                eval_set=[(X_val_scaled, y_val_enc)],
            )
            
            rf_model.fit(X_train_scaled, y_train_enc, sample_weight=sample_weights)
            
            # Create ensemble with fitted models
            ensemble.estimators_ = [xgb_model, lgb_model, rf_model]
            ensemble.le_ = LabelEncoder().fit(y_train_enc)
            ensemble.classes_ = self.label_encoder.classes_
            
            self.model = ensemble
            self._individual_models = {
                'xgb': xgb_model,
                'lgb': lgb_model,
                'rf': rf_model,
            }
            
        else:  # stacking
            from sklearn.ensemble import StackingClassifier
            from sklearn.linear_model import LogisticRegression
            
            # Train base models with early stopping
            xgb_model.fit(
                X_train_scaled, y_train_enc,
                sample_weight=sample_weights,
                eval_set=[(X_val_scaled, y_val_enc)],
                verbose=False,
            )
            
            lgb_model.fit(
                X_train_scaled, y_train_enc,
                sample_weight=sample_weights,
                eval_set=[(X_val_scaled, y_val_enc)],
            )
            
            rf_model.fit(X_train_scaled, y_train_enc, sample_weight=sample_weights)
            
            # Stacking with logistic regression meta-learner
            stack = StackingClassifier(
                estimators=[
                    ('xgb', xgb_model),
                    ('lgb', lgb_model),
                    ('rf', rf_model),
                ],
                final_estimator=LogisticRegression(
                    max_iter=500, 
                    multi_class='multinomial',
                    random_state=self.random_state
                ),
                cv=3,
                stack_method='predict_proba',
                n_jobs=-1,
            )
            
            # Fit stacking meta-learner
            stack.fit(X_train_scaled, y_train_enc)
            
            self.model = stack
            self._individual_models = {
                'xgb': xgb_model,
                'lgb': lgb_model,
                'rf': rf_model,
            }
        
        return self.model
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with ensemble."""
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_scaled)
            y_pred_enc = np.argmax(y_prob, axis=1)
        else:
            # For voting classifier
            y_pred_enc = self.model.predict(X_scaled)
            # Aggregate probabilities from individual models
            probs = []
            for model in self._individual_models.values():
                probs.append(model.predict_proba(X_scaled))
            y_prob = np.mean(probs, axis=0)
        
        y_pred = self.label_encoder.inverse_transform(y_pred_enc)
        
        return y_pred, y_prob
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Aggregate feature importance from all models."""
        if not hasattr(self, '_individual_models'):
            return super().get_feature_importance()
        
        # Average importance across models
        importances = {}
        n_models = 0
        
        for name, model in self._individual_models.items():
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                for i, feat in enumerate(self.feature_names):
                    importances[feat] = importances.get(feat, 0) + imp[i]
                n_models += 1
        
        if n_models > 0:
            importances = {k: v / n_models for k, v in importances.items()}
        
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))


def run_strategy(timeframe: str = "15min", ensemble_method: str = "voting"):
    """Run the momentum ensemble strategy."""
    strategy = MomentumEnsembleStrategy(
        timeframe=timeframe,
        ensemble_method=ensemble_method,
    )
    return strategy.run(verbose=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Momentum Ensemble Strategy")
    parser.add_argument("--timeframe", type=str, default="15min", 
                        choices=["5min", "15min", "1hr"])
    parser.add_argument("--ensemble", type=str, default="voting",
                        choices=["voting", "stacking"])
    
    args = parser.parse_args()
    
    result = run_strategy(args.timeframe, args.ensemble)
    print(f"\nFinal Sharpe Ratio: {result.sharpe_ratio:.4f}")
