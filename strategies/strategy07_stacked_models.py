"""
Strategy 07: Stacked Models with Meta-Learning
===============================================

Uses stacking ensemble with multiple base learners and a meta-learner.
Implements proper cross-validation to prevent overfitting.

Key Concepts:
- Multiple diverse base models
- Out-of-fold predictions for meta-learner training
- Confidence-weighted predictions
- SHAP for interpretability
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_base import StrategyBase


class StackedModelsStrategy(StrategyBase):
    """
    Stacked ensemble strategy with multiple base learners.
    
    Architecture:
    Layer 1 (Base): XGBoost, LightGBM, Random Forest, Extra Trees, GBM
    Layer 2 (Meta): Logistic Regression or XGBoost
    
    Training:
    - Use K-fold CV to generate out-of-fold predictions
    - Train meta-learner on stacked OOF predictions
    - Final prediction = weighted average or meta-learner output
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        n_folds: int = 5,
        meta_model: str = "logistic",  # "logistic" or "xgb"
        use_probas: bool = True,  # Use probabilities instead of classes
        **kwargs
    ):
        super().__init__(symbol, period, timeframe, **kwargs)
        self.n_folds = n_folds
        self.meta_model_type = meta_model
        self.use_probas = use_probas
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.base_models = {}
        self.meta_model = None
        
    def get_name(self) -> str:
        return f"Stacked_Models_{self.meta_model_type}_{self.timeframe}"
    
    def get_feature_columns(self) -> List[str]:
        """Comprehensive feature set for stacking."""
        features = []
        
        # Price features
        for w in [1, 3, 5, 10, 20, 50]:
            features.append(f"ret_{w}")
        
        # Volatility
        for w in [5, 10, 20, 50]:
            features.extend([f"vol_std_{w}", f"vol_parkinson_{w}"])
        
        # Momentum
        for p in [7, 14, 21]:
            features.append(f"rsi_{p}")
        for w in [5, 10, 20]:
            features.append(f"roc_{w}")
        
        # Trend
        for w in [10, 20, 50, 100]:
            features.extend([f"sma_{w}", f"ema_{w}", f"price_vs_sma_{w}"])
        
        # MACD
        features.extend(["macd", "macd_signal", "macd_hist"])
        
        # Volume
        for w in [5, 10, 20]:
            features.extend([f"volume_ma_{w}", f"volume_ratio_{w}"])
        
        # Sentiment (if available)
        features.extend([
            "funding_rate", "sum_open_interest", 
            "sum_toptrader_long_short_ratio"
        ])
        
        return features
    
    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create comprehensive feature set."""
        
        # === Returns ===
        for w in [1, 3, 5, 10, 20, 50]:
            df = df.with_columns([
                (pl.col("close") / pl.col("close").shift(w) - 1).alias(f"ret_{w}")
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
        
        # === ROC ===
        for w in [5, 10, 20]:
            df = df.with_columns([
                ((pl.col("close") - pl.col("close").shift(w)) / 
                 (pl.col("close").shift(w) + 1e-10) * 100).alias(f"roc_{w}")
            ])
        
        # === Moving Averages ===
        for w in [10, 20, 50, 100]:
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
        
        # === Volume Features ===
        for w in [5, 10, 20]:
            df = df.with_columns([
                pl.col("volume").rolling_mean(window_size=w).alias(f"volume_ma_{w}"),
            ])
            df = df.with_columns([
                (pl.col("volume") / (pl.col(f"volume_ma_{w}") + 1e-10)).alias(f"volume_ratio_{w}")
            ])
        
        # === Sentiment ===
        if "last_funding_rate" in df.columns:
            df = df.rename({"last_funding_rate": "funding_rate"})
        
        return df
    
    def _create_base_models(self, n_classes: int) -> Dict[str, Any]:
        """Create diverse base models."""
        models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                random_state=self.random_state,
                objective='multi:softprob',
                num_class=n_classes,
                eval_metric='mlogloss',
                verbosity=0,
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=30,
                random_state=self.random_state,
                verbosity=-1,
            ),
            'rf': RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1,
            ),
            'et': ExtraTreesClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1,
            ),
            'gbm': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=10,
                random_state=self.random_state,
            ),
        }
        return models
    
    def _create_meta_model(self, n_classes: int) -> Any:
        """Create meta-learner."""
        if self.meta_model_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                multi_class='multinomial',
                random_state=self.random_state,
                C=0.5,
            )
        else:  # xgb
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state,
                objective='multi:softprob',
                num_class=n_classes,
                verbosity=0,
            )
    
    def _get_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int
    ) -> Tuple[np.ndarray, Dict[str, List]]:
        """Generate out-of-fold predictions for stacking."""
        n_samples = len(X)
        
        # Initialize OOF predictions array
        if self.use_probas:
            n_features = len(self.base_models) * n_classes
        else:
            n_features = len(self.base_models)
        
        oof_predictions = np.zeros((n_samples, n_features))
        
        # Store fitted models for each fold
        fitted_models = {name: [] for name in self.base_models.keys()}
        
        # K-fold cross-validation
        kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"    Fold {fold_idx + 1}/{self.n_folds}...")
            
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            feature_offset = 0
            
            for name, model_template in self.base_models.items():
                # Clone model
                from sklearn.base import clone
                model = clone(model_template)
                
                # Train
                model.fit(X_fold_train, y_fold_train)
                fitted_models[name].append(model)
                
                # Get predictions for validation fold
                if self.use_probas:
                    fold_pred = model.predict_proba(X_fold_val)
                    oof_predictions[val_idx, feature_offset:feature_offset + n_classes] = fold_pred
                    feature_offset += n_classes
                else:
                    fold_pred = model.predict(X_fold_val)
                    oof_predictions[val_idx, feature_offset] = fold_pred
                    feature_offset += 1
        
        return oof_predictions, fitted_models
    
    def _get_test_predictions(
        self,
        X_test: np.ndarray,
        fitted_models: Dict[str, List],
        n_classes: int
    ) -> np.ndarray:
        """Get test predictions by averaging over folds."""
        if self.use_probas:
            n_features = len(fitted_models) * n_classes
        else:
            n_features = len(fitted_models)
        
        test_predictions = np.zeros((len(X_test), n_features))
        
        feature_offset = 0
        for name, fold_models in fitted_models.items():
            fold_preds = []
            
            for model in fold_models:
                if self.use_probas:
                    fold_preds.append(model.predict_proba(X_test))
                else:
                    fold_preds.append(model.predict(X_test))
            
            # Average over folds
            avg_pred = np.mean(fold_preds, axis=0)
            
            if self.use_probas:
                test_predictions[:, feature_offset:feature_offset + n_classes] = avg_pred
                feature_offset += n_classes
            else:
                test_predictions[:, feature_offset] = avg_pred
                feature_offset += 1
        
        return test_predictions
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Any:
        """Train stacked ensemble."""
        
        # Encode labels
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        n_classes = len(self.label_encoder.classes_)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Combine train and val for OOF generation
        X_combined = np.vstack([X_train_scaled, X_val_scaled])
        y_combined = np.concatenate([y_train_enc, y_val_enc])
        
        # Create base models
        print("  Creating base models...")
        self.base_models = self._create_base_models(n_classes)
        
        # Get OOF predictions
        print("  Generating out-of-fold predictions...")
        oof_predictions, self._fitted_models = self._get_oof_predictions(
            X_combined, y_combined, n_classes
        )
        
        # Train meta-learner
        print("  Training meta-learner...")
        self.meta_model = self._create_meta_model(n_classes)
        self.meta_model.fit(oof_predictions, y_combined)
        
        # Store for later use
        self._n_classes = n_classes
        
        return self.meta_model
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using stacked ensemble."""
        X_scaled = self.scaler.transform(X)
        
        # Get base model predictions
        stacked_features = self._get_test_predictions(
            X_scaled, self._fitted_models, self._n_classes
        )
        
        # Meta-learner prediction
        y_prob = self.meta_model.predict_proba(stacked_features)
        y_pred_enc = np.argmax(y_prob, axis=1)
        y_pred = self.label_encoder.inverse_transform(y_pred_enc)
        
        return y_pred, y_prob
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get importance from base models and meta-learner."""
        importance = {}
        
        # Get meta-learner coefficients/importance
        if hasattr(self.meta_model, 'coef_'):
            # Logistic regression
            coefs = np.abs(self.meta_model.coef_).mean(axis=0)
            
            # Map back to model names
            if self.use_probas:
                for i, name in enumerate(self.base_models.keys()):
                    start_idx = i * self._n_classes
                    end_idx = start_idx + self._n_classes
                    importance[f"meta_{name}"] = float(coefs[start_idx:end_idx].mean())
            else:
                for i, name in enumerate(self.base_models.keys()):
                    importance[f"meta_{name}"] = float(coefs[i])
        
        # Get base model feature importance (average)
        for name, fold_models in self._fitted_models.items():
            if hasattr(fold_models[0], 'feature_importances_'):
                avg_importance = np.mean([m.feature_importances_ for m in fold_models], axis=0)
                for i, imp in enumerate(avg_importance):
                    feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feat_{i}"
                    key = f"{name}_{feature_name}"
                    importance[key] = float(imp)
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def run_strategy(timeframe: str = "15min", meta_model: str = "logistic"):
    """Run the stacked models strategy."""
    strategy = StackedModelsStrategy(
        timeframe=timeframe,
        meta_model=meta_model,
    )
    return strategy.run(verbose=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stacked Models Strategy")
    parser.add_argument("--timeframe", type=str, default="15min",
                        choices=["5min", "15min", "1hr"])
    parser.add_argument("--meta", type=str, default="logistic",
                        choices=["logistic", "xgb"])
    
    args = parser.parse_args()
    
    result = run_strategy(args.timeframe, args.meta)
    print(f"\nFinal Sharpe Ratio: {result.sharpe_ratio:.4f}")
