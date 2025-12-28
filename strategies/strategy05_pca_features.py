"""
Strategy 05: PCA & Feature Transformation
==========================================

Uses dimensionality reduction and feature transformation techniques:
- PCA (Principal Component Analysis)
- Feature selection (RFE, mutual information)
- Polynomial features
- Interaction features

Goal: Find optimal feature representation for ML models.
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_base import StrategyBase


class PCAFeatureStrategy(StrategyBase):
    """
    Feature transformation strategy using PCA and feature selection.
    
    Approach:
    1. Create comprehensive feature set
    2. Apply various transformation techniques
    3. Use feature selection to find best subset
    4. Train model on transformed features
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        period: str = "2025_11",
        timeframe: str = "15min",
        n_components: float = 0.95,  # Variance to retain or number of components
        feature_method: str = "pca",  # "pca", "mutual_info", "rfe", "combined"
        add_interactions: bool = True,
        **kwargs
    ):
        super().__init__(symbol, period, timeframe, **kwargs)
        self.n_components = n_components
        self.feature_method = feature_method
        self.add_interactions = add_interactions
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_selector = None
        self.poly_features = None
        
    def get_name(self) -> str:
        return f"PCA_Features_{self.feature_method}_{self.timeframe}"
    
    def get_feature_columns(self) -> List[str]:
        """All available features before transformation."""
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
        
        # Order flow
        features.extend(["volume_imbalance", "trade_imbalance", "vpin"])
        
        # Sentiment (if available)
        features.extend([
            "funding_rate", "sum_open_interest", 
            "sum_toptrader_long_short_ratio", "count_toptrader_long_short_ratio"
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
        
        # === Order Flow (if available from aggregation) ===
        if "buy_volume" in df.columns and "sell_volume" in df.columns:
            df = df.with_columns([
                ((pl.col("buy_volume") - pl.col("sell_volume")) / 
                 (pl.col("volume") + 1e-10)).alias("volume_imbalance"),
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("volume_imbalance")])
        
        if "buy_count" in df.columns and "sell_count" in df.columns:
            df = df.with_columns([
                ((pl.col("buy_count") - pl.col("sell_count")).cast(pl.Float64) / 
                 (pl.col("trade_count").cast(pl.Float64) + 1e-10)).alias("trade_imbalance"),
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("trade_imbalance")])
        
        if "volume_imbalance" in df.columns:
            df = df.with_columns([
                pl.col("volume_imbalance").abs().rolling_mean(window_size=10).alias("vpin")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("vpin")])
        
        # === Sentiment (forward fill from metrics if available) ===
        if "last_funding_rate" in df.columns:
            df = df.rename({"last_funding_rate": "funding_rate"})
        
        return df
    
    def prepare_features(
        self,
        df: pl.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract and prepare features with transformations."""
        # Get available features
        available = [c for c in feature_cols if c in df.columns]
        
        X = df.select(available).to_numpy()
        y = df["label"].to_numpy()
        
        if "sample_weight" in df.columns:
            w = df["sample_weight"].to_numpy()
        else:
            w = np.ones(len(y))
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.nan_to_num(w, nan=1.0)
        w = np.clip(w, 0.01, 100.0)
        
        self.feature_names = available
        
        return X, y, w
    
    def _fit_transformers(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit feature transformers and return transformed data."""
        
        # Step 1: Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Step 2: Add interaction features (optional)
        if self.add_interactions:
            # Select top features for interactions to avoid explosion
            n_interact = min(10, X_train_scaled.shape[1])
            top_idx = np.argsort(np.abs(X_train_scaled).mean(axis=0))[-n_interact:]
            
            X_interact_train = X_train_scaled[:, top_idx]
            X_interact_val = X_val_scaled[:, top_idx]
            
            self.poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            X_poly_train = self.poly_features.fit_transform(X_interact_train)
            X_poly_val = self.poly_features.transform(X_interact_val)
            
            # Append polynomial features
            X_train_scaled = np.hstack([X_train_scaled, X_poly_train[:, n_interact:]])  # Skip original features
            X_val_scaled = np.hstack([X_val_scaled, X_poly_val[:, n_interact:]])
        
        # Step 3: Feature transformation/selection
        if self.feature_method == "pca":
            self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
            X_train_transformed = self.pca.fit_transform(X_train_scaled)
            X_val_transformed = self.pca.transform(X_val_scaled)
            
            print(f"  PCA: {X_train_scaled.shape[1]} -> {X_train_transformed.shape[1]} components")
            print(f"  Explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
            
        elif self.feature_method == "mutual_info":
            # Select features by mutual information
            n_select = min(50, X_train_scaled.shape[1])
            self.feature_selector = SelectKBest(
                score_func=mutual_info_classif,
                k=n_select
            )
            X_train_transformed = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_val_transformed = self.feature_selector.transform(X_val_scaled)
            
            print(f"  Mutual Info: {X_train_scaled.shape[1]} -> {n_select} features")
            
        elif self.feature_method == "rfe":
            # Recursive Feature Elimination with Random Forest
            base_model = RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=self.random_state, n_jobs=-1
            )
            n_select = min(50, X_train_scaled.shape[1])
            self.feature_selector = RFE(base_model, n_features_to_select=n_select, step=10)
            X_train_transformed = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_val_transformed = self.feature_selector.transform(X_val_scaled)
            
            print(f"  RFE: {X_train_scaled.shape[1]} -> {n_select} features")
            
        elif self.feature_method == "combined":
            # Combined approach: PCA + feature selection
            # First, reduce with PCA
            self.pca = PCA(n_components=0.99, random_state=self.random_state)
            X_pca_train = self.pca.fit_transform(X_train_scaled)
            X_pca_val = self.pca.transform(X_val_scaled)
            
            # Then, select best features
            n_select = min(40, X_pca_train.shape[1])
            self.feature_selector = SelectKBest(
                score_func=mutual_info_classif,
                k=n_select
            )
            X_train_transformed = self.feature_selector.fit_transform(X_pca_train, y_train)
            X_val_transformed = self.feature_selector.transform(X_pca_val)
            
            print(f"  Combined: {X_train_scaled.shape[1]} -> {X_pca_train.shape[1]} (PCA) -> {n_select} features")
            
        else:
            X_train_transformed = X_train_scaled
            X_val_transformed = X_val_scaled
        
        return X_train_transformed, X_val_transformed
    
    def _transform_test(self, X_test: np.ndarray) -> np.ndarray:
        """Apply saved transformations to test data."""
        X_scaled = self.scaler.transform(X_test)
        
        if self.add_interactions and self.poly_features is not None:
            n_interact = min(10, X_test.shape[1])
            top_idx = np.argsort(np.abs(X_scaled).mean(axis=0))[-n_interact:]
            X_interact = X_scaled[:, top_idx]
            X_poly = self.poly_features.transform(X_interact)
            X_scaled = np.hstack([X_scaled, X_poly[:, n_interact:]])
        
        if self.pca is not None and self.feature_method in ["pca", "combined"]:
            X_transformed = self.pca.transform(X_scaled)
            if self.feature_selector is not None:
                X_transformed = self.feature_selector.transform(X_transformed)
        elif self.feature_selector is not None:
            X_transformed = self.feature_selector.transform(X_scaled)
        else:
            X_transformed = X_scaled
        
        return X_transformed
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Any:
        """Train model with transformed features."""
        
        # Encode labels
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        
        # Apply transformations
        X_train_transformed, X_val_transformed = self._fit_transformers(
            X_train, y_train_enc, X_val
        )
        
        # Store for later use in prediction
        self._X_train_transformed = X_train_transformed
        self._X_val_transformed = X_val_transformed
        
        n_classes = len(self.label_encoder.classes_)
        
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.9,  # Higher since features are already selected
            min_child_weight=5,
            reg_alpha=0.3,
            reg_lambda=1.5,
            random_state=self.random_state,
            objective='multi:softprob',
            num_class=n_classes,
            eval_metric='mlogloss',
            early_stopping_rounds=30,
            verbosity=0,
        )
        
        model.fit(
            X_train_transformed, y_train_enc,
            sample_weight=sample_weights,
            eval_set=[(X_val_transformed, y_val_enc)],
            verbose=False,
        )
        
        self.model = model
        return model
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on transformed features."""
        X_transformed = self._transform_test(X)
        
        y_prob = self.model.predict_proba(X_transformed)
        y_pred_enc = np.argmax(y_prob, axis=1)
        y_pred = self.label_encoder.inverse_transform(y_pred_enc)
        
        return y_pred, y_prob
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (PCA components or selected features)."""
        if self.model is None:
            return {}
        
        importances = self.model.feature_importances_
        
        if self.feature_method == "pca":
            # Return component importance
            return {f"PC{i+1}": float(imp) for i, imp in enumerate(importances)}
        else:
            # Return feature importance
            if hasattr(self.feature_selector, 'get_support'):
                selected_idx = self.feature_selector.get_support(indices=True)
                names = [self.feature_names[i] if i < len(self.feature_names) 
                        else f"poly_{i}" for i in selected_idx]
                return dict(sorted(
                    zip(names, importances), 
                    key=lambda x: x[1], 
                    reverse=True
                ))
        
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}


def run_strategy(timeframe: str = "15min", feature_method: str = "pca"):
    """Run the PCA feature strategy."""
    strategy = PCAFeatureStrategy(
        timeframe=timeframe,
        feature_method=feature_method,
    )
    return strategy.run(verbose=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PCA Feature Strategy")
    parser.add_argument("--timeframe", type=str, default="15min",
                        choices=["5min", "15min", "1hr"])
    parser.add_argument("--method", type=str, default="pca",
                        choices=["pca", "mutual_info", "rfe", "combined"])
    
    args = parser.parse_args()
    
    result = run_strategy(args.timeframe, args.method)
    print(f"\nFinal Sharpe Ratio: {result.sharpe_ratio:.4f}")
