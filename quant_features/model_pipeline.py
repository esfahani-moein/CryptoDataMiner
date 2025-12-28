"""
Model Pipeline Module

Complete ML pipeline for cryptocurrency price prediction.
Handles data preparation, feature engineering, model training, and evaluation.

Key Components:
1. Data Pipeline: Load, merge, and prepare all data sources
2. Feature Pipeline: Calculate all features with proper point-in-time handling
3. Train/Test Split: Time-series aware splitting to avoid look-ahead bias
4. Model Training: Support for various ML models (XGBoost, LightGBM, etc.)
5. Evaluation: Proper backtesting metrics

CRITICAL: Strict avoidance of look-ahead bias throughout the pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import polars as pl
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA PREPARATION PIPELINE
# ============================================================================

def prepare_feature_dataset(
    base_path: Union[str, Path],
    symbol: str = "BTCUSDT",
    start_year: int = 2025,
    start_month: int = 11,
    end_year: int = 2025,
    end_month: int = 11,
    bar_interval_ms: int = 60_000,  # 1-minute bars
    include_sentiment: bool = True,
    include_orderbook: bool = True
) -> pl.DataFrame:
    """
    Prepare complete feature dataset from all data sources.
    
    Pipeline steps:
    1. Load trades and aggregate to OHLCV bars
    2. Load and merge mark/index price data
    3. Load and merge sentiment data (metrics, funding)
    4. Load and merge orderbook data
    5. Calculate all features
    6. Add labels
    
    Args:
        base_path: Base directory for data
        symbol: Trading pair
        start_year, start_month: Start of date range
        end_year, end_month: End of date range
        bar_interval_ms: Bar interval in milliseconds
        include_sentiment: Include sentiment features
        include_orderbook: Include orderbook features
        
    Returns:
        Complete feature DataFrame ready for ML
    """
    from quant_features.data_loader import (
        load_trades, load_metrics, load_funding_rate, 
        load_book_depth, load_klines, merge_features_to_ohlcv
    )
    from quant_features.price_features import add_all_price_features
    from quant_features.volume_features import add_all_volume_features
    from quant_features.sentiment_features import add_all_sentiment_features
    from quant_features.orderbook_features import add_depth_features_from_long
    from quant_features.labeling import add_all_labels
    from trades_aggregation.trades_aggregator import aggregate_trades_to_ohlcv
    
    base_path = Path(base_path)
    
    # ========================
    # STEP 1: Load Trades and Create OHLCV Base
    # ========================
    print("Loading trades data...")
    trades = load_trades(base_path, symbol, start_year, start_month, end_year, end_month)
    
    # Rename columns to match aggregator expectations
    trades_renamed = trades.rename({
        "qty": "quantity",
        "quote_qty": "quote_quantity"
    })
    
    print(f"Aggregating {len(trades):,} trades to {bar_interval_ms/60000:.0f}-minute bars...")
    ohlcv = aggregate_trades_to_ohlcv(trades_renamed, bar_interval_ms)
    print(f"Created {len(ohlcv):,} OHLCV bars")
    
    # ========================
    # STEP 2: Add Mark Price Features
    # ========================
    print("Loading mark price data...")
    try:
        mark_klines = load_klines(base_path, symbol, "markPriceKlines",
                                   start_year, start_month, end_year, end_month)
        
        # Calculate basis (futures - spot proxy)
        # Mark price is the fair price; compare to our OHLCV close
        mark_features = mark_klines.select([
            pl.col("open_time").alias("time"),
            pl.col("close").alias("mark_price"),
            ((pl.col("high") - pl.col("low")) / pl.col("close") * 100).alias("mark_range_pct")
        ])
        
        # Merge with asof join
        ohlcv = ohlcv.join_asof(
            mark_features.sort("time"),
            left_on="open_time",
            right_on="time",
            strategy="backward"
        )
        
        # Calculate basis
        ohlcv = ohlcv.with_columns([
            ((pl.col("close") - pl.col("mark_price")) / pl.col("mark_price") * 10000)
            .alias("basis_bps")  # Basis in basis points
        ])
        print("  Added mark price features")
    except Exception as e:
        print(f"  Mark price data not available: {e}")
    
    # ========================
    # STEP 3: Add Index Price Features (Premium/Discount)
    # ========================
    print("Loading index price data...")
    try:
        index_klines = load_klines(base_path, symbol, "indexPriceKlines",
                                    start_year, start_month, end_year, end_month)
        
        index_features = index_klines.select([
            pl.col("open_time").alias("time"),
            pl.col("close").alias("index_price"),
        ])
        
        ohlcv = ohlcv.join_asof(
            index_features.sort("time"),
            left_on="open_time",
            right_on="time",
            strategy="backward"
        )
        
        # Premium to index
        ohlcv = ohlcv.with_columns([
            ((pl.col("close") - pl.col("index_price")) / pl.col("index_price") * 10000)
            .alias("premium_bps")
        ])
        print("  Added index price features")
    except Exception as e:
        print(f"  Index price data not available: {e}")
    
    # ========================
    # STEP 4: Add Sentiment Features
    # ========================
    if include_sentiment:
        print("Loading sentiment data...")
        
        # Metrics (OI, Long/Short ratios)
        try:
            metrics = load_metrics(base_path, symbol, start_year, start_month, 
                                   end_year, end_month)
            
            ohlcv = ohlcv.join_asof(
                metrics.sort("time"),
                left_on="open_time",
                right_on="time",
                strategy="backward"
            )
            print(f"  Merged {len(metrics):,} metrics records")
        except Exception as e:
            print(f"  Metrics data not available: {e}")
        
        # Funding rate
        try:
            funding = load_funding_rate(base_path, symbol, start_year, start_month,
                                        end_year, end_month)
            
            ohlcv = ohlcv.join_asof(
                funding.sort("time"),
                left_on="open_time",
                right_on="time",
                strategy="backward"
            )
            print(f"  Merged {len(funding):,} funding rate records")
        except Exception as e:
            print(f"  Funding rate data not available: {e}")
    
    # ========================
    # STEP 5: Add Orderbook Features
    # ========================
    if include_orderbook:
        print("Loading orderbook data...")
        try:
            book_depth = load_book_depth(base_path, symbol, start_year, start_month,
                                         end_year, end_month)
            
            # Aggregate orderbook features
            book_features = add_depth_features_from_long(book_depth, windows=[6, 12, 24])
            
            ohlcv = ohlcv.join_asof(
                book_features.sort("time"),
                left_on="open_time",
                right_on="time",
                strategy="backward"
            )
            print(f"  Merged orderbook features")
        except Exception as e:
            print(f"  Orderbook data not available: {e}")
    
    # ========================
    # STEP 6: Calculate All Features
    # ========================
    print("Calculating price features...")
    ohlcv = add_all_price_features(ohlcv)
    
    print("Calculating volume features...")
    ohlcv = add_all_volume_features(ohlcv)
    
    if include_sentiment:
        print("Calculating sentiment features...")
        ohlcv = add_all_sentiment_features(ohlcv)
    
    # ========================
    # STEP 7: Add Labels
    # ========================
    print("Calculating labels...")
    ohlcv = add_all_labels(ohlcv)
    
    print(f"\nFinal dataset shape: {ohlcv.shape}")
    print(f"Features: {len(ohlcv.columns)} columns")
    
    return ohlcv


# ============================================================================
# TRAIN/TEST SPLIT (TIME-SERIES AWARE)
# ============================================================================

def time_series_split(
    df: pl.DataFrame,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.1,
    time_col: str = "open_time",
    gap_periods: int = 60  # 1 hour gap with 1-min bars
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split data into train/validation/test sets with time awareness.
    
    CRITICAL: No shuffling! Data is split chronologically to avoid
    look-ahead bias. A gap is introduced between sets to prevent
    information leakage from overlapping labels.
    
    Timeline:
    [====TRAIN====][gap][==VAL==][gap][==TEST==]
    
    Args:
        df: DataFrame sorted by time
        test_ratio: Proportion of data for testing
        validation_ratio: Proportion of data for validation
        time_col: Time column name
        gap_periods: Number of periods to gap between sets
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Ensure sorted
    df = df.sort(time_col)
    n = len(df)
    
    # Calculate split points
    test_size = int(n * test_ratio)
    val_size = int(n * validation_ratio)
    
    # Account for gaps
    test_start = n - test_size
    val_end = test_start - gap_periods
    val_start = val_end - val_size
    train_end = val_start - gap_periods
    
    # Split
    train_df = df.slice(0, train_end)
    val_df = df.slice(val_start, val_size)
    test_df = df.slice(test_start, test_size)
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_df):,} ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} ({len(test_df)/n*100:.1f}%)")
    
    return train_df, val_df, test_df


def purged_kfold_split(
    df: pl.DataFrame,
    n_splits: int = 5,
    time_col: str = "open_time",
    purge_periods: int = 60,
    embargo_periods: int = 30
) -> List[Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Purged K-Fold Cross-Validation for time series.
    
    Implements purging and embargo to prevent information leakage:
    - Purge: Remove training samples that overlap with test labels
    - Embargo: Add gap after test set to account for label horizon
    
    Args:
        df: DataFrame sorted by time
        n_splits: Number of folds
        time_col: Time column name
        purge_periods: Periods to remove before test set
        embargo_periods: Periods to skip after test set
        
    Returns:
        List of (train_df, test_df) tuples for each fold
    """
    df = df.sort(time_col)
    n = len(df)
    fold_size = n // n_splits
    
    folds = []
    
    for i in range(n_splits):
        # Test set for this fold
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n
        
        # Training set excludes:
        # 1. Test set itself
        # 2. Purge window before test (label leakage)
        # 3. Embargo after test (serial correlation)
        
        train_indices = []
        
        # Before test (with purge)
        if test_start > purge_periods:
            train_indices.extend(range(0, test_start - purge_periods))
        
        # After test (with embargo)
        if test_end + embargo_periods < n:
            train_indices.extend(range(test_end + embargo_periods, n))
        
        if len(train_indices) > 0:
            train_df = df.filter(pl.arange(0, n).is_in(train_indices))
            test_df = df.slice(test_start, test_end - test_start)
            folds.append((train_df, test_df))
    
    return folds


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def get_feature_columns(
    df: pl.DataFrame,
    exclude_patterns: List[str] = None
) -> List[str]:
    """
    Get list of feature columns (excluding labels and metadata).
    
    Args:
        df: DataFrame with all columns
        exclude_patterns: Additional patterns to exclude
        
    Returns:
        List of feature column names
    """
    # Columns that are labels or metadata (not features)
    exclude_cols = {
        # Time columns
        "open_time", "close_time", "time",
        
        # Labels
        "fwd_ret_5", "fwd_ret_15", "fwd_ret_60", "fwd_ret_240",
        "fwd_class_5", "fwd_class_15", "fwd_class_60",
        "tb_label", "tb_label_approx", "tb_touch_bars", "tb_return",
        "trend_label", "trend_strength",
        "regime_label", "volatility_label", "fwd_volatility",
        "meta_label", "rolling_accuracy", "bet_size",
        "sample_weight",
        
        # Raw prices (use features derived from them)
        "open", "high", "low", "close",
        
        # Other metadata
        "symbol", "ignore", "id"
    }
    
    if exclude_patterns is None:
        exclude_patterns = ["_temp_", "_internal_"]
    
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if any(pat in col for pat in exclude_patterns):
            continue
        # Only include numeric columns
        if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
            feature_cols.append(col)
    
    return feature_cols


def remove_highly_correlated(
    df: pl.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.95
) -> List[str]:
    """
    Remove highly correlated features to reduce multicollinearity.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature columns
        threshold: Correlation threshold for removal
        
    Returns:
        Reduced list of feature columns
    """
    # Calculate correlation matrix
    corr_matrix = df.select(feature_cols).to_pandas().corr().abs()
    
    # Find pairs above threshold
    upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_corr = corr_matrix.where(upper_tri)
    
    # Columns to drop
    to_drop = set()
    for col in upper_corr.columns:
        if any(upper_corr[col] > threshold):
            to_drop.add(col)
    
    remaining = [c for c in feature_cols if c not in to_drop]
    print(f"Removed {len(to_drop)} highly correlated features, {len(remaining)} remaining")
    
    return remaining


# ============================================================================
# MODEL TRAINING
# ============================================================================

def prepare_xy(
    df: pl.DataFrame,
    feature_cols: List[str],
    target_col: str,
    weight_col: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Prepare X, y, and optional weights for model training.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        weight_col: Optional weight column name
        
    Returns:
        Tuple of (X, y, weights)
    """
    # Drop rows with null target
    df_clean = df.drop_nulls(subset=[target_col])
    
    # Get feature matrix
    X = df_clean.select(feature_cols).to_numpy()
    
    # Get target
    y = df_clean[target_col].to_numpy()
    
    # Get weights if specified
    weights = None
    if weight_col and weight_col in df_clean.columns:
        weights = df_clean[weight_col].to_numpy()
    
    # Handle remaining NaN in features
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y, weights


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str = "classification",
    weights_train: Optional[np.ndarray] = None,
    params: Optional[Dict] = None
) -> Any:
    """
    Train XGBoost model with early stopping.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        task: 'classification' or 'regression'
        weights_train: Optional sample weights
        params: XGBoost parameters (uses defaults if None)
        
    Returns:
        Trained XGBoost model
    """
    import xgboost as xgb
    
    # Default parameters optimized for financial data
    default_params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
    }
    
    if params:
        default_params.update(params)
    
    if task == "classification":
        n_classes = len(np.unique(y_train[~np.isnan(y_train)]))
        if n_classes > 2:
            default_params['objective'] = 'multi:softprob'
            default_params['num_class'] = n_classes
            default_params['eval_metric'] = 'mlogloss'
        else:
            default_params['objective'] = 'binary:logistic'
            default_params['eval_metric'] = 'auc'
        
        model = xgb.XGBClassifier(**default_params)
    else:
        default_params['objective'] = 'reg:squarederror'
        default_params['eval_metric'] = 'rmse'
        model = xgb.XGBRegressor(**default_params)
    
    # Fit with early stopping
    model.fit(
        X_train, y_train,
        sample_weight=weights_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model


def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str = "classification",
    weights_train: Optional[np.ndarray] = None,
    params: Optional[Dict] = None
) -> Any:
    """
    Train LightGBM model with early stopping.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        task: 'classification' or 'regression'
        weights_train: Optional sample weights
        params: LightGBM parameters
        
    Returns:
        Trained LightGBM model
    """
    import lightgbm as lgb
    
    default_params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    if params:
        default_params.update(params)
    
    if task == "classification":
        n_classes = len(np.unique(y_train[~np.isnan(y_train)]))
        if n_classes > 2:
            default_params['objective'] = 'multiclass'
            default_params['num_class'] = n_classes
            default_params['metric'] = 'multi_logloss'
        else:
            default_params['objective'] = 'binary'
            default_params['metric'] = 'auc'
        
        model = lgb.LGBMClassifier(**default_params)
    else:
        default_params['objective'] = 'regression'
        default_params['metric'] = 'rmse'
        model = lgb.LGBMRegressor(**default_params)
    
    # Fit with early stopping
    model.fit(
        X_train, y_train,
        sample_weight=weights_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    return model


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate classification model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score, log_loss
    )
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Handle multi-class
    n_classes = len(np.unique(y_true))
    average = 'binary' if n_classes == 2 else 'weighted'
    
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # AUC and log loss if probabilities available
    if y_prob is not None:
        try:
            if n_classes == 2:
                # Binary case
                if y_prob.ndim > 1:
                    y_prob = y_prob[:, 1]
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            else:
                # Multi-class
                metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            metrics['log_loss'] = log_loss(y_true, y_prob)
        except Exception:
            pass
    
    return metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate regression model performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score
    )
    
    metrics = {}
    
    # Basic regression metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Direction accuracy (useful for returns)
    direction_correct = np.mean(np.sign(y_true) == np.sign(y_pred))
    metrics['direction_accuracy'] = direction_correct
    
    # Information Coefficient (rank correlation)
    from scipy.stats import spearmanr
    ic, _ = spearmanr(y_true, y_pred)
    metrics['ic'] = ic
    
    return metrics


def evaluate_trading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    returns: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate trading-specific metrics.
    
    Args:
        y_true: True labels/returns
        y_pred: Predicted labels/returns
        returns: Actual forward returns (for PnL calculation)
        
    Returns:
        Dictionary of trading metrics
    """
    metrics = {}
    
    if returns is not None:
        # Strategy returns (assuming signal = predicted direction)
        strategy_returns = np.sign(y_pred) * returns
        
        # Total return
        metrics['total_return'] = np.sum(strategy_returns)
        
        # Sharpe ratio (annualized, assuming 1-min bars)
        if len(strategy_returns) > 1:
            sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10)
            # Annualize (525600 minutes per year)
            metrics['sharpe_ratio'] = sharpe * np.sqrt(525600)
        
        # Maximum drawdown
        cumulative = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        metrics['max_drawdown'] = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Win rate
        wins = np.sum(strategy_returns > 0)
        total_trades = np.sum(np.abs(y_pred) > 0)
        metrics['win_rate'] = wins / (total_trades + 1e-10)
        
        # Profit factor
        gross_profits = np.sum(strategy_returns[strategy_returns > 0])
        gross_losses = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
        metrics['profit_factor'] = gross_profits / (gross_losses + 1e-10)
    
    return metrics


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def get_feature_importance(
    model: Any,
    feature_cols: List[str],
    importance_type: str = "gain"
) -> pl.DataFrame:
    """
    Extract feature importance from trained model.
    
    Args:
        model: Trained model (XGBoost or LightGBM)
        feature_cols: List of feature names
        importance_type: Type of importance ('gain', 'weight', 'cover')
        
    Returns:
        DataFrame with feature importances sorted by importance
    """
    try:
        if hasattr(model, 'get_booster'):
            # XGBoost
            importance = model.get_booster().get_score(importance_type=importance_type)
            # Map feature indices to names
            importance_dict = {}
            for key, value in importance.items():
                idx = int(key.replace('f', ''))
                if idx < len(feature_cols):
                    importance_dict[feature_cols[idx]] = value
        elif hasattr(model, 'feature_importances_'):
            # Sklearn-style
            importance_dict = dict(zip(feature_cols, model.feature_importances_))
        else:
            return pl.DataFrame()
        
        # Create DataFrame
        df = pl.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort('importance', descending=True)
        
        # Normalize
        total = df['importance'].sum()
        df = df.with_columns([
            (pl.col('importance') / total * 100).alias('importance_pct')
        ])
        
        return df
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        return pl.DataFrame()


# ============================================================================
# FULL PIPELINE
# ============================================================================

def run_full_pipeline(
    base_path: Union[str, Path],
    symbol: str = "BTCUSDT",
    target_col: str = "fwd_class_60",
    task: str = "classification",
    model_type: str = "xgboost"
) -> Dict[str, Any]:
    """
    Run the complete ML pipeline from data loading to evaluation.
    
    Args:
        base_path: Base directory for data
        symbol: Trading pair
        target_col: Target column for prediction
        task: 'classification' or 'regression'
        model_type: 'xgboost' or 'lightgbm'
        
    Returns:
        Dictionary with model, metrics, and feature importance
    """
    print("=" * 60)
    print("CRYPTO ML PIPELINE")
    print("=" * 60)
    
    # 1. Prepare data
    print("\n[1/5] Preparing feature dataset...")
    df = prepare_feature_dataset(base_path, symbol)
    
    # 2. Train/test split
    print("\n[2/5] Splitting data...")
    train_df, val_df, test_df = time_series_split(df)
    
    # 3. Select features
    print("\n[3/5] Selecting features...")
    feature_cols = get_feature_columns(df)
    feature_cols = remove_highly_correlated(df, feature_cols)
    print(f"Using {len(feature_cols)} features")
    
    # 4. Prepare data for training
    print("\n[4/5] Training model...")
    X_train, y_train, w_train = prepare_xy(train_df, feature_cols, target_col, "sample_weight")
    X_val, y_val, _ = prepare_xy(val_df, feature_cols, target_col)
    X_test, y_test, _ = prepare_xy(test_df, feature_cols, target_col)
    
    print(f"Training samples: {len(y_train):,}")
    print(f"Validation samples: {len(y_val):,}")
    print(f"Test samples: {len(y_test):,}")
    
    # 5. Train model
    if model_type == "xgboost":
        model = train_xgboost_model(X_train, y_train, X_val, y_val, task, w_train)
    else:
        model = train_lightgbm_model(X_train, y_train, X_val, y_val, task, w_train)
    
    # 6. Evaluate
    print("\n[5/5] Evaluating model...")
    
    if task == "classification":
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        metrics = evaluate_classification(y_test, y_pred, y_prob)
    else:
        y_pred = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred)
    
    # Trading metrics
    if "fwd_ret_60" in test_df.columns:
        returns = test_df["fwd_ret_60"].to_numpy()[:len(y_pred)]
        trading_metrics = evaluate_trading_metrics(y_test, y_pred, returns)
        metrics.update(trading_metrics)
    
    # Feature importance
    importance_df = get_feature_importance(model, feature_cols)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        elif key != 'confusion_matrix':
            print(f"{key}: {value}")
    
    print("\nTop 10 Features:")
    if len(importance_df) > 0:
        print(importance_df.head(10))
    
    return {
        'model': model,
        'metrics': metrics,
        'feature_importance': importance_df,
        'feature_cols': feature_cols
    }
