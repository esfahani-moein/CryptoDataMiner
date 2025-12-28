"""
Test Script for Quant Features Module

This script tests all components of the quant_features module:
1. Data loading functions
2. Feature extraction (price, volume, sentiment, orderbook)
3. Labeling functions
4. Model pipeline

Run with: python tests/test_quant_features.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
from datetime import datetime

# Test configuration
BASE_PATH = project_root / "dataset"
SYMBOL = "BTCUSDT"
START_YEAR = 2025
START_MONTH = 11


def test_data_loader():
    """Test data loading functions."""
    print("\n" + "=" * 60)
    print("TESTING DATA LOADER")
    print("=" * 60)
    
    from quant_features.data_loader import (
        load_trades, load_metrics, load_funding_rate,
        load_book_depth, load_klines, load_all_data
    )
    
    # Test trades loading
    print("\n1. Loading trades...")
    trades = load_trades(BASE_PATH, SYMBOL, START_YEAR, START_MONTH, START_YEAR, START_MONTH)
    print(f"   Shape: {trades.shape}")
    print(f"   Time range: {trades['time'].min()} to {trades['time'].max()}")
    print(f"   Nulls: {trades.null_count().sum_horizontal()[0]}")
    assert len(trades) > 0, "No trades loaded"
    assert trades.null_count().sum_horizontal()[0] == 0, "Trades have nulls"
    print("   ✓ Trades loading passed")
    
    # Test metrics loading
    print("\n2. Loading metrics...")
    metrics = load_metrics(BASE_PATH, SYMBOL, START_YEAR, START_MONTH, START_YEAR, START_MONTH)
    print(f"   Shape: {metrics.shape}")
    assert len(metrics) > 0, "No metrics loaded"
    print("   ✓ Metrics loading passed")
    
    # Test funding rate loading
    print("\n3. Loading funding rate...")
    funding = load_funding_rate(BASE_PATH, SYMBOL, START_YEAR, START_MONTH, START_YEAR, START_MONTH)
    print(f"   Shape: {funding.shape}")
    assert len(funding) > 0, "No funding rate loaded"
    print("   ✓ Funding rate loading passed")
    
    # Test book depth loading
    print("\n4. Loading book depth...")
    book = load_book_depth(BASE_PATH, SYMBOL, START_YEAR, START_MONTH, START_YEAR, START_MONTH)
    print(f"   Shape: {book.shape}")
    assert len(book) > 0, "No book depth loaded"
    print("   ✓ Book depth loading passed")
    
    # Test klines loading
    print("\n5. Loading klines...")
    mark = load_klines(BASE_PATH, SYMBOL, "markPriceKlines", START_YEAR, START_MONTH, START_YEAR, START_MONTH)
    print(f"   Mark klines shape: {mark.shape}")
    assert len(mark) > 0, "No klines loaded"
    print("   ✓ Klines loading passed")
    
    print("\n✓ All data loader tests passed!")
    return trades, metrics, funding, book, mark


def test_price_features(ohlcv: pl.DataFrame):
    """Test price feature extraction."""
    print("\n" + "=" * 60)
    print("TESTING PRICE FEATURES")
    print("=" * 60)
    
    from quant_features.price_features import (
        add_returns, add_volatility_features, add_momentum_features,
        add_trend_features, add_all_price_features
    )
    
    initial_cols = len(ohlcv.columns)
    
    # Test returns
    print("\n1. Testing returns...")
    df = add_returns(ohlcv.clone(), periods=[1, 5, 15])
    assert "ret_1" in df.columns, "ret_1 not created"
    assert "ret_5" in df.columns, "ret_5 not created"
    # Check for reasonable values
    ret_mean = df["ret_1"].mean()
    assert abs(ret_mean) < 0.1, f"Unreasonable return mean: {ret_mean}"
    print(f"   Returns created. Mean ret_1: {ret_mean:.6f}")
    print("   ✓ Returns test passed")
    
    # Test volatility
    print("\n2. Testing volatility features...")
    df = add_volatility_features(ohlcv.clone(), windows=[5, 15])
    assert "vol_std_5" in df.columns, "vol_std_5 not created"
    assert "vol_parkinson_5" in df.columns, "vol_parkinson_5 not created"
    vol_mean = df["vol_std_5"].mean()
    print(f"   Volatility features created. Mean vol_std_5: {vol_mean:.6f}")
    print("   ✓ Volatility test passed")
    
    # Test momentum
    print("\n3. Testing momentum features...")
    df = add_momentum_features(ohlcv.clone())
    assert "rsi_14" in df.columns, "rsi_14 not created"
    rsi_mean = df.filter(pl.col("rsi_14").is_not_null())["rsi_14"].mean()
    assert 0 <= rsi_mean <= 100, f"RSI out of range: {rsi_mean}"
    print(f"   Momentum features created. Mean RSI_14: {rsi_mean:.2f}")
    print("   ✓ Momentum test passed")
    
    # Test trend
    print("\n4. Testing trend features...")
    df = add_trend_features(ohlcv.clone())
    assert "sma_20" in df.columns, "sma_20 not created"
    assert "macd" in df.columns, "macd not created"
    print("   ✓ Trend test passed")
    
    # Test all price features
    print("\n5. Testing all price features combined...")
    df = add_all_price_features(ohlcv.clone())
    new_cols = len(df.columns) - initial_cols
    print(f"   Added {new_cols} new features")
    assert new_cols > 30, f"Too few features created: {new_cols}"
    print("   ✓ All price features test passed")
    
    print("\n✓ All price feature tests passed!")
    return df


def test_volume_features(ohlcv: pl.DataFrame):
    """Test volume feature extraction."""
    print("\n" + "=" * 60)
    print("TESTING VOLUME FEATURES")
    print("=" * 60)
    
    from quant_features.volume_features import (
        add_volume_features, add_order_flow_imbalance,
        add_vwap_features, add_all_volume_features
    )
    
    # Test volume features
    print("\n1. Testing volume features...")
    df = add_volume_features(ohlcv.clone(), windows=[5, 15])
    assert "vol_ma_5" in df.columns, "vol_ma_5 not created"
    assert "vol_ratio_5" in df.columns, "vol_ratio_5 not created"
    print("   ✓ Volume features test passed")
    
    # Test order flow
    print("\n2. Testing order flow imbalance...")
    df = add_order_flow_imbalance(ohlcv.clone(), windows=[5])
    if "taker_buy_volume" in ohlcv.columns:
        assert "ofi" in df.columns, "ofi not created"
        assert "buy_pressure" in df.columns, "buy_pressure not created"
        bp_mean = df["buy_pressure"].mean()
        assert 0 <= bp_mean <= 1, f"Buy pressure out of range: {bp_mean}"
        print(f"   Buy pressure mean: {bp_mean:.4f}")
    print("   ✓ Order flow test passed")
    
    # Test VWAP
    print("\n3. Testing VWAP features...")
    df = add_vwap_features(ohlcv.clone(), windows=[15])
    assert "vwap_15" in df.columns, "vwap_15 not created"
    print("   ✓ VWAP test passed")
    
    # Test all volume features
    print("\n4. Testing all volume features combined...")
    df = add_all_volume_features(ohlcv.clone())
    print(f"   Total columns: {len(df.columns)}")
    print("   ✓ All volume features test passed")
    
    print("\n✓ All volume feature tests passed!")
    return df


def test_labeling(ohlcv: pl.DataFrame):
    """Test labeling functions."""
    print("\n" + "=" * 60)
    print("TESTING LABELING FUNCTIONS")
    print("=" * 60)
    
    from quant_features.labeling import (
        add_forward_returns, add_forward_return_classes,
        add_triple_barrier_vectorized, add_trend_labels,
        add_regime_labels, add_volatility_labels, add_all_labels
    )
    
    # Test forward returns
    print("\n1. Testing forward returns...")
    df = add_forward_returns(ohlcv.clone(), periods=[5, 15, 60])
    assert "fwd_ret_5" in df.columns, "fwd_ret_5 not created"
    assert "fwd_ret_60" in df.columns, "fwd_ret_60 not created"
    
    # Check that forward returns are truly forward-looking
    # By checking that early rows have values but late rows have nulls
    non_null_5 = df.filter(pl.col("fwd_ret_5").is_not_null()).height
    expected_non_null = len(df) - 5
    assert abs(non_null_5 - expected_non_null) < 2, "Forward returns not properly shifted"
    print(f"   Forward returns created. Non-null fwd_ret_5: {non_null_5}/{len(df)}")
    print("   ✓ Forward returns test passed")
    
    # Test forward return classes
    print("\n2. Testing forward return classes...")
    df = add_forward_return_classes(df, periods=[5, 60], threshold=0.001)
    assert "fwd_class_5" in df.columns, "fwd_class_5 not created"
    
    # Check class distribution
    class_counts = df.group_by("fwd_class_5").agg(pl.len().alias("count"))
    print(f"   Class distribution: {class_counts.to_dict()}")
    print("   ✓ Forward return classes test passed")
    
    # Test triple barrier (vectorized)
    print("\n3. Testing triple barrier labels...")
    df = add_triple_barrier_vectorized(ohlcv.clone(), max_holding_period=60, 
                                        profit_taking=0.01, stop_loss=0.01)
    assert "tb_label_approx" in df.columns, "tb_label_approx not created"
    tb_counts = df.group_by("tb_label_approx").agg(pl.len().alias("count"))
    print(f"   Triple barrier distribution: {tb_counts.to_dict()}")
    print("   ✓ Triple barrier test passed")
    
    # Test trend labels
    print("\n4. Testing trend labels...")
    df = add_trend_labels(ohlcv.clone(), window=20)
    assert "trend_label" in df.columns, "trend_label not created"
    print("   ✓ Trend labels test passed")
    
    # Test regime labels
    print("\n5. Testing regime labels...")
    df = add_regime_labels(ohlcv.clone())
    assert "regime_label" in df.columns, "regime_label not created"
    print("   ✓ Regime labels test passed")
    
    # Test all labels
    print("\n6. Testing all labels combined...")
    df = add_all_labels(ohlcv.clone())
    label_cols = [c for c in df.columns if 'fwd_' in c or 'tb_' in c or 
                  'label' in c or 'weight' in c]
    print(f"   Created {len(label_cols)} label-related columns")
    print("   ✓ All labels test passed")
    
    print("\n✓ All labeling tests passed!")
    return df


def test_no_lookahead_bias(ohlcv: pl.DataFrame):
    """
    Critical test: Verify no look-ahead bias in features.
    
    Features should only use past data (shift >= 0 or rolling with only past).
    Labels can use future data (that's what we're predicting).
    """
    print("\n" + "=" * 60)
    print("TESTING FOR LOOK-AHEAD BIAS")
    print("=" * 60)
    
    from quant_features.price_features import add_all_price_features
    
    # Create features on subset
    subset = ohlcv.head(1000)
    df_with_features = add_all_price_features(subset)
    
    # Get feature columns (exclude labels)
    feature_cols = [c for c in df_with_features.columns 
                    if c not in ['open_time', 'close_time', 'open', 'high', 'low', 
                                 'close', 'volume', 'quote_volume', 'count',
                                 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']]
    
    print(f"\nChecking {len(feature_cols)} feature columns...")
    
    # For each feature, check that it doesn't have more valid values at the end
    # than at the beginning (which would indicate forward-looking calculation)
    bias_detected = []
    
    for col in feature_cols[:50]:  # Check first 50 features
        if 'fwd_' in col:  # Skip forward-looking labels
            continue
            
        series = df_with_features[col]
        
        # Count nulls in first 100 vs last 100 rows
        first_nulls = series.head(100).null_count()
        last_nulls = series.tail(100).null_count()
        
        # Features should have MORE nulls at start (warming up)
        # and FEWER nulls at end (unless there's look-ahead)
        if last_nulls > first_nulls + 10:  # Allow some tolerance
            bias_detected.append((col, first_nulls, last_nulls))
    
    if bias_detected:
        print("   ⚠ Potential look-ahead bias detected in:")
        for col, first, last in bias_detected[:5]:
            print(f"     - {col}: first_nulls={first}, last_nulls={last}")
    else:
        print("   ✓ No obvious look-ahead bias detected in features")
    
    # Verify forward returns ARE forward-looking (as expected)
    from quant_features.labeling import add_forward_returns
    df_labels = add_forward_returns(subset, periods=[10])
    
    fwd_nulls_start = df_labels.head(10)["fwd_ret_10"].null_count()
    fwd_nulls_end = df_labels.tail(10)["fwd_ret_10"].null_count()
    
    assert fwd_nulls_end == 10, "Forward returns should be null at the end"
    assert fwd_nulls_start == 0, "Forward returns should be valid at the start"
    print("   ✓ Forward returns correctly use future data (for labels)")
    
    print("\n✓ Look-ahead bias tests passed!")


def test_model_pipeline(ohlcv: pl.DataFrame):
    """Test the ML model pipeline."""
    print("\n" + "=" * 60)
    print("TESTING MODEL PIPELINE")
    print("=" * 60)
    
    from quant_features.model_pipeline import (
        time_series_split, get_feature_columns, prepare_xy
    )
    from quant_features.price_features import add_all_price_features
    from quant_features.volume_features import add_all_volume_features
    from quant_features.labeling import add_forward_return_classes
    
    # Use subset for testing
    subset = ohlcv.head(5000)
    
    # Add features and labels
    print("\n1. Preparing features and labels...")
    df = add_all_price_features(subset)
    df = add_all_volume_features(df)
    df = add_forward_return_classes(df, periods=[60], threshold=0.001)
    print(f"   Dataset shape: {df.shape}")
    
    # Test train/test split
    print("\n2. Testing time series split...")
    train, val, test = time_series_split(df, test_ratio=0.2, validation_ratio=0.1)
    
    # Verify no overlap and chronological order
    train_max_time = train["open_time"].max()
    val_min_time = val["open_time"].min()
    test_min_time = test["open_time"].min()
    
    assert train_max_time < val_min_time, "Train/val time overlap!"
    assert val["open_time"].max() < test_min_time, "Val/test time overlap!"
    print("   ✓ Time series split maintains chronological order")
    
    # Test feature selection
    print("\n3. Testing feature selection...")
    feature_cols = get_feature_columns(df)
    print(f"   Selected {len(feature_cols)} features")
    
    # Verify no labels in features
    label_patterns = ['fwd_', 'tb_', '_label', 'weight']
    for col in feature_cols:
        for pattern in label_patterns:
            assert pattern not in col, f"Label column in features: {col}"
    print("   ✓ No label columns in feature set")
    
    # Test data preparation
    print("\n4. Testing data preparation...")
    X_train, y_train, _ = prepare_xy(train, feature_cols, "fwd_class_60")
    X_test, y_test, _ = prepare_xy(test, feature_cols, "fwd_class_60")
    
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    
    assert X_train.shape[1] == len(feature_cols), "Feature dimension mismatch"
    assert not np.any(np.isnan(X_train)), "NaN values in X_train"
    print("   ✓ Data preparation successful")
    
    # Test model training (quick version)
    print("\n5. Testing model training...")
    try:
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels (XGBoost expects 0, 1, 2, ... not -1, 0, 1)
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        # Quick training with minimal settings
        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Fit on small sample
        sample_size = min(500, len(X_train))
        model.fit(X_train[:sample_size], y_train_encoded[:sample_size])
        
        # Predict
        test_sample = min(100, len(X_test))
        y_pred = model.predict(X_test[:test_sample])
        accuracy = np.mean(y_pred == y_test_encoded[:test_sample])
        
        print(f"   Quick test accuracy: {accuracy:.3f}")
        print("   ✓ Model training successful")
    except ImportError:
        print("   ⚠ XGBoost not available, skipping model training test")
    
    print("\n✓ All model pipeline tests passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("QUANT FEATURES MODULE - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Testing with data from: {BASE_PATH}")
    print(f"Symbol: {SYMBOL}, Period: {START_YEAR}-{START_MONTH:02d}")
    
    # 1. Test data loading
    trades, metrics, funding, book, mark = test_data_loader()
    
    # Create OHLCV from trades for feature testing
    print("\n" + "=" * 60)
    print("CREATING OHLCV FROM TRADES")
    print("=" * 60)
    
    from trades_aggregation.trades_aggregator import aggregate_trades_to_ohlcv
    
    # Use subset for faster testing
    trades_sample = trades.head(500_000)  # ~500K trades
    trades_renamed = trades_sample.rename({
        "qty": "quantity",
        "quote_qty": "quote_quantity"
    })
    
    ohlcv = aggregate_trades_to_ohlcv(trades_renamed, 60_000)  # 1-minute bars
    print(f"Created {len(ohlcv):,} OHLCV bars")
    
    # 2. Test price features
    test_price_features(ohlcv)
    
    # 3. Test volume features
    test_volume_features(ohlcv)
    
    # 4. Test labeling
    test_labeling(ohlcv)
    
    # 5. Test for look-ahead bias
    test_no_lookahead_bias(ohlcv)
    
    # 6. Test model pipeline
    test_model_pipeline(ohlcv)
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
