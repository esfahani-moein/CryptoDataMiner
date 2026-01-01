"""
Feature Builder

Single-entry functional pipeline to load raw Binance-style datasets, run
point-in-time merges, and generate feature + label tables with Polars.

Design goals:
- No classes; pure functions for readability and easy extension
- Strict avoidance of look-ahead bias (backward asof joins only)
- Reuse existing building blocks (trades_aggregation, price/volume/orderbook
  features, sentiment features, labeling)
- Small helpers for data quality checks and warmup trimming
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import polars as pl

from quant_features.data_loader import (
    load_book_depth,
    load_funding_rate,
    load_klines,
    load_metrics,
    load_trades,
)
from quant_features.labeling import add_all_labels
from quant_features.orderbook_features import add_depth_features_from_long
from quant_features.price_features import add_all_price_features
from quant_features.sentiment_features import add_all_sentiment_features
from quant_features.volume_features import add_all_volume_features
from trades_aggregation.trades_aggregator import TimeInterval, aggregate_trades_to_ohlcv

DataDict = Dict[str, Optional[pl.DataFrame]]


# ---------------------------------------------------------------------------
# Lightweight QC helpers
# ---------------------------------------------------------------------------

def summarize_frame(name: str, df: Optional[pl.DataFrame], time_col: Optional[str] = None) -> Dict:
    """Return basic shape/null summary for a frame (safe for None)."""
    if df is None:
        return {"name": name, "rows": 0, "cols": 0, "nulls": {}, "time_range": None}

    nulls = df.null_count().to_dict(as_series=False)
    time_range = None
    if time_col and time_col in df.columns:
        time_range = (df[time_col].min(), df[time_col].max())

    return {
        "name": name,
        "rows": len(df),
        "cols": len(df.columns),
        "nulls": nulls,
        "time_range": time_range,
    }


def summarize_sources(data: DataDict) -> pl.DataFrame:
    """Tabular summary for already-loaded dataframes."""
    summaries = [
        summarize_frame("trades", data.get("trades"), "time"),
        summarize_frame("metrics", data.get("metrics"), "time"),
        summarize_frame("funding", data.get("funding"), "time"),
        summarize_frame("book_depth", data.get("book_depth"), "time"),
        summarize_frame("mark_klines", data.get("mark_klines"), "open_time"),
        summarize_frame("index_klines", data.get("index_klines"), "open_time"),
        summarize_frame("premium_klines", data.get("premium_klines"), "open_time"),
    ]
    return pl.DataFrame(summaries)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data_range(
    base_path: Union[str, Path],
    symbol: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    load_all: bool = True,
) -> DataDict:
    """Load all supported sources for a date range (errors tolerated individually)."""
    base_path = Path(base_path)

    def _safe_load(loader):
        try:
            return loader(base_path, symbol, start_year, start_month, end_year, end_month)
        except FileNotFoundError:
            return None

    data: DataDict = {
        "trades": _safe_load(load_trades),
        "metrics": _safe_load(load_metrics) if load_all else None,
        "funding": _safe_load(load_funding_rate) if load_all else None,
        "book_depth": _safe_load(load_book_depth) if load_all else None,
        "mark_klines": _safe_load(lambda *args, **kwargs: load_klines(*args, kline_type="markPriceKlines", **kwargs)),
        "index_klines": _safe_load(lambda *args, **kwargs: load_klines(*args, kline_type="indexPriceKlines", **kwargs)),
        "premium_klines": _safe_load(lambda *args, **kwargs: load_klines(*args, kline_type="premiumIndexKlines", **kwargs)),
    }
    return data


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def merge_backward_on_time(
    base_df: pl.DataFrame,
    feat_df: Optional[pl.DataFrame],
    left_on: str,
    right_on: str,
    suffix: str = "",
) -> pl.DataFrame:
    """Backward asof join helper to enforce point-in-time alignment."""
    if feat_df is None or len(feat_df) == 0:
        return base_df

    # sort only the necessary columns to avoid shuffles
    left = base_df.sort(left_on)
    right = feat_df.sort(right_on)

    joined = left.join_asof(right, left_on=left_on, right_on=right_on, strategy="backward", suffix=suffix)
    return joined


def prepare_ohlcv_from_trades(trades: pl.DataFrame, interval_ms: int) -> pl.DataFrame:
    """Aggregate trades to OHLCV using the existing fast aggregator."""
    trades_renamed = trades.rename({"qty": "quantity", "quote_qty": "quote_quantity"}) if "qty" in trades.columns else trades
    ohlcv = aggregate_trades_to_ohlcv(trades_renamed, interval_ms, ensure_sorted=True)
    return ohlcv


def attach_reference_prices(ohlcv: pl.DataFrame, data: DataDict) -> pl.DataFrame:
    """Join mark/index/premium reference prices without look-ahead."""
    result = ohlcv

    mark_df = data.get("mark_klines")
    if mark_df is not None:
        mark_feats = mark_df.select([
            pl.col("open_time").alias("mark_time"),
            pl.col("close").alias("mark_price"),
            ((pl.col("high") - pl.col("low")) / pl.col("close") * 100).alias("mark_range_pct"),
        ])
        result = merge_backward_on_time(result, mark_feats, "open_time", "mark_time", suffix="_mark")
        if "mark_price" in result.columns:
            result = result.with_columns([
                ((pl.col("close") - pl.col("mark_price")) / (pl.col("mark_price") + 1e-12) * 10_000).alias("basis_bps"),
            ])

    index_df = data.get("index_klines")
    if index_df is not None:
        idx_feats = index_df.select([
            pl.col("open_time").alias("index_time"),
            pl.col("close").alias("index_price"),
        ])
        result = merge_backward_on_time(result, idx_feats, "open_time", "index_time", suffix="_index")
        if "index_price" in result.columns:
            result = result.with_columns([
                ((pl.col("close") - pl.col("index_price")) / (pl.col("index_price") + 1e-12) * 10_000).alias("premium_bps"),
            ])

    prem_df = data.get("premium_klines")
    if prem_df is not None:
        prem_feats = prem_df.select([
            pl.col("open_time").alias("premium_time"),
            pl.col("close").alias("premium_index"),
        ])
        result = merge_backward_on_time(result, prem_feats, "open_time", "premium_time", suffix="_premium")

    return result


def attach_sentiment_inputs(ohlcv: pl.DataFrame, data: DataDict) -> pl.DataFrame:
    """Merge metrics and funding series before feature calculation."""
    result = ohlcv

    metrics_df = data.get("metrics")
    if metrics_df is not None:
        result = merge_backward_on_time(result, metrics_df, "open_time", "time", suffix="_metrics")

    funding_df = data.get("funding")
    if funding_df is not None:
        result = merge_backward_on_time(result, funding_df, "open_time", "time", suffix="_funding")

    return result


def attach_orderbook_features(ohlcv: pl.DataFrame, data: DataDict, windows: Optional[list[int]] = None) -> pl.DataFrame:
    """Calculate and merge orderbook depth features if available."""
    book_df = data.get("book_depth")
    if book_df is None:
        return ohlcv

    windows = windows or [6, 12, 24]
    depth_feats = add_depth_features_from_long(book_df, windows)
    merged = merge_backward_on_time(ohlcv, depth_feats, "open_time", "time", suffix="_ob")
    return merged


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def build_feature_set(
    base_path: Union[str, Path],
    symbol: str = "BTCUSDT",
    start_year: int = 2025,
    start_month: int = 11,
    end_year: int = 2025,
    end_month: int = 11,
    bar_interval_ms: int = TimeInterval.MINUTE,
    include_orderbook: bool = True,
    include_sentiment: bool = True,
    drop_warmup: bool = True,
    preloaded: Optional[DataDict] = None,
) -> pl.DataFrame:
    """End-to-end feature + label builder.

    All joins use backward asof to prevent look-ahead. Rolling features rely on
    past values only. Warmup trimming removes the early rows where rolling
    windows have insufficient history.
    """
    data = preloaded if preloaded is not None else load_data_range(
        base_path,
        symbol,
        start_year,
        start_month,
        end_year,
        end_month,
        load_all=True,
    )

    trades = data.get("trades")
    if trades is None or len(trades) == 0:
        raise ValueError("Trades data is required to build features")

    ohlcv = prepare_ohlcv_from_trades(trades, bar_interval_ms)
    ohlcv = attach_reference_prices(ohlcv, data)
    ohlcv = attach_sentiment_inputs(ohlcv, data) if include_sentiment else ohlcv
    ohlcv = attach_orderbook_features(ohlcv, data) if include_orderbook else ohlcv

    # Feature blocks
    feats = add_all_price_features(ohlcv)
    feats = add_all_volume_features(feats)
    if include_sentiment:
        feats = add_all_sentiment_features(feats)

    feats = add_all_labels(feats)

    # Trim warmup to remove leading null-heavy rows
    if drop_warmup:
        warmup = 250  # covers the largest 240-bar windows plus buffer
        feats = feats.slice(warmup)

    return feats.sort("open_time")


__all__ = [
    "build_feature_set",
    "load_data_range",
    "summarize_sources",
    "prepare_ohlcv_from_trades",
    "merge_backward_on_time",
    "attach_reference_prices",
    "attach_sentiment_inputs",
    "attach_orderbook_features",
]
