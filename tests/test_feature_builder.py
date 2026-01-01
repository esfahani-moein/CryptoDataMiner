from pathlib import Path
import sys

import polars as pl

# Allow running tests without installing the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_features.feature_builder import (  # noqa: E402
    build_feature_set,
    merge_backward_on_time,
    prepare_ohlcv_from_trades,
)


def test_merge_backward_no_lookahead():
    base = pl.DataFrame({"open_time": [1_000, 2_000, 3_000]})
    feats = pl.DataFrame({"time": [900, 2_500], "val": [1, 2]})

    merged = merge_backward_on_time(base, feats, left_on="open_time", right_on="time")

    assert merged["val"].to_list() == [1, 1, 2]


def test_pipeline_smoke_with_synthetic_trades():
    trades = pl.DataFrame(
        {
            "time": [0, 500, 1_000, 1_500, 2_000],
            "price": [100.0, 101.0, 102.0, 101.5, 103.0],
            "qty": [1.0, 1.0, 1.0, 1.0, 1.0],
            "quote_qty": [100.0, 101.0, 102.0, 101.5, 103.0],
            "is_buyer_maker": [True, False, True, False, True],
        }
    )

    ohlcv = prepare_ohlcv_from_trades(trades, interval_ms=1_000)

    # Minimal preloaded dict to avoid disk I/O in tests
    preloaded = {
        "trades": trades,
        "metrics": None,
        "funding": None,
        "book_depth": None,
        "mark_klines": None,
        "index_klines": None,
        "premium_klines": None,
    }

    feats = build_feature_set(
        base_path=".",
        symbol="TEST",
        start_year=2025,
        start_month=1,
        end_year=2025,
        end_month=1,
        bar_interval_ms=1_000,
        include_orderbook=False,
        include_sentiment=False,
        drop_warmup=False,
        preloaded=preloaded,
    )

    expected_cols = {"open", "high", "low", "close", "volume", "ret_1", "bar_return"}
    assert expected_cols.issubset(set(feats.columns))
    assert len(feats) == len(ohlcv)
