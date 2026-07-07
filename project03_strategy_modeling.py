from pathlib import Path
import polars as pl
from trades_aggregation import (
    aggregate_trades_to_ohlcv,
    aggregate_trades_from_file,
    validate_aggregation_accuracy,
    print_validation_results,
    TimeInterval
)
import lightweight_charts as tv_chart
import time
from data_aggregate import normalize_timestamp_to_datetime 
import pandas as pd
import numpy as np



def prepare_data(results, timeframe):
    klines = results[timeframe].sort("open_time")
    # ensure ascending order and limit to last 500 candles
   
    # out1 = (
    #     klines
    #     .select(["open_time", "open", "high", "low", "close", "volume"])
    #     .with_columns(pl.col("open_time").dt.strftime("%Y-%m-%d %H:%M:%S").alias("time"))
    #     .select(["time", "open", "high", "low", "close", "volume"])
    #     .to_pandas()
    # )

    # out = (
    # klines.select(["open_time", "open", "high", "low", "close", "volume", "ma7","ma25","ma99","rsi","macd","macd_signal","macd_hist"])
    #       .with_columns(pl.col("open_time").dt.strftime("%Y-%m-%d %H:%M:%S").alias("time"))
    #       .select(["time","open","high","low","close","volume","ma7","ma25","ma99","rsi","macd","macd_signal","macd_hist"])
    #       .to_pandas()
    # )
    return (
        klines
        .with_columns(pl.col("open_time").dt.strftime("%Y-%m-%d %H:%M:%S").alias("time"))
        .select([
            "time","open","high","low","close","volume",
            "ma7","ma25","ma99","rsi","macd","macd_signal","macd_hist",
        ])
        .to_pandas()
    )

def as_line(df, col):  # for line series
    return df[["time", col]].rename(columns={col: "value"}).to_dict("records")
def as_hist(df, col):  # for hist series
    return df[["time", col]].rename(columns={col: "value"}).to_dict("records")


def compute_indicators(klines: pl.DataFrame) -> pl.DataFrame:
    kl = klines.sort("open_time")
    # SMAs (Polars rolling)
    kl = kl.with_columns([
        pl.col("close").rolling_mean(7).alias("ma7"),
        pl.col("close").rolling_mean(25).alias("ma25"),
        pl.col("close").rolling_mean(99).alias("ma99"),
    ])

    # RSI(14) (Polars rolling average of gains/losses)
    n = 14
    delta = pl.col("close").diff().fill_null(0)
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)
    kl = kl.with_columns([
        delta.alias("delta"),
        gain.alias("gain"),
        loss.alias("loss"),
        gain.rolling_mean(n).alias("avg_gain"),
        loss.rolling_mean(n).alias("avg_loss"),
    ])
    kl = kl.with_columns(
        pl.when(pl.col("avg_loss") == 0)
          .then(100.0)
          .otherwise(100.0 - (100.0 / (1.0 + (pl.col("avg_gain") / pl.col("avg_loss")))))
          .alias("rsi")
    )

    # MACD (pandas EWM fallback)
    pdf_close = kl.select("close").to_pandas()["close"]
    ema12 = pdf_close.ewm(span=12, adjust=False).mean()
    ema26 = pdf_close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal

    kl = kl.with_columns([
        pl.Series("macd", macd_line.values).cast(pl.Float64),
        pl.Series("macd_signal", signal.values).cast(pl.Float64),
        pl.Series("macd_hist", hist.values).cast(pl.Float64),
    ])

    # drop intermediate helper cols if you want
    return kl.drop(["delta", "gain", "loss", "avg_gain", "avg_loss"])


def klines_to_indicators(klines):
    # Calculate indicators (MA, RSI, MACD) using polars
    klines = klines.sort("open_time")
    # Moving Averages
    klines = klines.with_columns(
        pl.col("close").rolling_mean(window_size=7).alias("ma7"),
        pl.col("close").rolling_mean(window_size=25).alias("ma25"),
        pl.col("close").rolling_mean(window_size=99).alias("ma99"),
    )
    # RSI calculation
    # RSI with divide-by-zero guard
    klines = klines.with_columns(
        pl.when(pl.col("avg_loss") == 0)
        .then(100.0)
        .otherwise(100.0 - (100.0 / (1.0 + (pl.col("avg_gain") / pl.col("avg_loss")))))
        .alias("rsi")
    )
    # MACD calculation
    ema12 = klines["close"].ewm(span=12).mean()
    ema26 = klines["close"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    macd_hist = macd - signal
    klines = klines.with_columns([
        macd.alias("macd"),
        signal.alias("macd_signal"),
        macd_hist.alias("macd_hist")
    ])
    return klines


def on_timeframe_selection(chart):
    tf = chart.topbar["timeframe"].value
    df = prepare_data(results, tf)
    if df.empty:
        return
    chart.set(df[["time","open","high","low","close","volume"]], True)
    ma7_series.set(as_line(df, "ma7"))
    ma25_series.set(as_line(df, "ma25"))
    ma99_series.set(as_line(df, "ma99"))
    rsi_series.set(as_line(df, "rsi"))
    macd_hist_series.set(as_hist(df, "macd_hist"))
    macd_line_series.set(as_line(df, "macd"))
    macd_signal_series.set(as_line(df, "macd_signal"))

if __name__ == '__main__':
    # Load and aggregate data (same as notebook)
    trades_path = Path.cwd() / "dataset" / "dataset_BTCUSDT" / "2025_11" / "trades" / "BTCUSDT-trades-2025-11.parquet"
    trades_df = pl.read_parquet(trades_path)
    trades_df = trades_df.rename({"quote_qty": "quote_quantity",
                                   "qty": "quantity"})
    
    timeframes = {
        # "1-second": TimeInterval.SECOND,
        # "1-minute": TimeInterval.MINUTE,
        # "5-minute": 5 * TimeInterval.MINUTE,
        "15-minute": 15 * TimeInterval.MINUTE,
        "1-hour": TimeInterval.HOUR,
        "4-hour": 4 * TimeInterval.HOUR,
        "1-day": TimeInterval.DAY,
    }

    t1 = time.time()
    print("Aggregating trades to multiple timeframes...")
    results = {}
    for name, interval in timeframes.items():
        t2 = time.time()
        klines = aggregate_trades_to_ohlcv(trades_df, interval) 
        klines = klines.with_columns(pl.col("open_time").cast(pl.Datetime("ms")).alias("open_time"))
        klines = klines.with_columns(pl.col("close_time").cast(pl.Datetime("ms")).alias("close_time"))
        klines = compute_indicators(klines)
        results[name] = klines
        elapsed_ms = (time.time() - t1) * 1000
        print(f"  {name:15s}: {len(klines):>6,} candles in {elapsed_ms:>8.2f} ms")
    
    t3 = time.time()
    print(f"Aggregating trades done in {(t3 - t1) * 1000:.2f} milliseconds.")
    
    del trades_df  # Free memory

    ma7_series = chart.line(name="MA7", color="blue")
    # same for ma25, ma99
    chart.new_pane("rsi", height=100)
    rsi_series = chart.line(pane="rsi", name="RSI", color="green")
    chart.new_pane("macd", height=120)
    macd_hist_series = chart.hist(pane="macd", color="gray")
    macd_line_series = chart.line(pane="macd", name="MACD", color="blue")
    macd_signal_series = chart.line(pane="macd", name="Signal", color="red")



    # Create chart with switcher for timeframes
    chart = tv_chart.Chart(toolbox=True)
    chart.legend(True)
    
    chart.topbar.switcher('timeframe', tuple(timeframes.keys()), default='15-minute',
                          func=on_timeframe_selection)
    
    # Set initial data
    df = prepare_data(results, '15-minute')

    # chart.set(df)

    # after computing indicators and producing `df` (pandas)
    chart.set(df[["time","open","high","low","close","volume"]])
    ma7_series.set(as_line(df, "ma7"))
    ma25_series.set(as_line(df, "ma25"))
    ma99_series.set(as_line(df, "ma99"))
    rsi_series.set(as_line(df, "rsi"))
    macd_hist_series.set(as_hist(df, "macd_hist"))
    macd_line_series.set(as_line(df, "macd"))
    macd_signal_series.set(as_line(df, "macd_signal"))

    # overlay MAs on price pane (API name may differ; adapt to library)
    chart.line(df[["time","ma7"]].to_dict("records"), name="MA7", color="blue")
    chart.line(df[["time","ma25"]].to_dict("records"), name="MA25", color="orange")
    chart.line(df[["time","ma99"]].to_dict("records"), name="MA99", color="purple")

    # RSI pane
    chart.new_pane("rsi", height=100)                 # check exact method name
    chart.line(df[["time","rsi"]].to_dict("records"), pane="rsi", name="RSI", color="green")

    # MACD pane (histogram + lines)
    chart.new_pane("macd", height=120)
    chart.hist(df[["time","macd_hist"]].to_dict("records"), pane="macd", color="gray")
    chart.line(df[["time","macd"]].to_dict("records"), pane="macd", name="MACD", color="blue")
    chart.line(df[["time","macd_signal"]].to_dict("records"), pane="macd", name="Signal", color="red")
    
    chart.show(block=True)