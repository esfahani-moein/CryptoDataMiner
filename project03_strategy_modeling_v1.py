from pathlib import Path
import time
import polars as pl
from trades_aggregation import aggregate_trades_to_ohlcv, TimeInterval
import lightweight_charts as tv_chart


def prepare_data(results, timeframe, limit=500):
    klines = results[timeframe].sort("open_time")
    if limit:
        klines = klines.tail(limit)
    return (
        klines
        .with_columns(pl.col("open_time").dt.strftime("%Y-%m-%d %H:%M:%S").alias("time"))
        .select([
            "time", "open", "high", "low", "close", "volume",
            "ma7", "ma25", "ma99", "rsi", "macd", "macd_signal", "macd_hist",
        ])
        .to_pandas()
    )


def as_line(df, col):
    return df[["time", col]].rename(columns={col: "value"}).to_dict("records")


def as_hist(df, col):
    return df[["time", col]].rename(columns={col: "value"}).to_dict("records")


def compute_indicators(klines: pl.DataFrame) -> pl.DataFrame:
    kl = klines.sort("open_time")
    kl = kl.with_columns([
        pl.col("close").rolling_mean(7).alias("ma7"),
        pl.col("close").rolling_mean(25).alias("ma25"),
        pl.col("close").rolling_mean(99).alias("ma99"),
    ])

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

    return kl.drop(["delta", "gain", "loss", "avg_gain", "avg_loss"])


def make_timeframe_callback(results, series_handles):
    def _cb(chart):
        tf = chart.topbar["timeframe"].value
        df = prepare_data(results, tf)
        if df.empty:
            return
        chart.set(df[["time", "open", "high", "low", "close", "volume"]], True)
        series_handles["ma7"].set(as_line(df, "ma7"))
        series_handles["ma25"].set(as_line(df, "ma25"))
        series_handles["ma99"].set(as_line(df, "ma99"))
        series_handles["rsi"].set(as_line(df, "rsi"))
        series_handles["macd_hist"].set(as_hist(df, "macd_hist"))
        series_handles["macd"].set(as_line(df, "macd"))
        series_handles["macd_signal"].set(as_line(df, "macd_signal"))
    return _cb


if __name__ == "__main__":
    trades_path = Path.cwd() / "dataset" / "dataset_BTCUSDT" / "2025_11" / "trades" / "BTCUSDT-trades-2025-11.parquet"
    trades_df = pl.read_parquet(trades_path)
    trades_df = trades_df.rename({"quote_qty": "quote_quantity", "qty": "quantity"})

    timeframes = {
        "15-minute": 15 * TimeInterval.MINUTE,
        "1-hour": TimeInterval.HOUR,
        "4-hour": 4 * TimeInterval.HOUR,
        "1-day": TimeInterval.DAY,
    }

    t1 = time.time()
    print("Aggregating trades to multiple timeframes...")
    results = {}
    for name, interval in timeframes.items():
        klines = aggregate_trades_to_ohlcv(trades_df, interval)
        klines = klines.with_columns(pl.col("open_time").cast(pl.Datetime("ms")).alias("open_time"))
        klines = klines.with_columns(pl.col("close_time").cast(pl.Datetime("ms")).alias("close_time"))
        klines = compute_indicators(klines)
        results[name] = klines
        elapsed_ms = (time.time() - t1) * 1000
        print(f"  {name:15s}: {len(klines):>6,} candles in {elapsed_ms:>8.2f} ms")

    print(f"Aggregating trades done in {(time.time() - t1) * 1000:.2f} milliseconds.")
    del trades_df

    chart = tv_chart.Chart(toolbox=True)
    chart.legend(True)

    chart.add_pane("rsi", height=100)
    chart.add_pane("macd", height=120)

    ma7_series = chart.add_line(name="MA7", color="blue")
    ma25_series = chart.add_line(name="MA25", color="orange")
    ma99_series = chart.add_line(name="MA99", color="purple")
    rsi_series = chart.add_line(pane="rsi", name="RSI", color="green")
    macd_hist_series = chart.add_histogram(pane="macd", color="gray")
    macd_line_series = chart.add_line(pane="macd", name="MACD", color="blue")
    macd_signal_series = chart.add_line(pane="macd", name="Signal", color="red")

    series_handles = {
        "ma7": ma7_series,
        "ma25": ma25_series,
        "ma99": ma99_series,
        "rsi": rsi_series,
        "macd_hist": macd_hist_series,
        "macd": macd_line_series,
        "macd_signal": macd_signal_series,
    }

    df = prepare_data(results, "15-minute")
    chart.set(df[["time", "open", "high", "low", "close", "volume"]])
    ma7_series.set(as_line(df, "ma7"))
    ma25_series.set(as_line(df, "ma25"))
    ma99_series.set(as_line(df, "ma99"))
    rsi_series.set(as_line(df, "rsi"))
    macd_hist_series.set(as_hist(df, "macd_hist"))
    macd_line_series.set(as_line(df, "macd"))
    macd_signal_series.set(as_line(df, "macd_signal"))

    chart.topbar.switcher("timeframe", tuple(timeframes.keys()), default="15-minute",
                          func=make_timeframe_callback(results, series_handles))

    chart.show(block=True)