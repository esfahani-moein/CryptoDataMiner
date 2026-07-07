import polars as pl

def calculate_cvd(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates the Cumulative Volume Delta (CVD) using Taker/Aggressor volume.
    
    """
    #  Schema Validation
    required_cols = {"volume", "taker_buy_base_volume", "open_time"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"DF missing required columns: {required_cols - set(df.columns)}")

    #  Calculation
    # Delta = Buy Volume (Aggressor) - Sell Volume (Aggressor)
    # Sell Volume = Total Volume - Buy Volume
    # Therefore: Delta = Buy - (Total - Buy) = 2 * Buy - Total
    return df.with_columns([
        (2 * pl.col("taker_buy_base_volume") - pl.col("volume")).alias("delta")
    ]).with_columns([
        pl.col("delta").cum_sum().alias("cvd")
    ])