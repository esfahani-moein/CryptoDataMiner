import polars as pl

def calculate_vpin(df: pl.DataFrame, bucket_size: float, window_size: int = 50) -> pl.DataFrame:
    """
    Calculates VPIN using Volume-Synchronized Bucketing.
    
    Args:
        df: Input OHLCV Polars DataFrame.
        bucket_size: Total volume required per bucket (e.g., 1/50th of Avg Daily Volume).
        window_size: Number of buckets to use for the rolling VPIN calculation.
    """
    # 1. Schema Validation
    required_cols = {"volume", "taker_buy_base_volume", "open_time"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError("DF missing required columns for VPIN calculation.")

    # 2. Assign bars to Volume Buckets
    # We use cumulative volume to determine the 'Bucket ID'
    vpin_df = df.with_columns([
        (pl.col("volume").cum_sum() // bucket_size).alias("bucket_id"),
        (2 * pl.col("taker_buy_base_volume") - pl.col("volume")).abs().alias("abs_imbalance")
    ])

    # 3. Aggregate Imbalance per Bucket
    # Note: We group by bucket_id to get the total imbalance for that volume unit
    buckets = vpin_df.group_by("bucket_id").agg([
        pl.col("abs_imbalance").sum().alias("bucket_oi"),
        pl.col("open_time").last().alias("timestamp") # To join back to time-series
    ]).sort("bucket_id")

    # 4. Calculate VPIN
    # VPIN = (Sum of Absolute Imbalances over N buckets) / (N * Bucket Size)
    buckets = buckets.with_columns([
        (pl.col("bucket_oi").rolling_sum(window_size=window_size) / 
         (window_size * bucket_size)).alias("vpin")
    ])

    # 5. Join back to original DF
    # This allows the bot to see the VPIN value at every time-stamp/bar.
    return df.join(
        buckets.select(["bucket_id", "vpin"]),
        on=vpin_df["bucket_id"], # Match using the bucket_id from step 2
        how="left"
    ).fill_null(strategy="forward")