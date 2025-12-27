import polars as pl

def save_df_to_parquet(df: pl.DataFrame, file_path) -> None:
    """
    Save a Polars DataFrame to Parquet format with ZSTD compression level 18.
    
    Args:
        df: Polars DataFrame to save
        file_path: Full file path (including filename) where the DataFrame will be saved
    """
    
    df.write_parquet(file_path, compression='zstd', compression_level=18)
    # df.write_parquet(file_path, compression='brotli', compression_level=9)
    