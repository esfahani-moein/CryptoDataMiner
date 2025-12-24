import polars as pl
import os

def save_df_to_parquet(df: pl.DataFrame, folder_path: str, file_name: str) -> None:
    """
    Save a Polars DataFrame to Parquet format with ZSTD compression level 15.
    
    Args:
        df: Polars DataFrame to save
        folder_path: Folder path (e.g., 'data_info') where 'request_df.parquet' will be saved
        file_name: Name of the file without extension
    """
    
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name + '.parquet')
    df.write_parquet(file_path, compression='zstd', compression_level=15)
    print(f"DataFrame saved to {file_path}")