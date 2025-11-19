import zipfile
import os
import polars as pl
import time 
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from .data_types import get_data_config
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_save import save_df_to_parquet


def process_single_file(zip_path: str, config: dict, local_folder: str, save_folder: str = None) -> pl.DataFrame:
    """
    Process a single ZIP file: unzip, read CSV, and optionally save to Parquet.
    
    Args:
        zip_path: Path to the ZIP file
        config: Data type config
        local_folder: Folder for temp extraction
        save_folder: If provided, save to Parquet and return path; else return DataFrame
    
    Returns:
        DataFrame if not saving, else Parquet file path
    """
    start_time = time.perf_counter()
    try:
        # Unzip to a unique temporary directory per file
        zip_basename = os.path.splitext(os.path.basename(zip_path))[0]
        extract_dir = os.path.join(local_folder, f'temp_extract_{zip_basename}')
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the CSV file (assume one per ZIP)
        csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV found in {zip_path}")
        
        csv_path = os.path.join(extract_dir, csv_files[0])
        # Read with Polars (headerless CSV)
        df = pl.read_csv(
            csv_path,
            has_header=False,
            new_columns=config['columns'],
            dtypes=config['dtypes']
        )
        
        # Cast timestamps 
        if 'timestamp_cols' in config:
            casts = {
                col: pl.col(col).cast(pl.Datetime(time_unit=config.get('time_unit', 'ms')))
                for col in config['timestamp_cols']
            }
            df = df.with_columns(**casts)
        
        # Clean up temp files
        os.remove(csv_path)
        if not os.listdir(extract_dir):
            os.rmdir(extract_dir)
        
        # Optionally save to Parquet
        if save_folder:
            save_df_to_parquet(df, save_folder, zip_basename)
            save_path = os.path.join(save_folder, f"{zip_basename}.parquet")
            print(f"file Saved in: {save_path}")
            
        
        return None
    except Exception as e:
        print(f"Error processing {zip_path}: {e}")
        return None
    finally:
        elapsed_time = time.perf_counter() - start_time
        zip_basename = os.path.splitext(os.path.basename(zip_path))[0]
        print(f"Processing {zip_basename} done in {elapsed_time * 1000:.2f} milliseconds.")



def process_downloaded_files(local_folder: str, files_df: pl.DataFrame, data_type: str, save_folder: str = None, max_workers: int = None):
    """
    Unzip and read CSV files from downloaded ZIPs in parallel using multiple CPUs.
    Optionally save to Parquet.
    
    Args:
        local_folder: Folder containing the ZIP files
        files_df: Polars DataFrame with 'filename' column listing the files to process
        data_type: Type of data (e.g., 'klines') to load config
        save_folder: If provided, save to Parquet and return paths; else return DataFrames
        max_workers: Number of parallel workers (default: None for CPU count)
    
    Returns:
        List of DataFrames or Parquet paths
    """
    config = get_data_config(data_type)
    
    # Get ZIP files from files_df
    zip_files = [
        os.path.join(local_folder, row['filename'])
        for row in files_df.iter_rows(named=True)
        if row['filename'].endswith('.zip')
    ]
    if not zip_files:
        print("No ZIP files found in files_df.")
        return []
    
    desc = "Processing and saving files" if save_folder else "Processing files"
    with tqdm(total=len(zip_files), desc=desc) as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_file, zip_file, config, local_folder, save_folder): zip_file for zip_file in zip_files}
            for future in as_completed(futures):
                pbar.update(1)
    return None