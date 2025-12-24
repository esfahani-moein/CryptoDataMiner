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


def process_single_file(zip_path: str, config: dict, local_folder: str, save_folder: str = None) -> bool:
    """
    Process a single ZIP file: unzip, read CSV, and optionally save to Parquet.
    
    Args:
        zip_path: Path to the ZIP file
        config: Data type config
        local_folder: Folder for temp extraction
        save_folder: If provided, save to Parquet and return path; else return DataFrame
    
    Returns:
        True if successful, False otherwise
    """
    start_time = time.perf_counter()
    zip_basename = os.path.splitext(os.path.basename(zip_path))[0]
    
    try:
        # Unzip to a unique temporary directory per file
        extract_dir = os.path.join(local_folder, f'temp_extract_{zip_basename}')
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the CSV file (assume one per ZIP)
        csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV found in {zip_path}")
        
        csv_path = os.path.join(extract_dir, csv_files[0])
        
        # Detect if CSV has headers by reading first line
        with open(csv_path, 'r') as f:
            first_line = f.readline().strip()
            # Check if first line looks like a header (contains non-numeric characters in expected positions)
            has_header = any(field.replace('_', '').replace('.', '').isalpha() for field in first_line.split(','))
        
        if has_header:
            # Read with header and create column mapping
            df = pl.read_csv(csv_path, has_header=True)
            
            # Create simple column mapping based on common patterns
            column_mapping = {}
            for actual_col in df.columns:
                norm = actual_col.lower().strip()
                if norm in ['id', 'trade_id']:
                    column_mapping[actual_col] = 'trade_id'
                elif norm == 'price':
                    column_mapping[actual_col] = 'price'
                elif norm in ['qty', 'quantity']:
                    column_mapping[actual_col] = 'quantity'
                elif norm in ['quote_qty', 'quote_quantity']:
                    column_mapping[actual_col] = 'quote_quantity'
                elif norm == 'time':
                    column_mapping[actual_col] = 'time'
                elif norm in ['is_buyer_maker', 'isbuyermaker']:
                    column_mapping[actual_col] = 'is_buyer_maker'
                elif norm in config['columns']:
                    column_mapping[actual_col] = norm
            
            df = df.rename(column_mapping)
        else:
            # Read without header
            df = pl.read_csv(
                csv_path,
                has_header=False,
                new_columns=config['columns']
            )
        
        # Cast to correct dtypes
        for col, dtype in config['dtypes'].items():
            if col in df.columns:
                try:
                    df = df.with_columns(pl.col(col).cast(dtype))
                except Exception as e:
                    print(f"Warning: Could not cast {col} to {dtype}: {e}")
        
        # Cast timestamps 
        if 'timestamp_cols' in config:
            for col in config['timestamp_cols']:
                if col in df.columns:
                    try:
                        df = df.with_columns(
                            pl.col(col).cast(pl.Datetime(time_unit=config.get('time_unit', 'ms')))
                        )
                    except Exception as e:
                        print(f"Warning: Could not cast timestamp {col}: {e}")
        
        # Clean up temp files and directory
        if os.path.exists(csv_path):
            os.remove(csv_path)
        
        # Remove temp extract directory
        if os.path.exists(extract_dir):
            import shutil
            try:
                shutil.rmtree(extract_dir)
            except Exception as cleanup_error:
                print(f"Warning: Could not remove temp directory {extract_dir}: {cleanup_error}")
        
        # Optionally save to Parquet
        if save_folder:
            save_df_to_parquet(df, save_folder, zip_basename)
            save_path = os.path.join(save_folder, f"{zip_basename}.parquet")
            elapsed_time = time.perf_counter() - start_time
            print(f"✓ {zip_basename} processed and saved in {elapsed_time:.2f}s ({len(df):,} rows)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing {zip_basename}: {e}")
        import traceback
        traceback.print_exc()
        return False



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
        Dict with success/failure counts
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
        return {'total': 0, 'successful': 0, 'failed': 0}
    
    desc = "Processing and saving files" if save_folder else "Processing files"
    successful = 0
    failed = 0
    
    with tqdm(total=len(zip_files), desc=desc) as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_file, zip_file, config, local_folder, save_folder): zip_file for zip_file in zip_files}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Exception in worker: {e}")
                    failed += 1
                pbar.update(1)
    
    return {'total': len(zip_files), 'successful': successful, 'failed': failed}