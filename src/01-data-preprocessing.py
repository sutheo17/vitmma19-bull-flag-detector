import json
import os
import pandas as pd
import numpy as np
import config
import requests
import zipfile
from utils import get_logger

# Initialize Logger
logger = get_logger()

def download_and_setup_data(url=config.DATABASE_LINK, output_dir=config.DATA_DIR):
    """
    Downloads the ZIP file from the specified URL and extracts it to the target directory.
    Uses the central logger.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Directory created: {output_dir}")

    temp_zip = "temp_dataset_download.zip"
    logger.info(f"Starting download from: {url} ...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = 0
        with open(temp_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        size_mb = total_size / (1024 * 1024)
        logger.info(f"Download complete! Size: {size_mb:.2f} MB")
        logger.info("Unzipping in progress...")

        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            
        logger.info(f"Successfully extracted to: {output_dir}")

    except Exception as e:
        logger.error(f"Error occurred during download/extraction: {e}")
    
    finally:
        if os.path.exists(temp_zip):
            os.remove(temp_zip)
            logger.info("Temporary files deleted.")

def find_csv_file(target_filename, search_dir):
    """Locates the file in the specified directory."""
    exact_path = os.path.join(search_dir, target_filename)
    if os.path.exists(exact_path):
        return exact_path
    
    simple_name = target_filename.split('-')[-1] if '-' in target_filename else target_filename
    
    try:
        files_in_dir = os.listdir(search_dir)
    except FileNotFoundError:
        return None

    for f in files_in_dir:
        if f.endswith(simple_name) or simple_name in f:
            return os.path.join(search_dir, f)
    return None

def interpolate_sequence(seq, target_length):
    """Resizes the sequence using linear interpolation."""
    if len(seq) == 0: return np.zeros((target_length, seq.shape[1]))
    result = []
    for col in range(seq.shape[1]):
        original = seq[:, col]
        x_old = np.linspace(0, 1, len(original))
        x_new = np.linspace(0, 1, target_length)
        result.append(np.interp(x_new, x_old, original))
    return np.stack(result, axis=1)

def get_label_id(label_str):
    """Mapping labels to IDs (0-5)."""
    if "Bullish" in label_str:
        if "Pennant" in label_str: return 1
        if "Wedge" in label_str:   return 2
        return 0 # Normal
    
    elif "Bearish" in label_str:
        if "Pennant" in label_str: return 4
        if "Wedge" in label_str:   return 5
        return 3 # Normal
        
    return -1

def robust_parse_dates(df):
    """Improved date parsing for DataFrame columns."""
    col_map = {c.lower(): c for c in df.columns}
    ts_col = col_map.get('timestamp') or col_map.get('time') or col_map.get('date')
    
    if not ts_col:
        return df

    temp_col = pd.to_numeric(df[ts_col], errors='coerce')
    
    if temp_col.notna().sum() > len(df) * 0.8:
        temp_col = temp_col.ffill().bfill()
        first_val = temp_col.iloc[0]
        if first_val > 3e10:
            df[ts_col] = pd.to_datetime(temp_col, unit='ms')
        else:
            df[ts_col] = pd.to_datetime(temp_col, unit='s')
    else:
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

    df = df.set_index(ts_col).sort_index()
    return df

def parse_annotation_time(time_val):
    """Safe parsing of a single JSON time value."""
    if time_val is None:
        return None
    try:
        numeric_ts = float(time_val)
        if numeric_ts > 3e10: 
            return pd.to_datetime(numeric_ts, unit='ms')
        else:
            return pd.to_datetime(numeric_ts, unit='s')
    except (ValueError, TypeError):
        return pd.to_datetime(time_val)

def find_optimal_pole_start(df, flag_start_time, label_id, max_lookback=60):
    """
    Finds the optimal pole start (impulse start) before the flag consolidation
    based on slope maximization.
    """
    if flag_start_time not in df.index:
        try:
            loc_idx = df.index.get_loc(flag_start_time, method='nearest')
        except:
            return flag_start_time
    else:
        loc_idx = df.index.get_indexer([flag_start_time])[0]

    # If too close to the beginning of data
    if loc_idx < 5:
        return flag_start_time

    start_search_idx = max(0, loc_idx - max_lookback)
    
    # window_df rows: [candidate_pole_start ... flag_start]
    window_df = df.iloc[start_search_idx : loc_idx + 1]
    
    if len(window_df) < 2:
        return flag_start_time

    # Anchor points (start of the flag)
    anchor_high = window_df.iloc[-1]['High']
    anchor_low  = window_df.iloc[-1]['Low']

    best_start_time = flag_start_time
    max_slope = -1.0 

    # 0,1,2 = Bullish (Prev trend was Up), 3,4,5 = Bearish (Prev trend was Down)
    is_bullish = label_id in [0, 1, 2]
    
    flag_idx_in_window = len(window_df) - 1
    
    # Iterate backwards to find best start
    for i in range(len(window_df) - 1):
        candidate_bar = window_df.iloc[i]
        steps_distance = flag_idx_in_window - i
        
        if steps_distance == 0: continue

        slope = 0
        if is_bullish:
            # Bullish: How much did it rise? (Flag High - Candidate Low)
            price_diff = anchor_high - candidate_bar['Low']
            if price_diff > 0:
                slope = price_diff / steps_distance
        else:
            # Bearish: How much did it fall? (Candidate High - Flag Low)
            price_diff = candidate_bar['High'] - anchor_low
            if price_diff > 0:
                slope = price_diff / steps_distance
        
        if slope > max_slope:
            max_slope = slope
            best_start_time = candidate_bar.name # Timestamp
            
    return best_start_time

def main():
    if config.DOWNLOAD_FROM_ONEDRIVE:
        download_and_setup_data()
    else:
        logger.info("Skipping download (DOWNLOAD_FROM_ONEDRIVE=False). Using local data.")
    
    if not os.path.exists(config.DATA_DIR):
        logger.error(f"Data directory not found: {config.DATA_DIR}")
        return

    all_X = []
    all_y = []
    
    subdirs = [d for d in os.listdir(config.DATA_DIR) if os.path.isdir(os.path.join(config.DATA_DIR, d))]
    total_tasks_processed = 0

    for subdir in subdirs:
        current_dir = os.path.join(config.DATA_DIR, subdir)
        
        try:
            files_in_subdir = os.listdir(current_dir)
            json_filename = next((f for f in files_in_subdir if f.endswith('.json')), None)
            
            if json_filename is None:
                continue
                
            labels_path = os.path.join(current_dir, json_filename)
            
        except Exception as e:
            logger.warning(f"Error accessing folder {subdir}: {e}")
            continue

        logger.info(f"Processing folder: {subdir} (Found JSON: {json_filename})")
        
        try:
            with open(labels_path, 'r') as f:
                tasks = json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON {labels_path}: {e}")
            continue

        for task in tasks:
            original_filename = task.get('file_upload')
            if not original_filename: continue
                
            csv_path = find_csv_file(original_filename, current_dir)
            if not csv_path: continue
            
            try:
                df = pd.read_csv(csv_path)
                # Ensure column names are standardized
                df.columns = [c.capitalize() for c in df.columns] 
                df = robust_parse_dates(df)
                
            except Exception as e:
                logger.warning(f"Skipping {csv_path} due to load error: {e}")
                continue

            annotations = task.get('annotations', [])
            
            for ann in annotations:
                results = ann.get('result', [])
                for res in results:
                    val = res.get('value', {})
                    labels = val.get('timeserieslabels', [])
                    
                    if not labels: continue
                    
                    label_id = get_label_id(labels[0])
                    if label_id == -1: continue
                    
                    # 1. Parse original timestamps
                    flag_start_time = parse_annotation_time(val.get('start'))
                    flag_end_time = parse_annotation_time(val.get('end'))
                    
                    if flag_start_time is None or flag_end_time is None:
                        continue
                    
                    # 2. Find Optimal Pole Start (Context Expansion)
                    pole_start_time = find_optimal_pole_start(df, flag_start_time, label_id, max_lookback=60)
                    
                    # 3. Slice window: From Pole Start to Flag End
                    mask = (df.index >= pole_start_time) & (df.index <= flag_end_time)
                    
                    try:
                        window = df.loc[mask, ['Open', 'High', 'Low', 'Close']].values
                    except KeyError:
                        try:
                            window = df.loc[mask, ['open', 'high', 'low', 'close']].values
                        except KeyError:
                            continue

                    if len(window) > 5:
                        # 4. Normalize
                        # window[0] is now the start of the pole.
                        # This preserves the trend direction (Pos for Bull, Neg for Bear)
                        window_norm = window / window[0] - 1.0
                        
                        # 5. Interpolate to fixed size
                        window_resized = interpolate_sequence(window_norm, config.SEQ_LENGTH)
                        
                        all_X.append(window_resized)
                        all_y.append(label_id)
            
            total_tasks_processed += 1

    if len(all_X) == 0:
        logger.error("No valid samples extracted.")
        return

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    
    logger.info("Processing Complete!")
    logger.info(f"Total Samples: {len(X)}")
    
    np.savez(config.PROCESSED_DATA_PATH, X=X, y=y)
    logger.info(f"Saved to: {config.PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()