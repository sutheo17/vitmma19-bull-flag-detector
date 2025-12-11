import json
import os
import pandas as pd
import numpy as np
import random
import config

def find_csv_file(target_filename, search_dir):
    """
    Searches for a file in a specific directory, ignoring the Label Studio UUID prefix.
    Example: '6252ed62-XAUUSD_5min.csv' -> Finds 'XAUUSD_5min.csv'.
    """
    # 1. Try exact match
    exact_path = os.path.join(search_dir, target_filename)
    if os.path.exists(exact_path):
        return exact_path
    
    # 2. Try matching by original filename (ignoring UUID prefix)
    # Assumes the generated ID is separated by a hyphen '-'
    simple_name = target_filename.split('-')[-1] if '-' in target_filename else target_filename
    
    # List files in the specific search directory
    try:
        files_in_dir = os.listdir(search_dir)
    except FileNotFoundError:
        return None

    for f in files_in_dir:
        if f.endswith(simple_name) or simple_name in f:
            return os.path.join(search_dir, f)
            
    return None

def interpolate_sequence(seq, target_length):
    """
    Resizes the sequence to a fixed length (SEQ_LENGTH) using linear interpolation.
    """
    if len(seq) == 0: return np.zeros((target_length, seq.shape[1]))
    result = []
    for col in range(seq.shape[1]):
        original = seq[:, col]
        x_old = np.linspace(0, 1, len(original))
        x_new = np.linspace(0, 1, target_length)
        result.append(np.interp(x_new, x_old, original))
    return np.stack(result, axis=1)

def get_label_id(label_str):
    """Returns numeric ID based on text label (7 classes)."""
    # Bullish types
    if "Bullish" in label_str:
        if "Pennant" in label_str: return 2
        if "Wedge" in label_str:   return 3
        return 1 # Normal
    
    # Bearish types
    elif "Bearish" in label_str:
        if "Pennant" in label_str: return 5
        if "Wedge" in label_str:   return 6
        return 4 # Normal
        
    return -1 # Unknown

def main():
    print(f"Starting Data Processing...")
    print(f"Root Data Directory: {config.DATA_DIR}")
    
    if not os.path.exists(config.DATA_DIR):
        print(f"ERROR: Data directory not found: {config.DATA_DIR}")
        return

    all_X = []
    all_y = []
    
    # Iterate over all items in the Data directory
    subdirs = [d for d in os.listdir(config.DATA_DIR) if os.path.isdir(os.path.join(config.DATA_DIR, d))]
    
    print(f"Found {len(subdirs)} subdirectories: {subdirs}")

    total_tasks_processed = 0

    for subdir in subdirs:
        current_dir = os.path.join(config.DATA_DIR, subdir)
        labels_path = os.path.join(current_dir, "labels.json")
        
        # Check if labels.json exists in this subdirectory
        if not os.path.exists(labels_path):
            print(f"Skipping {subdir}: No labels.json found.")
            continue

        print(f"\nProcessing folder: {subdir}")
        
        try:
            with open(labels_path, 'r') as f:
                tasks = json.load(f)
        except json.JSONDecodeError:
            print(f"ERROR: Could not decode JSON in {subdir}")
            continue
            
        print(f"  -> Found {len(tasks)} tasks.")

        for task in tasks:
            # Identify CSV file
            original_filename = task.get('file_upload')
            if not original_filename: continue
                
            # Search for the CSV *inside the current subdirectory*
            csv_path = find_csv_file(original_filename, current_dir)
            
            if not csv_path:
                print(f"  -> Skipped (CSV not found): {original_filename}")
                continue
            
            # Load CSV
            try:
                df = pd.read_csv(csv_path)
                # Standardize column names if necessary (optional)
                df.columns = [c.capitalize() for c in df.columns] 
                
                if 'Timestamp' in df.columns:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                    df = df.set_index('Timestamp').sort_index()
            except Exception as e:
                print(f"  -> ERROR reading CSV {csv_path}: {e}")
                continue

            # Process Annotations
            labeled_intervals = []
            annotations = task.get('annotations', [])
            
            for ann in annotations:
                results = ann.get('result', [])
                for res in results:
                    val = res.get('value', {})
                    labels = val.get('timeserieslabels', [])
                    
                    if not labels: continue
                    
                    label_id = get_label_id(labels[0])
                    if label_id == -1: continue
                    
                    start_time = pd.to_datetime(val.get('start'))
                    end_time = pd.to_datetime(val.get('end'))
                    
                    # Extract window
                    mask = (df.index >= start_time) & (df.index <= end_time)
                    # Ensure columns exist (Open, High, Low, Close)
                    try:
                        window = df.loc[mask, ['Open', 'High', 'Low', 'Close']].values
                    except KeyError:
                        # Fallback for lowercase columns if needed
                        window = df.loc[mask, ['open', 'high', 'low', 'close']].values

                    if len(window) > 5:
                        # Normalize
                        window_norm = window / window[0] - 1.0
                        # Interpolate
                        window_resized = interpolate_sequence(window_norm, config.SEQ_LENGTH)
                        
                        all_X.append(window_resized)
                        all_y.append(label_id)
                        labeled_intervals.append((start_time, end_time))

            # Generate Negative Samples (Noise)
            num_positives = len(labeled_intervals)
            if num_positives > 0 and len(df) > config.SEQ_LENGTH:
                attempts = 0
                generated = 0
                while generated < num_positives and attempts < num_positives * 20:
                    attempts += 1
                    rand_idx = random.randint(0, len(df) - config.SEQ_LENGTH - 1)
                    random_window = df.iloc[rand_idx : rand_idx + config.SEQ_LENGTH]
                    
                    start_r, end_r = random_window.index[0], random_window.index[-1]
                    collision = False
                    for (s, e) in labeled_intervals:
                        if (start_r <= e) and (end_r >= s):
                            collision = True
                            break
                    
                    if not collision:
                        try:
                            w_vals = random_window[['Open', 'High', 'Low', 'Close']].values
                        except KeyError:
                            w_vals = random_window[['open', 'high', 'low', 'close']].values
                            
                        w_norm = w_vals / w_vals[0] - 1.0
                        all_X.append(w_norm)
                        all_y.append(0) # 0 = Noise
                        generated += 1
            
            total_tasks_processed += 1

    # Save Results
    if len(all_X) == 0:
        print("\nERROR: No samples extracted. Please check your JSON labels and CSV dates.")
        return

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    
    print(f"\nProcessing Complete!")
    print(f"Total Tasks Processed: {total_tasks_processed}")
    print(f"Total Samples Generated: {len(X)}")
    print(f"Data Shape: {X.shape}")
    
    # Save to the root Data directory (or Output if you prefer, check config.py)
    save_path = config.PROCESSED_DATA_PATH
    np.savez(save_path, X=X, y=y)
    print(f"Saved processed data to: {save_path}")

if __name__ == "__main__":
    main()