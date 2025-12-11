import json
import os
import pandas as pd
import numpy as np
import config

def find_csv_file(target_filename, search_dir):
    """Megkeresi a fájlt a mappában."""
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
    """Lineáris interpolációval átméretezi a szekvenciát."""
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
    """
    Javított dátumfelismerés DataFrame oszlopra.
    """
    col_map = {c.lower(): c for c in df.columns}
    ts_col = col_map.get('timestamp') or col_map.get('time') or col_map.get('date')
    
    if not ts_col:
        return df

    # Próbáljuk meg numerikussá alakítani
    temp_col = pd.to_numeric(df[ts_col], errors='coerce')
    
    # Ha a többség szám (Unix timestamp)
    if temp_col.notna().sum() > len(df) * 0.8:
        temp_col = temp_col.ffill().bfill()
        first_val = temp_col.iloc[0]
        
        # > 3e10 (kb 1971-es év másodpercben) -> valószínűleg ms
        if first_val > 3e10:
            df[ts_col] = pd.to_datetime(temp_col, unit='ms')
        else:
            df[ts_col] = pd.to_datetime(temp_col, unit='s')
    else:
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

    df = df.set_index(ts_col).sort_index()
    return df

def parse_annotation_time(time_val):
    """
    Egyetlen JSON időérték biztonságos parse-olása.
    Kezeli a string-be csomagolt Unix timestamp-et (ms) is.
    """
    if time_val is None:
        return None
        
    # 1. Próbáljuk meg számmá konvertálni (float kezeli a string int-et is)
    try:
        numeric_ts = float(time_val)
        # Ha nagyon nagy szám, akkor ms (pl. 1707829200000)
        # Ha kicsi, akkor s (pl. 1707829200)
        if numeric_ts > 3e10: 
            return pd.to_datetime(numeric_ts, unit='ms')
        else:
            return pd.to_datetime(numeric_ts, unit='s')
    except (ValueError, TypeError):
        # Nem szám, hanem valódi dátum string (pl. "2024-02-14")
        return pd.to_datetime(time_val)

def main():
    print(f"Data Processing...")
    
    if not os.path.exists(config.DATA_DIR):
        print(f"ERROR: Data directory not found: {config.DATA_DIR}")
        return

    all_X = []
    all_y = []
    
    subdirs = [d for d in os.listdir(config.DATA_DIR) if os.path.isdir(os.path.join(config.DATA_DIR, d))]
    
    total_tasks_processed = 0

    for subdir in subdirs:
        current_dir = os.path.join(config.DATA_DIR, subdir)
        labels_path = os.path.join(current_dir, "labels.json")
        
        if not os.path.exists(labels_path):
            continue

        print(f"Processing folder: {subdir}")
        
        try:
            with open(labels_path, 'r') as f:
                tasks = json.load(f)
        except:
            continue

        for task in tasks:
            original_filename = task.get('file_upload')
            if not original_filename: continue
                
            csv_path = find_csv_file(original_filename, current_dir)
            if not csv_path: continue
            
            try:
                df = pd.read_csv(csv_path)
                df.columns = [c.capitalize() for c in df.columns] 
                
                # 1. CSV Időbélyeg javítása
                df = robust_parse_dates(df)
                
            except Exception as e:
                print(f"Skipping {csv_path} due to load error: {e}")
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
                    
                    # 2. JSON Időbélyegek javítása (ITT VOLT A HIBA)
                    start_time = parse_annotation_time(val.get('start'))
                    end_time = parse_annotation_time(val.get('end'))
                    
                    if start_time is None or end_time is None:
                        continue
                    
                    # Időablak kivágása
                    mask = (df.index >= start_time) & (df.index <= end_time)
                    
                    try:
                        window = df.loc[mask, ['Open', 'High', 'Low', 'Close']].values
                    except KeyError:
                        try:
                            window = df.loc[mask, ['open', 'high', 'low', 'close']].values
                        except KeyError:
                            continue

                    if len(window) > 5:
                        window_norm = window / window[0] - 1.0
                        window_resized = interpolate_sequence(window_norm, config.SEQ_LENGTH)
                        
                        all_X.append(window_resized)
                        all_y.append(label_id)
            
            total_tasks_processed += 1

    if len(all_X) == 0:
        print("ERROR: No valid samples extracted.")
        return

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    
    print(f"\nProcessing Complete!")
    print(f"Total Samples: {len(X)}")
    
    np.savez(config.PROCESSED_DATA_PATH, X=X, y=y)
    print(f"Saved to: {config.PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()