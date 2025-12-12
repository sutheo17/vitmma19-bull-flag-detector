import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import config
from utils import get_logger
from config import PENNANT_THRESHOLD, WEDGE_THRESHOLD, MA_WINDOW

# Initialize Logger
logger = get_logger()

def calculate_slope(values):
    """
    Fits linear regression to data and returns the slope.
    """
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    # Polyfit: y = mx + b, returns [m, b]
    slope, _ = np.polyfit(x, values, 1)
    return slope

def heuristic_predict_single(sequence):
    """
    Applies rule-based logic to a single sample.
    Input: (Seq_Len, 4) -> [Open, High, Low, Close]
    Output: Class ID (0-5)
    """
    # Break down data
    high_p = sequence[:, 1]
    low_p  = sequence[:, 2]
    close_p = sequence[:, 3]
    
    # 1. Determine Main Trend (Close vs Open)
    start_price = sequence[0, 0] # Open t=0 (Pole Start)
    end_price = sequence[-1, 3]  # Close t=last
    
    is_bullish = end_price > start_price
    
    # 2. Determine Consolidation Start (Extremes)
    if is_bullish:
        # Bullish: Pole goes up to the High
        pivot_idx = np.argmax(high_p)
    else:
        # Bearish: Pole goes down to the Low
        pivot_idx = np.argmin(low_p)
        
    # If pivot is at the very end, no consolidation -> Default Normal
    if pivot_idx >= len(sequence) - 5: 
        slope = 1.0 # Artificially high number to force NORMAL
    else:
        # 3. Calculate Moving Average for consolidation phase
        cons_data = close_p[pivot_idx:]
        
        # Pandas rolling mean
        ma_series = pd.Series(cons_data).rolling(window=MA_WINDOW).mean()
        
        # Fill NaN values created by MA
        ma_values = ma_series.bfill().fillna(0).values 
        
        # 4. Calculate Slope
        slope = calculate_slope(ma_values)

    # 5. Determine Subtype based on Slope
    # Class Map:
    # Bullish: Normal=0, Pennant=1, Wedge=2
    # Bearish: Normal=3, Pennant=4, Wedge=5
    
    slope_abs = abs(slope)
    
    if is_bullish:
        if slope_abs <= PENNANT_THRESHOLD:
            return 1 # Pennant (Flat)
        elif slope_abs <= WEDGE_THRESHOLD:
            return 2 # Wedge (Mild)
        else:
            return 0 # Normal (Steep or No Consolidation)
    else: # Bearish
        if slope_abs <= PENNANT_THRESHOLD:
            return 4 # Pennant
        elif slope_abs <= WEDGE_THRESHOLD:
            return 5 # Wedge
        else:
            return 3 # Normal

def run_baseline_comparison():
    logger.info("--- Baseline (Heuristic) vs Deep Learning Comparison ---")

    # 1. Load Data
    if not os.path.exists(config.PROCESSED_DATA_PATH):
        logger.error(f"Data not found at {config.PROCESSED_DATA_PATH}")
        return

    data = np.load(config.PROCESSED_DATA_PATH)
    X = data['X'] 
    y_true = data['y']
    
    class_names = [
        "Bull Normal", "Bull Pennant", "Bull Wedge",
        "Bear Normal", "Bear Pennant", "Bear Wedge"
    ]

    # ----------------------------------------
    # A. Run Heuristic Baseline
    # ----------------------------------------
    logger.info("Running Heuristic Baseline Model...")
    y_pred_baseline = []
    
    for i in range(len(X)):
        pred = heuristic_predict_single(X[i])
        y_pred_baseline.append(pred)
        
    y_pred_baseline = np.array(y_pred_baseline)
    acc_baseline = accuracy_score(y_true, y_pred_baseline)
    
    report_baseline = classification_report(y_true, y_pred_baseline, target_names=class_names, zero_division=0)

    # ----------------------------------------
    # B. Run Deep Learning Model (If exists)
    # ----------------------------------------
    y_pred_dl = None
    try:
        from importlib import import_module
        train_module = import_module('02-train')
        FlagClassifier = train_module.FlagClassifier
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FlagClassifier(input_size=config.NUM_FEATURES, num_classes=config.NUM_CLASSES).to(device)
        
        if os.path.exists(config.MODEL_SAVE_PATH):
            logger.info(f"Loading DL model from {config.MODEL_SAVE_PATH}...")
            model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
            model.eval()
            
            X_tensor = torch.FloatTensor(X).to(device)
            dl_preds = []
            
            batch_size = config.BATCH_SIZE
            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    batch = X_tensor[i:i+batch_size]
                    logits = model(batch)
                    _, preds = torch.max(logits, 1)
                    dl_preds.extend(preds.cpu().numpy())
            
            y_pred_dl = np.array(dl_preds)
            acc_dl = accuracy_score(y_true, y_pred_dl)
        else:
            logger.warning("DL Model file not found. Skipping DL comparison.")
            
    except Exception as e:
        logger.error(f"Failed to load DL model: {e}")

    # ----------------------------------------
    # C. Print and Save Results
    # ----------------------------------------
    print("\n" + "="*60)
    print(f"BASELINE HEURISTIC REPORT (Accuracy: {acc_baseline*100:.2f}%)")
    print("="*60)
    print(report_baseline)
    
    # Save Baseline Metrics to File
    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)
    baseline_report_path = os.path.join(output_dir, "baseline_metrics.txt")
    
    with open(baseline_report_path, "w") as f:
        f.write(f"Baseline Heuristic Report (Accuracy: {acc_baseline*100:.2f}%)\n")
        f.write("==================================================\n")
        f.write(report_baseline)
    logger.info(f"Baseline metrics saved to {baseline_report_path}")

    # Save Baseline Confusion Matrix
    cm_base = confusion_matrix(y_true, y_pred_baseline)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Baseline Confusion Matrix')
    
    cm_path = os.path.join(output_dir, "baseline_confusion_matrix.png")
    plt.savefig(cm_path)
    logger.info(f"Baseline Confusion Matrix saved to {cm_path}")
    plt.close()

    if y_pred_dl is not None:
        print("-"*60)

        print("\n" + "="*60)
        improvement = (acc_dl - acc_baseline) * 100
        print(f"DEEP LEARNING MODEL IMPROVEMENT: {improvement:+.2f}%")
        print("="*60)

        cm_dl = confusion_matrix(y_true, y_pred_dl)
        print("\nCorrect Predictions per Class (Baseline vs DL):")
        for i, name in enumerate(class_names):
            if i < len(cm_base) and i < len(cm_dl):
                print(f"  {name:15s}: {cm_base[i,i]} vs {cm_dl[i,i]}")

if __name__ == "__main__":
    run_baseline_comparison()