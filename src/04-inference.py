import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import config
from utils import get_logger

# Dynamically import the model class from 02-train.py
from importlib import import_module
train_module = import_module('02-train')
FlagClassifier = train_module.FlagClassifier

logger = get_logger()

# Define Class Mappings
CLASS_NAMES = [
    "Bull Normal", 
    "Bull Pennant", 
    "Bull Wedge", 
    "Bear Normal", 
    "Bear Pennant", 
    "Bear Wedge"
]

def interpolate_sequence(seq, target_length):
    """
    Resizes the sequence using linear interpolation to match model input size.
    Replicated from 01-data-preprocessing.py to ensure consistency.
    """
    if len(seq) == 0: return np.zeros((target_length, seq.shape[1]))
    result = []
    # Interpolate each feature column (Open, High, Low, Close) independently
    for col in range(seq.shape[1]):
        original = seq[:, col]
        x_old = np.linspace(0, 1, len(original))
        x_new = np.linspace(0, 1, target_length)
        result.append(np.interp(x_new, x_old, original))
    return np.stack(result, axis=1)

def preprocess_file(filepath):
    """
    Reads a CSV file and transforms it into a model-ready tensor.
    Applies Normalization (x / x[0] - 1) and Resizing.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Standardize columns
        df.columns = [c.capitalize() for c in df.columns]
        
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"File {os.path.basename(filepath)} missing required columns: {required_cols}")
            return None

        # Extract values
        sequence = df[required_cols].values
        
        if len(sequence) < 2:
            logger.warning(f"File {os.path.basename(filepath)} is too short.")
            return None

        # 1. Normalize: Relative change from the start (Pole Start)
        # This matches the logic in 01-data-preprocessing.py
        seq_norm = sequence / sequence[0] - 1.0
        
        # 2. Resize: Interpolate to config.SEQ_LENGTH
        seq_resized = interpolate_sequence(seq_norm, config.SEQ_LENGTH)
        
        # 3. Convert to Tensor: (1, SEQ_LENGTH, NUM_FEATURES)
        # Type must be float32 for PyTorch
        tensor = torch.tensor(seq_resized, dtype=torch.float32).unsqueeze(0)
        
        return tensor

    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return None

def run_inference():
    logger.info("="*50)
    logger.info("STARTING INFERENCE")
    logger.info("="*50)

    # 1. Check directories and Model
    if not os.path.exists(config.INFERENCE_DIR):
        logger.error(f"Inference directory not found: {config.INFERENCE_DIR}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model Architecture
    model = FlagClassifier(input_size=config.NUM_FEATURES, num_classes=config.NUM_CLASSES)
    
    # Load Weights
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        logger.info(f"Model loaded successfully from {config.MODEL_SAVE_PATH}")
    else:
        logger.error(f"Model file not found at {config.MODEL_SAVE_PATH}. Please run 02-train.py first.")
        return

    model.to(device)
    model.eval()

    # 2. Process Files
    files = [f for f in os.listdir(config.INFERENCE_DIR) if f.endswith('.csv')]
    
    if not files:
        logger.warning("No CSV files found in inference directory.")
        return

    logger.info(f"Found {len(files)} files to process.\n")
    
    results = []

    print(f"{'FILENAME':<30} | {'PREDICTION':<15} | {'CONFIDENCE':<10}")
    print("-" * 65)

    with torch.no_grad():
        for filename in files:
            filepath = os.path.join(config.INFERENCE_DIR, filename)
            
            # Preprocess
            input_tensor = preprocess_file(filepath)
            if input_tensor is None:
                continue
            
            input_tensor = input_tensor.to(device)
            
            # Forward Pass
            outputs = model(input_tensor)
            
            # Calculate Probabilities using Softmax
            probs = F.softmax(outputs, dim=1)
            
            # Get Prediction and Confidence
            max_prob, predicted_idx = torch.max(probs, 1)
            
            confidence = max_prob.item() * 100
            pred_class = CLASS_NAMES[predicted_idx.item()]
            
            # Output to console
            print(f"{filename:<30} | {pred_class:<15} | {confidence:.2f}%")
            
            results.append({
                "file": filename,
                "prediction": pred_class,
                "confidence": confidence
            })

    logger.info("Inference complete.")
    
    # Optional: Save results to CSV
    output_csv = os.path.join(config.OUTPUT_DIR, "inference_results.csv")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    logger.info(f"Results saved to {output_csv}")

if __name__ == "__main__":
    run_inference()