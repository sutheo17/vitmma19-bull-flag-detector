import torch
import torch.nn.functional as F
import numpy as np
import os
import random
import config
from utils import get_logger

# Importáljuk a modell osztályt a training scriptből
from importlib import import_module
train_module = import_module('02-train')
FlagClassifier = train_module.FlagClassifier

logger = get_logger()

def run_inference():
    logger.info("--- Inference Demo Started ---")

    # 1. Adatok betöltése (Hogy legyen mit tesztelni)
    if not os.path.exists(config.PROCESSED_DATA_PATH):
        logger.error(f"Data not found at {config.PROCESSED_DATA_PATH}")
        return

    data = np.load(config.PROCESSED_DATA_PATH)
    X = torch.FloatTensor(data['X'])
    y = torch.LongTensor(data['y'])

    # 2. Modell betöltése
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlagClassifier(input_size=config.NUM_FEATURES, num_classes=config.NUM_CLASSES).to(device)
    
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        logger.info(f"Model loaded successfully from {config.MODEL_SAVE_PATH}")
    else:
        logger.error("Model file not found. Please train the model first.")
        return

    model.eval()

    # 3. Mintavételezés és Predikció
    # Válasszunk 5 véletlenszerű mintát a demonstrációhoz
    num_samples = 5
    if len(X) < num_samples:
        num_samples = len(X)
        
    indices = random.sample(range(len(X)), num_samples)
    
    # Osztálynevek (Zaj nélkül, a 01-preprocessing.py és config.py alapján)
    class_names = [
        "Bull Normal", "Bull Pennant", "Bull Wedge",
        "Bear Normal", "Bear Pennant", "Bear Wedge"
    ]

    logger.info(f"Running inference on {num_samples} random samples:")
    logger.info("=" * 60)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Minta előkészítése: (Sequence, Features) -> (1, Features, Sequence)
            sample = X[idx].unsqueeze(0).to(device)
            true_label_idx = y[idx].item()
            
            # Forward pass
            logits = model(sample)
            probs = F.softmax(logits, dim=1)
            
            # Eredmény kinyerése
            conf, pred_idx = torch.max(probs, 1)
            
            predicted_class = class_names[pred_idx.item()]
            actual_class = class_names[true_label_idx]
            confidence = conf.item() * 100
            
            # Eredmény logolása
            result_status = "✅ CORRECT" if pred_idx.item() == true_label_idx else "❌ WRONG"
            
            logger.info(f"Sample #{i+1} (ID: {idx})")
            logger.info(f"   True Label: {actual_class}")
            logger.info(f"   Prediction: {predicted_class}")
            logger.info(f"   Confidence: {confidence:.2f}%")
            logger.info(f"   Result:     {result_status}")
            logger.info("-" * 60)

    logger.info("Inference demo completed.")

if __name__ == "__main__":
    run_inference()