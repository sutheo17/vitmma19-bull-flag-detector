# src/config.py
import os

# --- Útvonalak (Docker kompatibilis) ---
DATA_DIR = "/app/data"  # A docker run -v paranccsal ide mountoljuk a fájlokat
LABELS_JSON_PATH = os.path.join(DATA_DIR, "labels.json")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.npz")
MODEL_SAVE_PATH = "model.pth"

# --- Adatfeldolgozási paraméterek ---
SEQ_LENGTH = 60       # Bemeneti ablak hossza (pl. 60 gyertya)
NUM_FEATURES = 4      # Open, High, Low, Close

# --- Osztályok (Labels) ---
# 0: Zaj (Noise)
# 1: Bullish Normal
# 2: Bullish Pennant
# 3: Bullish Wedge
# 4: Bearish Normal
# 5: Bearish Pennant
# 6: Bearish Wedge
NUM_CLASSES = 7 

# --- Tanítási hiperparaméterek ---
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2