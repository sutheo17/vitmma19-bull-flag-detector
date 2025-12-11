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
# 1: Bullish Normal
# 2: Bullish Pennant
# 3: Bullish Wedge
# 4: Bearish Normal
# 5: Bearish Pennant
# 6: Bearish Wedge
NUM_CLASSES = 6

# --- HEURISZTIKA BEÁLLÍTÁSOK ---
# A normalizált adatokhoz (ahol 0.01 kb 1% árváltozást jelent) igazított küszöbértékek.
# Ezek határozzák meg, hogy a konszolidáció dőlésszöge "lapos" (Pennant) vagy "meredek" (Flag/Wedge).
PENNANT_THRESHOLD = 0.0005  # Ha a meredekség abszolút értéke ez alatt van -> Pennant (közel vízszintes)
WEDGE_THRESHOLD = 0.0025    # Ha a meredekség ez alatt van (de Pennant felett) -> Wedge (kevésbé meredek)
                            # Ha ezen felül van -> Normal Flag (erős korrekció)

MA_WINDOW = 5  # Mozgóátlag ablak mérete a konszolidációs szakaszon

# --- Hiperparaméterek ---
BATCH_SIZE = 32          # Növeltük 16-ról 32-re a stabilabb tanulásért
EPOCHS = 200             # Több epoch, mert lassabban tanulunk
LEARNING_RATE = 0.0005   # Csökkentettük 0.001-ről (kisebb lépések)
WEIGHT_DECAY = 1e-4      # L2 Regularizáció (új)
DROPOUT_RATE = 0.5       # Dropout
TEST_SPLIT = 0.2