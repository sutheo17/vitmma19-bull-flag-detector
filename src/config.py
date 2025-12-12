# src/config.py
import os

# --- Útvonalak (Docker kompatibilis) ---
DATA_DIR = "/app/data"  # A docker run -v paranccsal ide mountoljuk a fájlokat
LABELS_JSON_PATH = os.path.join(DATA_DIR, "labels.json")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.npz")
MODEL_SAVE_PATH = "model.pth"
DATABASE_LINK = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQAK7NwTLL-kSZzXylIIkQFOAX1TXEZYIDQomXH4sf0-zLQ?download=1"

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
PENNANT_THRESHOLD = 0.00004  # Ha a meredekség abszolút értéke ez alatt van -> Pennant (közel vízszintes)
WEDGE_THRESHOLD = 0.00015   # Ha a meredekség ez alatt van (de Pennant felett) -> Wedge (kevésbé meredek)
                            # Ha ezen felül van -> Normal Flag (erős korrekció)

MA_WINDOW = 5  # Mozgóátlag ablak mérete a konszolidációs szakaszon

BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.0008
WEIGHT_DECAY = 0.0
DROPOUT_RATE = 0.15

# scheduler / early stop
LR_FACTOR = 0.5
LR_PATIENCE = 10
EARLY_STOP_PATIENCE = 40

# grad clip
MAX_GRAD_NORM = 5.0

# augmentation params
SCALE_JITTER = 0.03  # +/-3%
SHIFT_ENABLED = False
SHIFT_RANGE = (-1, 1)  # if enabled, small shifts only

TEST_SPLIT = 0.2