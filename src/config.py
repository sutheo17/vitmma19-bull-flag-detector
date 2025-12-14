# src/config.py
import os

# --- Paths (Docker compatible) ---
DATA_DIR = "/app/data"  # Mount point for docker run -v
OUTPUT_DIR = "/app/output"
INFERENCE_DIR = "/app/data/inference"
LABELS_JSON_PATH = os.path.join(DATA_DIR, "labels.json")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.npz")
MODEL_SAVE_PATH = "model.pth"
DATABASE_LINK = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQCPx-aGyYgLTYeUnwMncbknAaI2NMSz4DnbS13UYvMQIOI?download=1"

# --- Data Processing Parameters ---
SEQ_LENGTH = 80      # Input window length (increased to accommodate pole + flag)
NUM_FEATURES = 4      # Open, High, Low, Close
TEST_SPLIT = 0.2

# --- Classes (Labels) ---
# 0: Bullish Normal
# 1: Bullish Pennant
# 2: Bullish Wedge
# 3: Bearish Normal
# 4: Bearish Pennant
# 5: Bearish Wedge
NUM_CLASSES = 6

# --- Training Hyperparameters ---
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.0008
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.15

# --- Optimization / Scheduler ---
LR_FACTOR = 0.5
LR_PATIENCE = 10
EARLY_STOP_PATIENCE = 40
MAX_GRAD_NORM = 5.0   # Gradient clipping

# --- Augmentation Parameters ---
SCALE_JITTER = 0.03
SHIFT_ENABLED = False # Time shifting
SHIFT_RANGE = (-1, 1)

# --- Heuristic / Baseline Settings ---
# Thresholds adjusted for normalized data (where 0.01 approx 1% price change).
# Determines if consolidation slope is "flat" (Pennant) or "steep" (Flag/Wedge).
PENNANT_THRESHOLD = 0.00004  # Absolute slope below this -> Pennant (near horizontal)
WEDGE_THRESHOLD = 0.00015    # Slope below this (but above Pennant) -> Wedge
                             # Above this -> Normal Flag (strong correction)

MA_WINDOW = 5                # Moving Average window for consolidation analysis

# --- Evaluation Settings ---
SAVE_ERRORS_TO_OUPUT = False  # Save graphs of misclassified samples