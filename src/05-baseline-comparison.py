import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import config
from utils import get_logger
from config import PENNANT_THRESHOLD, WEDGE_THRESHOLD, MA_WINDOW

# Logger inicializálása
logger = get_logger()

def calculate_slope(values):
    """
    Lineáris regressziót illeszt az adatokra és visszaadja a meredekséget (slope).
    """
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    # Polyfit: y = mx + b, visszaadja [m, b]-t
    slope, _ = np.polyfit(x, values, 1)
    return slope

def heuristic_predict_single(sequence):
    """
    Egyetlen mintára (sequence) alkalmazza a szabályalapú logikát.
    Bemenet: (Seq_Len, 4) -> [Open, High, Low, Close]
    Kimenet: Class ID (0-5)
    """
    # Adatok szétbontása
    high_p = sequence[:, 1]
    low_p  = sequence[:, 2]
    close_p = sequence[:, 3]
    
    # 1. Fő Trend meghatározása (Záróár vs Kezdőár)
    # A preprocessing alapján a sequence az időben rendezett.
    start_price = sequence[0, 0] # Open t=0
    end_price = sequence[-1, 3]  # Close t=last
    
    is_bullish = end_price > start_price
    
    # 2. Konszolidáció kezdetének meghatározása (Extrémumok)
    if is_bullish:
        # Bullish: A zászló nyele a csúcsig tart (Maximum High)
        pivot_idx = np.argmax(high_p)
    else:
        # Bearish: A zászló nyele a mélypontig tart (Minimum Low)
        pivot_idx = np.argmin(low_p)
        
    # Ha a pivot a minta legvégén van, nincs konszolidáció -> Default Normal
    if pivot_idx >= len(sequence) - 5: # Kicsit szigorúbb feltétel (pl. utolsó 5 gyertya)
        slope = 1.0 # Mesterségesen nagy szám, hogy NORMAL legyen, ne Pennant!
    else:
        # 3. Mozgóátlag számítása a konszolidációs szakaszra
        cons_data = close_p[pivot_idx:]
        
        # Pandas rolling mean használata
        ma_series = pd.Series(cons_data).rolling(window=MA_WINDOW).mean()
        
        # A mozgóátlag elején keletkező NaN értékeket feltöltjük (backfill)
        ma_values = ma_series.bfill().fillna(0).values 
        
        # 4. Meredekség (Slope) számítása az egyenes illesztésével
        slope = calculate_slope(ma_values)

    # 5. Altípus meghatározása a meredekség alapján
    # Class Map:
    # Bullish: Normal=0, Pennant=1, Wedge=2
    # Bearish: Normal=3, Pennant=4, Wedge=5
    
    slope_abs = abs(slope)
    
    # Logika a csatolt ábra  geometriája alapján:
    # - Pennant: Szimmetrikus háromszög, az átlagos dőlésszög vízszintes közeli.
    # - Wedge/Flag: Az átlagos dőlésszög a trenddel ellentétes.
    
    if is_bullish:
        if slope_abs <= PENNANT_THRESHOLD:
            return 1 # Pennant (Lapos)
        elif slope_abs <= WEDGE_THRESHOLD:
            return 2 # Wedge (Enyhe)
        else:
            return 0 # Normal (Meredek VAGY Nincs konszolidáció)
    else: # Bearish
        if slope_abs <= PENNANT_THRESHOLD:
            return 4 # Pennant
        elif slope_abs <= WEDGE_THRESHOLD:
            return 5 # Wedge
        else:
            return 3 # Normal

def run_baseline_comparison():
    logger.info("--- Baseline (Heuristic) vs Deep Learning Comparison ---")

    # 1. Adatok betöltése
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
    # A. Heurisztikus Baseline Futtatása
    # ----------------------------------------
    logger.info("Running Heuristic Baseline Model...")
    y_pred_baseline = []
    
    for i in range(len(X)):
        pred = heuristic_predict_single(X[i])
        y_pred_baseline.append(pred)
        
    y_pred_baseline = np.array(y_pred_baseline)
    acc_baseline = accuracy_score(y_true, y_pred_baseline)

    # ----------------------------------------
    # B. Deep Learning Modell Futtatása (Ha létezik)
    # ----------------------------------------
    y_pred_dl = None
    try:
        from importlib import import_module
        train_module = import_module('02-train') # Dinamikus import a fájlnév miatt
        FlagClassifier = train_module.FlagClassifier
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FlagClassifier(input_size=config.NUM_FEATURES, num_classes=config.NUM_CLASSES).to(device)
        
        if os.path.exists(config.MODEL_SAVE_PATH):
            logger.info(f"Loading DL model from {config.MODEL_SAVE_PATH}...")
            model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
            model.eval()
            
            X_tensor = torch.FloatTensor(X).to(device)
            dl_preds = []
            
            # Batch feldolgozás a memória kímélése érdekében
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
    # C. Eredmények Kiírása
    # ----------------------------------------
    print("\n" + "="*60)
    print(f"BASELINE HEURISTIC REPORT (Accuracy: {acc_baseline*100:.2f}%)")
    print("="*60)
    print(classification_report(y_true, y_pred_baseline, target_names=class_names, zero_division=0))
    
    if y_pred_dl is not None:
        print("-"*60)

        print("\n" + "="*60)
        improvement = (acc_dl - acc_baseline) * 100
        print(f"DEEP LEARNING MODEL IMPROVEMENT: {improvement:+.2f}%")
        print("="*60)

        # Opcionális: Confusion Matrix összehasonlítás (csak szövegesen a diagonális)
        cm_base = confusion_matrix(y_true, y_pred_baseline)
        cm_dl = confusion_matrix(y_true, y_pred_dl)
        print("\nCorrect Predictions per Class (Baseline vs DL):")
        for i, name in enumerate(class_names):
            if i < len(cm_base) and i < len(cm_dl):
                print(f"  {name:15s}: {cm_base[i,i]} vs {cm_dl[i,i]}")

if __name__ == "__main__":
    run_baseline_comparison()