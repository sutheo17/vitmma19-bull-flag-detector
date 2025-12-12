import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
import config
from utils import get_logger

from importlib import import_module
train_module = import_module('02-train')
FlagClassifier = train_module.FlagClassifier

logger = get_logger()

def save_misclassified_examples(model, test_loader, device, output_dir):
    model.eval()
    
    # Osztályok indexei a te mappinged alapján
    # Bear Normal = 3
    # Bull Wedge = 2
    TRUE_CLASS = 3
    PRED_CLASS = 2
    
    count = 0
    os.makedirs(os.path.join(output_dir, "errors"), exist_ok=True)
    
    import matplotlib.pyplot as plt

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Keressük meg a hibásakat
            for i in range(len(labels)):
                if labels[i] == TRUE_CLASS and predicted[i] == PRED_CLASS:
                    # Kimentjük a képet
                    seq = inputs[i].cpu().numpy().T # (Features, Seq) -> (Seq, Features) transzponálás vissza
                    
                    plt.figure(figsize=(5, 3))
                    # Csak a Close árat (3. index) rajzoljuk ki, vagy az összeset
                    # Open=0, High=1, Low=2, Close=3
                    plt.plot(seq[:, 3], label='Close Price', color='black')
                    plt.plot(seq[:, 0], label='Open', linestyle='--', alpha=0.5)
                    plt.title(f"True: BearNormal(3) -> Pred: BullWedge(2)\nIndex: {count}")
                    plt.legend()
                    plt.grid(True)
                    
                    filename = os.path.join(output_dir, "errors", f"error_bear3_pred_bull2_{count}.png")
                    plt.savefig(filename)
                    plt.close()
                    
                    count += 1
                    if count >= 20: return # Elég az első 20-at látni

def evaluate():
    logger.info("--- Evaluation Script ---")

    if not os.path.exists(config.PROCESSED_DATA_PATH):
        logger.error("Data file not found.")
        return

    data = np.load(config.PROCESSED_DATA_PATH)
    X = torch.FloatTensor(data['X'])
    y = torch.LongTensor(data['y'])

    test_loader = DataLoader(TensorDataset(X, y), batch_size=config.BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlagClassifier(input_size=config.NUM_FEATURES, num_classes=config.NUM_CLASSES).to(device)
    
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        logger.info(f"Model loaded from {config.MODEL_SAVE_PATH}")
    else:
        logger.error("Model file not found. Run training first.")
        return

    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # FRISSÍTETT OSZTÁLYNEVEK (6 db)
    class_names = [
        "Bull Normal", "Bull Pennant", "Bull Wedge",
        "Bear Normal", "Bear Pennant", "Bear Wedge"
    ]
    
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    acc_dl = (np.array(all_preds) == np.array(all_labels)).mean()

    logger.info("\nClassification Report:")
    print("\n" + "="*60)
    print(f"DEEP LEARNING MODEL REPORT (Accuracy: {acc_dl*100:.2f}%)")
    print("="*60)
    print(report)
    print("-"*60)
    
    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "evaluation_metrics.txt")
    with open(report_path, "w") as f:
        f.write("Evaluation Metrics (DL)\n")
        f.write("===================================\n\n")
        f.write(report)
    logger.info(f"Metrics saved to {report_path}")

    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")

    save_misclassified_examples(model, test_loader, device, output_dir)

if __name__ == "__main__":
    evaluate()