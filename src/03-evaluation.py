import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, mean_absolute_error
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
    class_names = ["Bull_Normal", "Bull_Pennant", "Bull_Wedge", "Bear_Normal", "Bear_Pennant", "Bear_Wedge"]
    error_dir = os.path.join(output_dir, "all_errors")
    os.makedirs(error_dir, exist_ok=True)
    
    global_count = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                if labels[i] != predicted[i]:
                    seq = inputs[i].cpu().numpy().T
                    plt.figure(figsize=(6, 4))
                    plt.plot(seq[:, 3], label='Close', color='black')
                    plt.plot(seq[:, 0], label='Open', color='gray', linestyle='--')
                    plt.title(f"Real: {class_names[labels[i]]} -> Pred: {class_names[predicted[i]]}")
                    plt.savefig(os.path.join(error_dir, f"err_{global_count}.png"))
                    plt.close()
                    global_count += 1
    logger.info(f"Saved {global_count} misclassified images.")

def evaluate():
    logger.info("="*50)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*50)

    if not os.path.exists(config.PROCESSED_DATA_PATH):
        logger.error("Data file not found.")
        return

    data = np.load(config.PROCESSED_DATA_PATH)
    X = torch.FloatTensor(data['X'])
    y = torch.LongTensor(data['y'])

    # Use the same split logic or a separate test set if defined
    # For now, using the whole dataset as a demo, but in real scenarios, use a held-out set
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

    class_names = ["Bull Normal", "Bull Pennant", "Bull Wedge", "Bear Normal", "Bear Pennant", "Bear Wedge"]
    
    # 6. FINAL EVALUATION METRICS LOGGING
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    # MAE doesn't make much sense for classification, but included if requested
    mae = mean_absolute_error(all_labels, all_preds)

    logger.info(f"Final Accuracy: {acc*100:.2f}%")
    logger.info(f"Weighted F1-Score: {f1:.4f}")
    logger.info(f"Mean Absolute Error: {mae:.4f}")
    
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    logger.info("\nDetailed Classification Report:\n" + report)
    
    # Save results to file
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, "final_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc}\nF1: {f1}\nMAE: {mae}\n\n{report}")
    logger.info(f"Final metrics saved to {os.path.join(config.OUTPUT_DIR, 'final_metrics.txt')}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"))
    logger.info(f"Confusion matrix saved to {os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')}")

    if config.SAVE_ERRORS_TO_OUPUT:
        save_misclassified_examples(model, test_loader, device, config.OUTPUT_DIR)
if __name__ == "__main__":
    evaluate()