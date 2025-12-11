import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import config
from utils import get_logger

# Importáljuk a modellt a meglévő fájlból, vagy definiáljuk újra
# Itt újra definiálom a Dropout módosítása miatt
class FlagClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FlagClassifier, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2), # Dropout a konvolúciós rétegek közé is
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (config.SEQ_LENGTH // 4), 64),
            nn.ReLU(),
            nn.Dropout(0.5), # Erős dropout a fully connected rétegen
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

logger = get_logger()

# --- Adatbővítés (Data Augmentation) ---
class AugmentedDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone() # Clone to avoid modifying original
        y = self.y[idx]

        if self.augment:
            # 1. Random Noise (Zaj)
            noise = torch.randn_like(x) * 0.02
            x += noise
            
            # 2. Random Scaling (Skálázás - kicsit összenyomjuk vagy széthúzzuk függőlegesen)
            scale = 1.0 + (torch.rand(1) * 0.1 - 0.05) # 0.95 - 1.05
            x *= scale
            
        return x, y

def compute_class_weights(y_train, device):
    """Kiszámolja az osztálysúlyokat a ritka osztályok kompenzálására."""
    class_counts = torch.bincount(y_train)
    total_samples = len(y_train)
    num_classes = len(class_counts)
    
    # Formula: Total / (Num_Classes * Count)
    weights = total_samples / (num_classes * class_counts.float())
    
    # Normalizálás és áthelyezés GPU-ra
    return weights.to(device)

def train_improved():
    logger.info("--- Improved Training Script Started ---")
    
    # 1. Load Data
    if not os.path.exists(config.PROCESSED_DATA_PATH):
        logger.error("Data not found.")
        return

    data = np.load(config.PROCESSED_DATA_PATH)
    X_loaded = torch.FloatTensor(data['X'])
    y_loaded = torch.LongTensor(data['y'])
    
    # Train/Val Split
    indices = torch.randperm(len(X_loaded))
    split_idx = int(len(X_loaded) * (1 - config.TEST_SPLIT))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train, y_train = X_loaded[train_indices], y_loaded[train_indices]
    X_val, y_val = X_loaded[val_indices], y_loaded[val_indices]
    
    logger.info(f"Train size: {len(X_train)} | Val size: {len(X_val)}")

    # 2. Setup DataLoaders with Augmentation
    # Csak a training setre kapcsoljuk be az augmentációt!
    train_dataset = AugmentedDataset(X_train, y_train, augment=True)
    val_dataset = AugmentedDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 3. Model & Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlagClassifier(input_size=config.NUM_FEATURES, num_classes=config.NUM_CLASSES).to(device)
    
    # Osztálysúlyok számítása
    class_weights = compute_class_weights(y_train, device)
    logger.info(f"Class Weights: {class_weights}")
    
    # Standard CrossEntropyLoss súlyozással (stabilabb mint a custom loss kezdetben)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer Weight Decay-vel (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Scheduler: Ha a validation loss nem javul 10 epochig, csökkenti a LR-t
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    best_val_loss = float('inf')
    early_stop_counter = 0
    patience_limit = 25 # Ha 25 epochig nincs javulás, megállunk

    logger.info("Starting training loop...")

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping (Megakadályozza a loss ugrálást)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Scheduler lépés
        scheduler.step(val_loss)

        logger.info(f"Epoch [{epoch+1}/{config.EPOCHS}] "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            logger.info(" > Best model saved.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience_limit:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training completed.")

if __name__ == "__main__":
    train_improved()