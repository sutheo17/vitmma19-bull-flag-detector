import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import get_logger
import config

# --- CUDNN FIX START ---
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# --- CUDNN FIX END ---

logger = get_logger()

def compute_class_weights(y_train, device):
    """Computes inverse class weights to handle imbalance."""
    counts = torch.bincount(y_train)
    counts = torch.max(counts, torch.ones_like(counts))
    total = len(y_train)
    num_classes = len(counts)
    weights = total / (num_classes * counts.float())
    return weights.to(device)

def log_hyperparameters():
    """Logs all hyperparameters from config.py to meet grading requirements."""
    logger.info("="*50)
    logger.info("CONFIGURATION (Hyperparameters)")
    logger.info("="*50)
    logger.info(f"EPOCHS:            {config.EPOCHS}")
    logger.info(f"BATCH_SIZE:        {config.BATCH_SIZE}")
    logger.info(f"LEARNING_RATE:     {config.LEARNING_RATE}")
    logger.info(f"WEIGHT_DECAY:      {config.WEIGHT_DECAY}")
    logger.info(f"DROPOUT_RATE:      {config.DROPOUT_RATE}")
    logger.info(f"SEQ_LENGTH:        {config.SEQ_LENGTH}")
    logger.info(f"NUM_FEATURES:      {config.NUM_FEATURES}")
    logger.info(f"NUM_CLASSES:       {config.NUM_CLASSES}")
    logger.info(f"EARLY_STOP_PAT:    {config.EARLY_STOP_PATIENCE}")
    logger.info(f"LR_PATIENCE:       {config.LR_PATIENCE}")
    logger.info(f"MAX_GRAD_NORM:     {config.MAX_GRAD_NORM}")
    logger.info("-" * 50)

def log_model_summary(model):
    """Logs the model architecture and parameter count."""
    logger.info("="*50)
    logger.info("MODEL ARCHITECTURE SUMMARY")
    logger.info("="*50)
    logger.info(str(model)) # Prints the layer structure
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    logger.info("-" * 50)
    logger.info(f"Total Parameters:         {total_params:,}")
    logger.info(f"Trainable Parameters:     {trainable_params:,}")
    logger.info(f"Non-trainable Parameters: {non_trainable_params:,}")
    logger.info("="*50)

# ---------------- MODEL ----------------
class FlagClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FlagClassifier, self).__init__()
        self.conv_layer = nn.Sequential(
            # Layer 1
            nn.Conv1d(input_size, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2), 

            # Layer 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            # Layer 3
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )

        reduced_len = config.SEQ_LENGTH // 8  # three pooling layers divide length by 2^3=8
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * reduced_len, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # -> (batch, features, seq_len)
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

# ---------------- DATASET + AUG ----------------
class AugmentedDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            # Scaling jitter
            scale = 1.0 + (torch.rand(1) * 2 * config.SCALE_JITTER - config.SCALE_JITTER)
            x = x * scale

            # Time shifting (if enabled)
            if config.SHIFT_ENABLED:
                shift = int(torch.randint(low=config.SHIFT_RANGE[0], high=config.SHIFT_RANGE[1] + 1, size=(1,)).item())
                if shift > 0:
                    x = torch.roll(x, shifts=shift, dims=0)
                    first_valid = x[shift, :].clone()
                    x[:shift, :] = first_valid
                elif shift < 0:
                    x = torch.roll(x, shifts=shift, dims=0)
                    last_valid = x[shift-1, :].clone()
                    x[shift:, :] = last_valid

        return x, y

# ---------------- TRAIN ----------------

def train():
    logger.info('--- Training Process Started ---')
    
    # 1. LOGGING CONFIGURATION
    log_hyperparameters()

    # 2. DATA PREPARATION LOGGING
    if not os.path.exists(config.PROCESSED_DATA_PATH):
        logger.error('Processed data not found; path: %s', config.PROCESSED_DATA_PATH)
        return

    logger.info("Loading processed data...")
    data = np.load(config.PROCESSED_DATA_PATH)
    X_all = torch.FloatTensor(data['X'])
    y_all = torch.LongTensor(data['y'])
    
    logger.info(f"Data Loaded Successfully. Total Samples: {len(X_all)}")
    logger.info(f"Input Shape: {X_all.shape}")

    # Reproducible split
    X_train, X_val, y_train, y_val = train_test_split(
        X_all.numpy(), y_all.numpy(), test_size=config.TEST_SPLIT, stratify=y_all.numpy(), random_state=42
    )
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    
    logger.info(f"Train Set: {len(X_train)} samples | Validation Set: {len(X_val)} samples")

    train_dataset = AugmentedDataset(X_train, y_train, augment=True)
    val_dataset = AugmentedDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    model = FlagClassifier(input_size=config.NUM_FEATURES, num_classes=config.NUM_CLASSES).to(device)

    # 3. LOGGING MODEL ARCHITECTURE
    log_model_summary(model)

    class_weights = compute_class_weights(y_train.to(device), device)
    logger.info('Class Weights calculated for imbalance handling: %s', class_weights.cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.LR_FACTOR, patience=config.LR_PATIENCE)

    best_val_loss = float('inf')
    early_counter = 0

    logger.info("="*50)
    logger.info("TRAINING PROGRESS")
    logger.info("="*50)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM)
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

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
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        scheduler.step(val_loss)

        # 4 & 5. LOGGING TRAINING & VALIDATION METRICS
        logger.info('Epoch [%d/%d] Train Loss: %.4f Acc: %.2f%% | Val Loss: %.4f Acc: %.2f%%',
                    epoch+1, config.EPOCHS, train_loss, train_acc, val_loss, val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            logger.info(' > Best model saved (%.4f).', best_val_loss)
        else:
            early_counter += 1
            if early_counter >= config.EARLY_STOP_PATIENCE:
                logger.info('Early stopping triggered.')
                break

    logger.info('Training phase finished.')

if __name__ == '__main__':
    train()