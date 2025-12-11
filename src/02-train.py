# 02-train_tuned.py
# Tuned training script for OHLC flag classifier (drop-in replacement)
# - Reduced dropout, disabled time-shift augment, milder scaling
# - Slightly larger CNN (32->64 filters), bigger FC, grad clipping relaxed
# - LR, batch size, weight decay adjusted

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import get_logger
from utils import compute_class_weights
from config import (SEQ_LENGTH, NUM_FEATURES, NUM_CLASSES, BATCH_SIZE, EPOCHS,
                    LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE,
                    LR_FACTOR, LR_PATIENCE, EARLY_STOP_PATIENCE,
                    MAX_GRAD_NORM, SCALE_JITTER, SHIFT_ENABLED, SHIFT_RANGE, PROCESSED_DATA_PATH, MODEL_SAVE_PATH, TEST_SPLIT)

logger = get_logger()

# ---------------- MODEL ----------------
class FlagClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FlagClassifier, self).__init__()
        self.conv_layer = nn.Sequential(
            # widened convs but keep model small
            nn.Conv1d(input_size, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
        )

        reduced_len = SEQ_LENGTH // 8  # three poolings
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * reduced_len, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
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
            # scaling jitter
            scale = 1.0 + (torch.rand(1) * 2 * SCALE_JITTER - SCALE_JITTER)
            x = x * scale

            # small shift only if explicitly enabled (default OFF)
            if SHIFT_ENABLED:
                shift = int(torch.randint(low=SHIFT_RANGE[0], high=SHIFT_RANGE[1] + 1, size=(1,)).item())
                if shift > 0:
                    x = torch.roll(x, shifts=shift, dims=0)
                    x[:shift, :] = 0.0
                elif shift < 0:
                    x = torch.roll(x, shifts=shift, dims=0)
                    x[shift:, :] = 0.0

        return x, y

# ---------------- TRAIN ----------------

def train():
    logger.info('--- Tuned training started ---')
    if not os.path.exists(PROCESSED_DATA_PATH):
        logger.error('Processed data not found; path: %s', PROCESSED_DATA_PATH)
        return

    data = np.load(PROCESSED_DATA_PATH)
    X_all = torch.FloatTensor(data['X'])
    y_all = torch.LongTensor(data['y'])

    # reproducible split
    X_train, X_val, y_train, y_val = train_test_split(
        X_all.numpy(), y_all.numpy(), test_size=TEST_SPLIT, stratify=y_all.numpy(), random_state=42
    )
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)

    train_dataset = AugmentedDataset(X_train, y_train, augment=True)
    val_dataset = AugmentedDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FlagClassifier(input_size=NUM_FEATURES, num_classes=NUM_CLASSES).to(device)

    class_weights = compute_class_weights(y_train.to(device), device)
    logger.info('Class Weights: %s', class_weights.cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)

    best_val_loss = float('inf')
    early_counter = 0

    for epoch in range(EPOCHS):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # validation
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

        logger.info('Epoch [%d/%d] Train Loss: %.4f Acc: %.2f%% | Val Loss: %.4f Acc: %.2f%%',
                    epoch+1, EPOCHS, train_loss, train_acc, val_loss, val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(' > Best model saved (%.4f).', best_val_loss)
        else:
            early_counter += 1
            if early_counter >= EARLY_STOP_PATIENCE:
                logger.info('Early stopping triggered.')
                break

    logger.info('Training finished.')

if __name__ == '__main__':
    train()
