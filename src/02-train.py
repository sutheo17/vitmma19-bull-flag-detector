import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import config
from utils import get_logger
from loss import DirectionAwareLoss

# Initialize logger
logger = get_logger()

# --- Model Definition (1D CNN) ---
class FlagClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FlagClassifier, self).__init__()
        # Input shape: (Batch, Channels, Seq_Length) -> (32, 4, 60)
        
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Calculate size after pooling: 60 -> 30 -> 15
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (config.SEQ_LENGTH // 4), 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Swap dimensions for Conv1d: (Batch, Seq, Feat) -> (Batch, Feat, Seq)
        x = x.permute(0, 2, 1)
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

def train():
    logger.info("--- Training Script Started ---")
    
    # 1. Load Data
    if not os.path.exists(config.PROCESSED_DATA_PATH):
        logger.error(f"Processed data not found at {config.PROCESSED_DATA_PATH}. Run preprocessing first.")
        return

    logger.info(f"Loading data from {config.PROCESSED_DATA_PATH}...")
    data = np.load(config.PROCESSED_DATA_PATH)
    X_loaded = data['X']
    y_loaded = data['y']
    
    # Simple Train/Test Split (since preprocessing saved everything together)
    indices = np.arange(len(X_loaded))
    np.random.shuffle(indices)
    split_idx = int(len(X_loaded) * (1 - config.TEST_SPLIT))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train = torch.FloatTensor(X_loaded[train_indices])
    y_train = torch.LongTensor(y_loaded[train_indices])
    X_val = torch.FloatTensor(X_loaded[val_indices])
    y_val = torch.LongTensor(y_loaded[val_indices])
    
    logger.info(f"Train set size: {len(X_train)}, Validation set size: {len(X_val)}")
    
    # DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 2. Setup Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = FlagClassifier(input_size=config.NUM_FEATURES, num_classes=config.NUM_CLASSES).to(device)
    criterion = DirectionAwareLoss(direction_weight=3.0)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Log Model Architecture
    logger.info(f"Model Architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total Trainable Parameters: {total_params}")

    # 3. Training Loop
    best_val_loss = float('inf')
    
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
        
        logger.info(f"Epoch [{epoch+1}/{config.EPOCHS}] "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            logger.info(" > Validation loss decreased. Model saved.")

    logger.info("Training completed.")

if __name__ == "__main__":
    train()