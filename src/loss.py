import torch
import torch.nn as nn

# --- Custom Loss (Zaj nélkül, csak irány és típus) ---
class DirectionAwareLoss(nn.Module):
    def __init__(self, direction_weight=3.0):
        super(DirectionAwareLoss, self).__init__()
        self.direction_weight = direction_weight
        self.base_criterion = nn.CrossEntropyLoss()
        self.direction_criterion = nn.NLLLoss()

    def forward(self, logits, targets):
        # 1. Finom hiba (6 osztály)
        fine_loss = self.base_criterion(logits, targets)

        # 2. Irány hiba (Bull vs Bear)
        # Új mapping: 
        # 0,1,2 (Bull) -> 0
        # 3,4,5 (Bear) -> 1
        
        direction_targets = torch.zeros_like(targets)
        # Bullish csoport (ID 0, 1, 2) -> Irány ID 0
        direction_targets[targets <= 2] = 0 
        # Bearish csoport (ID 3, 4, 5) -> Irány ID 1
        direction_targets[targets >= 3] = 1 

        probs = torch.softmax(logits, dim=1)
        
        # Bullish valószínűségek összege (0, 1, 2 oszlopok)
        bull_prob  = torch.sum(probs[:, 0:3], dim=1)
        # Bearish valószínűségek összege (3, 4, 5 oszlopok)
        bear_prob  = torch.sum(probs[:, 3:6], dim=1)
        
        direction_probs = torch.stack([bull_prob, bear_prob], dim=1)
        direction_log_probs = torch.log(torch.clamp(direction_probs, min=1e-7))

        coarse_loss = self.direction_criterion(direction_log_probs, direction_targets)

        return fine_loss + (self.direction_weight * coarse_loss)