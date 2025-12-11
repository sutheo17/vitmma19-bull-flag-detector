import torch
import torch.nn as nn

class DirectionAwareLoss(nn.Module):
    def __init__(self, direction_weight=2.0):
        """
        Args:
            direction_weight (float): How much more important the direction (Bull/Bear) 
                                      is compared to the specific shape. 
                                      Higher = more penalty for wrong direction.
        """
        super(DirectionAwareLoss, self).__init__()
        self.direction_weight = direction_weight
        self.base_criterion = nn.CrossEntropyLoss()
        self.direction_criterion = nn.NLLLoss() # Negative Log Likelihood for aggregated probs

    def forward(self, logits, targets):
        # --- 1. Standard 7-Class Loss ---
        # Calculates loss for specific classes (0 to 6)
        fine_loss = self.base_criterion(logits, targets)

        # --- 2. Direction (Group) Loss ---
        # Map the 7 distinct classes to 3 groups:
        # 0 (Noise) -> 0
        # 1, 2, 3 (Bullish) -> 1
        # 4, 5, 6 (Bearish) -> 2
        
        # Create targets for direction
        direction_targets = torch.zeros_like(targets)
        direction_targets[(targets >= 1) & (targets <= 3)] = 1 # Bull Group
        direction_targets[(targets >= 4) & (targets <= 6)] = 2 # Bear Group

        # Aggregate probabilities from the model output
        probs = torch.softmax(logits, dim=1)
        
        # Sum probabilities for each group
        noise_prob = probs[:, 0]
        bull_prob  = torch.sum(probs[:, 1:4], dim=1) # Sum of cols 1, 2, 3
        bear_prob  = torch.sum(probs[:, 4:7], dim=1) # Sum of cols 4, 5, 6
        
        # Stack them to create a (Batch, 3) probability matrix
        direction_probs = torch.stack([noise_prob, bull_prob, bear_prob], dim=1)
        
        # Clamp to avoid log(0) errors and take log
        direction_log_probs = torch.log(torch.clamp(direction_probs, min=1e-7))

        # Calculate loss on the broad groups
        coarse_loss = self.direction_criterion(direction_log_probs, direction_targets)

        # --- 3. Combine ---
        # Total Loss = Standard Loss + (Weight * Direction Loss)
        return fine_loss + (self.direction_weight * coarse_loss)