import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCE(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move pos_weight to same device as logits
        pos_weight = self.pos_weight.to(logits.device)
        return F.binary_cross_entropy_with_logits(
            logits.view(-1), 
            targets.view(-1).float(), 
            pos_weight=pos_weight
        )

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits.view(-1), 
            targets.view(-1).float(), 
            reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def get_loss_fn(loss_name: str, **kwargs) -> nn.Module:
    if loss_name == "weighted_bce":
        return WeightedBCE(**kwargs)
    elif loss_name == "focal":
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_name}")
