"""
NeuroAegis Binary Focal Loss Implementation
Phase: Class Imbalance Handling Strategy

Implements numerically stable Binary Focal Loss operating directly on unnormalized logits.
Reference: Lin et al., "Focal Loss for Dense Object Detection", IEEE ICCV 2017.

Mathematical Formulation:
    p = sigmoid(logits)
    p_t = y * p + (1 - y) * (1 - p)
    alpha_t = y * alpha + (1 - y) * (1 - alpha)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    loss = focal_weight * BCEWithLogits(logits, y)

Key Properties:
1. Direct logit input avoids double sigmoid and maintains numerical stability.
2. When gamma = 0, identically reduces to alpha-weighted Binary Cross-Entropy.
3. Alpha balances positive vs negative importance; Gamma down-weights easy examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """
    Converts raw unnormalized model logits to calibrated probabilities.
    Sigmoid is applied exactly once here for inference and metric evaluation only.
    
    Args:
        logits: Tensor of raw model logits of arbitrary shape.
        
    Returns:
        Tensor of probabilities in [0.0, 1.0].
    """
    return torch.sigmoid(logits)


class BinaryFocalLossWithLogits(nn.Module):
    """
    Binary Focal Loss with direct logit input for numerical stability.
    
    Args:
        alpha: Weighting factor for the rare/positive class (0 < alpha < 1).
               The negative class receives weight (1 - alpha). Default: 0.25.
        gamma: Focusing parameter for modulating hard vs easy examples (gamma >= 0).
               When gamma=0, behaves identically to alpha-weighted BCE. Default: 2.0.
        reduction: Reduction method: 'mean', 'sum', or 'none'. Default: 'mean'.
        eps: Small epsilon for numerical clamping stability. Default: 1e-8.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Alpha must be in (0, 1), got {alpha}")
        if gamma < 0.0:
            raise ValueError(f"Gamma must be non-negative, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}. Must be 'mean', 'sum', or 'none'.")
            
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward computation of Binary Focal Loss.
        
        Args:
            logits: Unnormalized model predictions, shape (N, ...) or (N, 1) or (N,).
            targets: Ground truth binary labels in {0, 1}, same shape as logits or broadcastable.
            
        Returns:
            Computed scalar loss (if reduction is 'mean' or 'sum') or per-element loss tensor (if reduction is 'none').
        """
        if logits.shape != targets.shape:
            # Ensure shape alignment (e.g., (N, 1) vs (N,))
            targets = targets.view_as(logits)
            
        targets = targets.float()
        
        # Numerically stable standard BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        # Calculate p_t in a numerically stable manner
        # p = sigmoid(logits)
        # For targets == 1: p_t = p, alpha_t = alpha
        # For targets == 0: p_t = 1 - p, alpha_t = 1 - alpha
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        
        # Clamping p_t prevents exact 0 or 1 edge cases from causing NaN/Inf
        p_t = torch.clamp(p_t, min=self.eps, max=1.0 - self.eps)
        
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        
        # Modulating factor (1 - p_t)^gamma
        if self.gamma == 0.0:
            focal_weight = alpha_t
        else:
            focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
            
        loss = focal_weight * bce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
            
    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, gamma={self.gamma}, reduction='{self.reduction}'"
