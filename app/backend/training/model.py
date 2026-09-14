"""
PyTorch modules for the NeuroAegis attention pooling architecture.
"""
import torch
import torch.nn as nn
from typing import Tuple

class ChannelEncoder(nn.Module):
    """
    Encodes single channel features into a dense embedding.
    """
    def __init__(self, input_dim: int = 57, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Tensor of shape [..., input_dim].
            
        Returns:
            torch.Tensor: Encoded features of shape [..., embed_dim].
        """
        return self.net(x)

class AttentionPooling(nn.Module):
    """
    Attention pooling over multiple channels.
    """
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.scoring_mlp = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            embeddings (torch.Tensor): Set of channel embeddings of shape [B, N, embed_dim].
            mask (torch.Tensor): Boolean mask of shape [B, N] (True = valid, False = padded).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pooled vector of shape [B, embed_dim], and attention weights of shape [B, N].
        """
        # [B, N, 1]
        scores = self.scoring_mlp(embeddings)
        # [B, N]
        scores = scores.squeeze(-1)
        
        # Mask out padded positions
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # [B, N]
        alpha = torch.softmax(scores, dim=-1)
        
        # [B, embed_dim]
        pooled = torch.sum(alpha.unsqueeze(-1) * embeddings, dim=1)
        
        return pooled, alpha

class SeizureClassifier(nn.Module):
    """
    Classifier for seizure detection based on the pooled embedding.
    """
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z (torch.Tensor): Pooled vector of shape [B, embed_dim].
            
        Returns:
            torch.Tensor: Seizure probability of shape [B, 1].
        """
        return self.net(z)

class AttentionSeizureDetector(nn.Module):
    """
    End-to-end attention pooling model for seizure detection.
    """
    def __init__(self, input_dim: int = 57, embed_dim: int = 32):
        super().__init__()
        self.encoder = ChannelEncoder(input_dim, embed_dim)
        self.pooling = AttentionPooling(embed_dim)
        self.classifier = SeizureClassifier(embed_dim)
        
    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features (torch.Tensor): Features of shape [B, N_max, input_dim].
            mask (torch.Tensor): Boolean mask of shape [B, N_max].
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Prediction [B, 1] and attention weights [B, N_max].
        """
        # [B, N_max, embed_dim]
        embeddings = self.encoder(features)
        
        # [B, embed_dim], [B, N_max]
        pooled, alpha = self.pooling(embeddings, mask)
        
        # [B, 1]
        prediction = self.classifier(pooled)
        
        return prediction, alpha
        
    def compute_entropy_loss(self, alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the entropy loss of the attention weights to encourage sparsity.
        
        Args:
            alpha (torch.Tensor): Attention weights of shape [B, N].
            mask (torch.Tensor): Boolean mask of shape [B, N].
            
        Returns:
            torch.Tensor: Entropy loss.
        """
        eps = 1e-8
        
        # Only compute on valid positions
        valid_alpha = alpha * mask.float()
        
        entropy = -torch.sum(valid_alpha * torch.log(valid_alpha + eps), dim=-1)
        
        # Average over batch
        return torch.mean(entropy)
