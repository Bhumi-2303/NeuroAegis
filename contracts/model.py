from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Type, TypeVar

import numpy as np
import torch
import torch.nn as nn

__all__ = ["SeizureModel", "StubSeizureModel", "CNNSeizureModel"]

T = TypeVar('T', bound='SeizureModel')

class SeizureModel(nn.Module, abc.ABC):
    """Abstract base class for all seizure detection models."""

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward pass of the model.
        
        Args:
            x: Input tensor.
            
        Returns:
            dict containing:
                - "logits": [batch] tensor of raw logits.
                - "attention_weights": [batch, channels, time] or None.
                - "graph_adjacency": [batch, channels, channels] or None.
                - "embeddings": pre-classifier features, for SHAP/IG.
        """
        pass

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Get probabilities from logits."""
        with torch.no_grad():
            out = self.forward(x)
            logits = out["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions."""
        probs = self.predict_proba(x)
        return (probs >= threshold).astype(np.int32)

    @property
    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    @abc.abstractmethod
    def from_config(cls: Type[T], config: Any) -> T:
        """Create model instance from configuration."""
        pass

    def save_checkpoint(
        self, 
        path: str, 
        epoch: int, 
        optimizer_state: Dict[str, Any], 
        metrics: Dict[str, Any],
        config: Any = None
    ) -> None:
        """Save a checkpoint containing model and training state."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls: Type[T], path: str, config: Any, device: str = 'cpu') -> T:
        """Load model from a saved checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls.from_config(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model


class StubSeizureModel(SeizureModel):
    """Returns random outputs. Used for pipeline testing before real architecture exists."""
    
    def __init__(self, in_channels: int = 19):
        super().__init__()
        # Add a dummy parameter so that optimizer works and parameter_count > 0
        self.dummy_param = nn.Parameter(torch.zeros(1))
        
    @classmethod
    def from_config(cls, config: Any) -> StubSeizureModel:
        in_channels = getattr(config, 'in_channels', 19)
        if isinstance(config, dict) and 'in_channels' in config:
            in_channels = config['in_channels']
        return cls(in_channels=in_channels)
        
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        batch_size = x.shape[0]
        device = x.device
        return {
            "logits": torch.randn(batch_size, device=device) + self.dummy_param,
            "attention_weights": None,
            "graph_adjacency": None,
            "embeddings": torch.randn(batch_size, 64, device=device),
        }


class CNNSeizureModel(SeizureModel):
    """Minimal 1D-CNN baseline. Conv1d -> pool -> fc -> logit.
    Accepts input shape [batch, channels, time_samples]."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Layer 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Pool to [batch, 64, 1]
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(64, 1)

    @classmethod
    def from_config(cls, config: Any) -> CNNSeizureModel:
        in_channels = getattr(config, 'in_channels', 19)
        if isinstance(config, dict) and 'in_channels' in config:
            in_channels = config['in_channels']
        return cls(in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        # x shape: [batch, channels, time_samples]
        feat = self.features(x)         # [batch, 64, 1]
        embeddings = self.flatten(feat) # [batch, 64]
        logits = self.classifier(embeddings).squeeze(-1) # [batch]
        
        return {
            "logits": logits,
            "attention_weights": None,
            "graph_adjacency": None,
            "embeddings": embeddings,
        }
