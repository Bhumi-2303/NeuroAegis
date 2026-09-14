"""
NeuroAegis Phase 3: 1D CNN Baseline Architecture
Multi-channel temporal 1D Convolutional Neural Network for CHB-MIT Seizure Detection.
Processes canonical 23 bipolar EEG channels across 5.0-second windows (1280 samples @ 256 Hz).
CRITICAL PROTOCOL REQUIREMENT:
Outputs unnormalized linear logits. Does NOT contain a final nn.Sigmoid() layer.
Numerical stability is maintained via BinaryFocalLossWithLogits during training,
and logits_to_probabilities() during inference.
"""

import torch
import torch.nn as nn


class Baseline1DCNN(nn.Module):
    """
    Standard 1D CNN temporal baseline for continuous 23-channel scalp EEG.
    """
    def __init__(self, in_channels: int = 23, num_classes: int = 1, dropout: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # 4-stage temporal feature extractor
        self.features = nn.Sequential(
            # Stage 1: Broad receptive field for low-frequency rhythmic discharge
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout * 0.5),
            
            # Stage 2: Intermediate temporal dynamics
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout * 0.5),
            
            # Stage 3: Higher-level temporal-spectral patterns
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            
            # Stage 4: High-dimensional abstraction & temporal pooling
            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout)
        )
        
        # Linear classification head (outputs raw unnormalized logits)
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, 23, 1280)
        Returns:
            Raw unnormalized logits of shape (Batch,)
        """
        if x.dim() != 3 or x.size(1) != self.in_channels:
            raise ValueError(f"Expected input shape (Batch, {self.in_channels}, Time), got {tuple(x.shape)}")
            
        feat = self.features(x)         # (Batch, 128, 1)
        feat = feat.squeeze(-1)         # (Batch, 128)
        logits = self.classifier(feat)  # (Batch, 1)
        return logits.squeeze(-1)       # (Batch,)
        
    def get_num_parameters(self) -> int:
        """Returns total count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
