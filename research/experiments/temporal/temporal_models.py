"""
research/experiments/temporal/temporal_models.py
────────────────────────────────────────────────
Temporal sequence architectures for Experiment 1:
1. Reference GRU (hidden=64, 1 layer, unidirectional)
2. Causal LSTM (hidden=64, 1 layer, unidirectional)
3. Lightweight Causal TCN (1-3 dilated causal residual blocks, hidden=64)

All models receive sequences of 128-D spatial embeddings (Batch, 8, 128)
extracted from the frozen CNN + Spatial GNN backbone and output binary logits
through an identical classification head:
Linear(64, 32) -> ReLU -> Dropout(0.30) -> Linear(32, 1)
"""

import math
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    1D Convolution with causal left-padding.
    Guarantees output at step t has zero receptive field into steps > t.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Channels, Seq_Len)
        # Pad left side only by causal_padding, 0 on right side
        x_padded = F.pad(x, (self.causal_padding, 0))
        return self.conv(x_padded)


class TemporalResidualBlock(nn.Module):
    """
    Causal Dilated Residual Block for TCN.
    Two causal dilated convolutions with batch normalization, ReLU, and dropout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.10
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.downsample = nn.Identity()

        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        return self.final_relu(out + residual)


def build_classifier_head(in_features: int = 64, dropout: float = 0.30) -> nn.Sequential:
    """Standardized classification head identical to Model C."""
    head = nn.Sequential(
        nn.Linear(in_features, 32),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(32, 1)
    )
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return head


class CausalGRUModel(nn.Module):
    """
    Reference Model C Temporal Head:
    Causal Unidirectional GRU (hidden=64, 1 layer) + Standard Classifier Head.
    """
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout_classifier: float = 0.30
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )
        self.classifier = build_classifier_head(in_features=hidden_dim, dropout=dropout_classifier)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq_Len, input_dim)
        gru_out, _ = self.gru(x)
        # Causal readout: final time step
        h_final = gru_out[:, -1, :]
        logits = self.classifier(h_final).squeeze(-1)
        return logits

    def get_parameter_breakdown(self, frozen_backbone_params: int = 52497) -> Dict[str, Any]:
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = trainable_params + frozen_backbone_params
        return {
            "model_name": "Causal_GRU",
            "trainable_parameters": trainable_params,
            "frozen_backbone_parameters": frozen_backbone_params,
            "total_parameters": total_params,
            "parameter_memory_mb": round(total_params * 4 / (1024 * 1024), 3)
        }


class CausalLSTMModel(nn.Module):
    """
    Alternative Recurrent Architecture:
    Causal Unidirectional LSTM (hidden=64, 1 layer) + Standard Classifier Head.
    """
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout_classifier: float = 0.30
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )
        self.classifier = build_classifier_head(in_features=hidden_dim, dropout=dropout_classifier)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq_Len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Causal readout: final time step
        h_final = lstm_out[:, -1, :]
        logits = self.classifier(h_final).squeeze(-1)
        return logits

    def get_parameter_breakdown(self, frozen_backbone_params: int = 52497) -> Dict[str, Any]:
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = trainable_params + frozen_backbone_params
        return {
            "model_name": "Causal_LSTM",
            "trainable_parameters": trainable_params,
            "frozen_backbone_parameters": frozen_backbone_params,
            "total_parameters": total_params,
            "parameter_memory_mb": round(total_params * 4 / (1024 * 1024), 3)
        }


class CausalTCNModel(nn.Module):
    """
    Alternative Convolutional Architecture:
    Lightweight Causal Dilated TCN (2 residual blocks, hidden=64) + Standard Classifier Head.
    Receptive field strictly covers the sequence length causally with zero future leakage.
    """
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        dropout_tcn: float = 0.10,
        dropout_classifier: float = 0.30
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection 128 -> 64
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # 2 Dilated Causal Residual Blocks
        # Block 1: dilation 1 (receptive field span 4)
        # Block 2: dilation 2 (receptive field span 8)
        # Total receptive field = 1 + 2*(3-1)*1 + 2*(3-1)*2 = 13 timesteps >= 8
        self.block1 = TemporalResidualBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=1,
            dropout=dropout_tcn
        )
        self.block2 = TemporalResidualBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=2,
            dropout=dropout_tcn
        )

        self.classifier = build_classifier_head(in_features=hidden_dim, dropout=dropout_classifier)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq_Len, input_dim) -> transpose to (Batch, input_dim, Seq_Len)
        x_trans = x.transpose(1, 2)
        h = self.proj(x_trans)
        h = self.block1(h)
        h = self.block2(h)

        # Causal readout: final time step (Batch, hidden_dim)
        h_final = h[:, :, -1]
        logits = self.classifier(h_final).squeeze(-1)
        return logits

    def get_parameter_breakdown(self, frozen_backbone_params: int = 52497) -> Dict[str, Any]:
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = trainable_params + frozen_backbone_params
        return {
            "model_name": "Causal_TCN",
            "trainable_parameters": trainable_params,
            "frozen_backbone_parameters": frozen_backbone_params,
            "total_parameters": total_params,
            "parameter_memory_mb": round(total_params * 4 / (1024 * 1024), 3)
        }
