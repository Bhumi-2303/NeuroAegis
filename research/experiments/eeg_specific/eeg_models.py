"""
research/experiments/eeg_specific/eeg_models.py
───────────────────────────────────────────────
Standard lightweight EEG-specific deep learning architectures for Experiment 2:
1. EEGNet (Lawhern et al., 2018)
2. ShallowConvNet (Schirrmeister et al., 2017)
3. Lightweight DeepConvNet (Schirrmeister et al., 2017)
4. Lightweight 1D CNN (Phase 3 Multi-Scale Baseline)

All models take raw preprocessed EEG windows of shape (Batch, 23, 1280)
and output binary logits of shape (Batch,).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.phase_3.cnn_model import Baseline1DCNN


class EEGNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 23,
        samples: int = 1280,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_len: int = 64,
        dropout: float = 0.25
    ):
        super().__init__()
        self.in_channels = in_channels
        self.samples = samples
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # Block 1: Temporal Conv + Depthwise Spatial Conv
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_len), padding=(0, kernel_len // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable Conv (Depthwise + Pointwise)
        self.separable = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, samples)
            h = self.drop1(self.pool1(self.act1(self.bn2(self.depthwise(self.bn1(self.conv1(dummy)))))))
            h = self.drop2(self.pool2(self.act2(self.bn3(self.pointwise(self.separable(h))))))
            flat_dim = h.view(1, -1).shape[1]

        self.classifier = nn.Linear(flat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.depthwise(h)
        h = self.bn2(h)
        h = self.act1(h)
        h = self.pool1(h)
        h = self.drop1(h)

        h = self.separable(h)
        h = self.pointwise(h)
        h = self.bn3(h)
        h = self.act2(h)
        h = self.pool2(h)
        h = self.drop2(h)

        h_flat = h.view(h.size(0), -1)
        logits = self.classifier(h_flat).squeeze(-1)
        return logits

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 23,
        samples: int = 1280,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        dropout: float = 0.50
    ):
        super().__init__()
        self.conv_time = nn.Conv2d(1, n_filters_time, (1, filter_time_length), bias=False)
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_time, (in_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters_time)
        self.pool = nn.AvgPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        self.drop = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, samples)
            h = self.conv_spat(self.conv_time(dummy))
            h = self.bn(h)
            h = h ** 2
            h = self.pool(h)
            h = torch.log(torch.clamp(h, min=1e-6))
            h = self.drop(h)
            flat_dim = h.view(1, -1).shape[1]

        self.classifier = nn.Linear(flat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.conv_time(x)
        h = self.conv_spat(h)
        h = self.bn(h)
        h = h ** 2
        h = self.pool(h)
        h = torch.log(torch.clamp(h, min=1e-6))
        h = self.drop(h)
        h_flat = h.view(h.size(0), -1)
        logits = self.classifier(h_flat).squeeze(-1)
        return logits

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LightweightDeepConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 23,
        samples: int = 1280,
        n_filters_first: int = 25,
        n_filters_later: int = 50,
        filter_length: int = 10,
        pool_length: int = 3,
        dropout: float = 0.25
    ):
        super().__init__()
        self.conv1_time = nn.Conv2d(1, n_filters_first, (1, filter_length), bias=False)
        self.conv1_spat = nn.Conv2d(n_filters_first, n_filters_first, (in_channels, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters_first)
        self.pool1 = nn.MaxPool2d((1, pool_length), stride=(1, pool_length))
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(n_filters_first, n_filters_later, (1, filter_length), bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters_later)
        self.pool2 = nn.MaxPool2d((1, pool_length), stride=(1, pool_length))
        self.drop2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(n_filters_later, n_filters_later * 2, (1, filter_length), bias=False)
        self.bn3 = nn.BatchNorm2d(n_filters_later * 2)
        self.pool3 = nn.MaxPool2d((1, pool_length), stride=(1, pool_length))
        self.drop3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv2d(n_filters_later * 2, n_filters_later * 2, (1, filter_length), bias=False)
        self.bn4 = nn.BatchNorm2d(n_filters_later * 2)
        self.pool4 = nn.MaxPool2d((1, pool_length), stride=(1, pool_length))
        self.drop4 = nn.Dropout(dropout)

        self.act = nn.ELU()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, samples)
            h = self.drop1(self.pool1(self.act(self.bn1(self.conv1_spat(self.conv1_time(dummy))))))
            h = self.drop2(self.pool2(self.act(self.bn2(self.conv2(h)))))
            h = self.drop3(self.pool3(self.act(self.bn3(self.conv3(h)))))
            h = self.drop4(self.pool4(self.act(self.bn4(self.conv4(h)))))
            flat_dim = h.view(1, -1).shape[1]

        self.classifier = nn.Linear(flat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.conv1_time(x)
        h = self.conv1_spat(h)
        h = self.bn1(h)
        h = self.act(h)
        h = self.pool1(h)
        h = self.drop1(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.act(h)
        h = self.pool2(h)
        h = self.drop2(h)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.act(h)
        h = self.pool3(h)
        h = self.drop3(h)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.act(h)
        h = self.pool4(h)
        h = self.drop4(h)

        h_flat = h.view(h.size(0), -1)
        logits = self.classifier(h_flat).squeeze(-1)
        return logits

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_eeg_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "eegnet":
        return EEGNet(in_channels=23, samples=1280)
    elif name == "shallowconvnet":
        return ShallowConvNet(in_channels=23, samples=1280)
    elif name == "deepconvnet":
        return LightweightDeepConvNet(in_channels=23, samples=1280)
    elif name in ["1dcnn", "cnn1d"]:
        return Baseline1DCNN(in_channels=23, num_classes=1)
    else:
        raise ValueError(f"Unknown model name: {name}")
