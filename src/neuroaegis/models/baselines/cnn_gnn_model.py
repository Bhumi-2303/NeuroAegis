"""
NeuroAegis Phase 4A: 1D CNN + Spatial GNN Architecture
Combines channel-preserving temporal 1D CNN feature extraction with a 2-layer Spatial Graph
Convolutional Network (GCN) over the canonical 23 bipolar scalp EEG channels.

CRITICAL PROTOCOL REQUIREMENTS:
1. Channels are NOT collapsed into a single vector prior to the GNN. Each of the 23 channels
   yields a distinct 64-dimensional temporal representation.
2. The GNN explicitly models spatial message passing across 23 nodes using normalized adjacency.
3. Outputs raw unnormalized linear logits (shape: (Batch,)). Does NOT contain a final nn.Sigmoid().
4. Operates seamlessly with BinaryFocalLossWithLogits during training and logits_to_probabilities() during inference.
"""

import os
import sys
import json
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGCNLayer(nn.Module):
    """
    Standard Graph Convolutional Network (GCN) layer implementing Kipf & Welling formulation:
        H^{(l+1)} = \hat{A} H^{(l)} W + b
    where \hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} is the normalized adjacency.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.0)
            
    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node feature tensor of shape (Batch, Num_Nodes, In_Features)
            adj_norm: Normalized adjacency matrix of shape (Num_Nodes, Num_Nodes)
        Returns:
            Propagated node features of shape (Batch, Num_Nodes, Out_Features)
        """
        # Batch matrix multiplication: (Num_Nodes, Num_Nodes) @ (Batch, Num_Nodes, In_Features)
        # Broadcasting over batch dimension -> (Batch, Num_Nodes, In_Features)
        ah = torch.matmul(adj_norm, h)
        # Linear projection: (Batch, Num_Nodes, Out_Features)
        out = self.linear(ah)
        return out


class Baseline1DCNN_GNN(nn.Module):
    """
    Controlled architectural ablation for Phase 4A:
    Channel-preserving 1D CNN temporal backbone + 2-layer Spatial GNN + Dual Pooling Readout.
    """
    def __init__(
        self,
        in_channels: int = 23,
        input_samples: int = 1280,
        node_embedding_dim: int = 64,
        gnn_hidden_dim: int = 64,
        num_classes: int = 1,
        dropout: float = 0.20,
        adj_matrix: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_samples = input_samples
        self.node_embedding_dim = node_embedding_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # 1. Channel-Preserving Temporal 1D CNN Backbone
        # Each channel is processed independently with shared 1D conv filters
        self.temporal_backbone = nn.Sequential(
            # Block 1: 1 -> 16 channels, length 1280 -> 320
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout * 0.5),
            
            # Block 2: 16 -> 32 channels, length 320 -> 80
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout * 0.5),
            
            # Block 3: 32 -> 64 channels, length 80 -> 20
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            
            # Block 4: 64 -> node_embedding_dim (64), length 20 -> 1
            nn.Conv1d(64, node_embedding_dim, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(node_embedding_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout)
        )
        
        # 2. Spatial GNN Module (2-layer GCN)
        self.gcn1 = SpatialGCNLayer(node_embedding_dim, gnn_hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        
        self.gcn2 = SpatialGCNLayer(gnn_hidden_dim, gnn_hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        
        # 3. Readout & Classification Head
        # Readout concatenates mean pooling (global background) and max pooling (focal seizure onset)
        readout_dim = gnn_hidden_dim * 2  # 64 + 64 = 128
        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, 32),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(32, num_classes)
        )
        
        # Register normalized adjacency matrix as persistent buffer
        if adj_matrix is None:
            # Default identity with self-loops
            adj_matrix = np.eye(in_channels, dtype=np.float32)
        self.register_buffer("adj_norm", torch.tensor(adj_matrix, dtype=torch.float32))
        
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
                    
    def set_adjacency(self, adj_matrix: np.ndarray):
        """Update the registered adjacency matrix."""
        t_adj = torch.tensor(adj_matrix, dtype=torch.float32, device=self.adj_norm.device)
        self.adj_norm.copy_(t_adj)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw EEG tensor of shape (Batch, 23, 1280)
        Returns:
            Raw unnormalized linear logits of shape (Batch,)
        """
        if x.dim() != 3 or x.size(1) != self.in_channels:
            raise ValueError(f"Expected input shape (Batch, {self.in_channels}, {self.input_samples}), got {tuple(x.shape)}")
            
        b, c, t = x.shape
        
        # Reshape to treat each channel as an independent 1D signal: (B * 23, 1, 1280)
        x_reshaped = x.view(b * c, 1, t)
        
        # Channel-preserving temporal feature extraction
        temp_feat = self.temporal_backbone(x_reshaped)  # (B * 23, 64, 1)
        temp_feat = temp_feat.view(b, c, self.node_embedding_dim)  # (B, 23, 64)
        
        # Spatial GNN message passing across 23 nodes
        h1 = self.gcn1(temp_feat, self.adj_norm)       # (B, 23, 64)
        h1 = self.act1(h1)
        h1 = self.drop1(h1)
        
        h2 = self.gcn2(h1, self.adj_norm)              # (B, 23, 64)
        h2 = self.act2(h2)
        h2 = self.drop2(h2)
        
        # Graph Readout Pooling: concatenation of mean and max pooling
        h_mean = h2.mean(dim=1)                        # (B, 64)
        h_max = h2.max(dim=1)[0]                       # (B, 64)
        h_graph = torch.cat([h_mean, h_max], dim=-1)   # (B, 128)
        
        # Classification Head (outputs raw unnormalized logits)
        logits = self.classifier(h_graph)              # (B, 1)
        return logits.squeeze(-1)                      # (Batch,)
        
    def get_num_parameters(self) -> int:
        """Returns total count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_architecture_json(self, filepath: str) -> dict:
        """Exports complete layer breakdown and parameter specification to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        temporal_params = sum(p.numel() for p in self.temporal_backbone.parameters() if p.requires_grad)
        gnn1_params = sum(p.numel() for p in self.gcn1.parameters() if p.requires_grad)
        gnn2_params = sum(p.numel() for p in self.gcn2.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        total_params = self.get_num_parameters()
        
        arch_dict = {
            "model_name": "Baseline1DCNN_GNN",
            "phase": "Phase 4A",
            "hypothesis": "Does explicitly modeling spatial relationships between the 23 EEG channels improve seizure detection compared with the Phase 3 temporal-only 1D CNN?",
            "input_shape": [None, self.in_channels, self.input_samples],
            "channel_preservation": "Each of the 23 channels is encoded independently via shared 1D conv blocks into a 64-d node representation",
            "node_features": 64,
            "gnn_type": "Spatial Graph Convolutional Network (GCN)",
            "gnn_layers": 2,
            "gnn_hidden_dim": self.gnn_hidden_dim,
            "message_passing": "H^(l+1) = GELU(A_hat * H^(l) * W + b)",
            "readout": "Dual Global Pooling (Concat of Mean and Max over 23 nodes) -> 128-d graph representation",
            "head": "Linear(128, 32) -> GELU -> Dropout(0.3) -> Linear(32, 1)",
            "output": "Raw unnormalized scalar linear logit (NO sigmoid)",
            "total_parameters": total_params,
            "layer_breakdown": {
                "temporal_backbone_1d_cnn": {
                    "description": "Shared 4-stage 1D CNN across 23 channels",
                    "parameters": temporal_params,
                    "stages": [
                        "Conv1d(1, 16, k=15, s=2, p=7) + BN + GELU + MaxPool(2)",
                        "Conv1d(16, 32, k=9, s=2, p=4) + BN + GELU + MaxPool(2)",
                        "Conv1d(32, 64, k=7, s=2, p=3) + BN + GELU + MaxPool(2)",
                        "Conv1d(64, 64, k=5, s=1, p=2) + BN + GELU + AdaptiveAvgPool(1)"
                    ]
                },
                "spatial_gnn_gcn1": {
                    "description": "GCN Layer 1 (64 -> 64)",
                    "parameters": gnn1_params
                },
                "spatial_gnn_gcn2": {
                    "description": "GCN Layer 2 (64 -> 64)",
                    "parameters": gnn2_params
                },
                "classification_head": {
                    "description": "Linear(128, 32) + GELU + Dropout(0.3) + Linear(32, 1)",
                    "parameters": classifier_params
                }
            }
        }
        
        with open(filepath, "w") as f:
            json.dump(arch_dict, f, indent=2)
            
        return arch_dict
