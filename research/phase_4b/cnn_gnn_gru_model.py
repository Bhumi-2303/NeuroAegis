"""
NeuroAegis Phase 4B: CNN + Spatial GNN + Temporal GRU Architecture
Combines frozen 1D CNN + Spatial GNN spatial feature extractor with a causal,
unidirectional GRU for temporal sequence modeling across consecutive EEG windows.

CRITICAL REQUIREMENTS:
1. Frozen Backbone: Reuses the exact Phase 4A-C CNN + GNN architecture with θ=0.30 graph.
   Backbone weights are strictly frozen (requires_grad = False).
2. Causal Unidirectional GRU: Hidden state at window t depends ONLY on window t and windows
   before t. Does not access future windows (bidirectional = False).
3. Dual Readout Input: Operates on sequences of 128-dimensional spatial embeddings produced by
   concatenating global mean pooling and global max pooling over the 23 GNN nodes.
4. Output: Raw unnormalized linear logits (shape: (Batch,)). No final sigmoid.
5. Real-Time Streaming Support: Supports incremental step-by-step inference with explicit state management.
"""

import os
import sys
import json
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.phase_4a.cnn_gnn_model import Baseline1DCNN_GNN


class CNN_GNN_GRU(nn.Module):
    """
    Phase 4B Causal Temporal Architecture:
    Frozen CNN + GNN Backbone -> Unidirectional GRU -> Lightweight Classifier Head.
    """
    def __init__(
        self,
        frozen_backbone_path: str = "/Volumes/BLACK-BOX/NeuroAegis/research/phase_4a/frozen_cnn_gnn.pt",
        gru_input_dim: int = 128,
        gru_hidden_dim: int = 64,
        gru_layers: int = 1,
        dropout_classifier: float = 0.30,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.gru_input_dim = gru_input_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_layers = gru_layers
        self.dropout_classifier = dropout_classifier
        self.frozen_backbone_path = frozen_backbone_path

        # 1. Load and Freeze Phase 4A-C CNN + GNN Backbone
        self.backbone = Baseline1DCNN_GNN(in_channels=23, num_classes=1)
        if os.path.exists(frozen_backbone_path):
            ckpt = torch.load(frozen_backbone_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.backbone.load_state_dict(state_dict)
            self.backbone_loaded = True
        else:
            self.backbone_loaded = False
            
        # Freeze ALL backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # 2. Causal Unidirectional GRU
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )

        # 3. Lightweight Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_classifier),
            nn.Linear(32, 1)
        )

        self._init_head_weights()
        if device is not None:
            self.to(device)

    def _init_head_weights(self):
        """Initializes GRU and classification head weights."""
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_backbone_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 128-d spatial embedding from raw EEG window(s).
        
        Args:
            x: Raw EEG tensor of shape (Batch, 23, 1280)
        Returns:
            Spatial embeddings of shape (Batch, 128)
        """
        b, c, t = x.shape
        x_reshaped = x.view(b * c, 1, t)
        
        with torch.no_grad():
            temp_feat = self.backbone.temporal_backbone(x_reshaped)
            temp_feat = temp_feat.view(b, c, self.backbone.node_embedding_dim)
            h1 = self.backbone.drop1(self.backbone.act1(self.backbone.gcn1(temp_feat, self.backbone.adj_norm)))
            h2 = self.backbone.drop2(self.backbone.act2(self.backbone.gcn2(h1, self.backbone.adj_norm)))
            h_mean = h2.mean(dim=1)
            h_max = h2.max(dim=1)[0]
            h_graph = torch.cat([h_mean, h_max], dim=-1)
            
        return h_graph

    def forward_from_embeddings(self, seq_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from precomputed sequence embeddings.
        
        Args:
            seq_embeddings: Tensor of shape (Batch, Seq_Len, 128)
        Returns:
            Logits of shape (Batch,)
        """
        # GRU forward: output shape (Batch, Seq_Len, Hidden_Dim)
        gru_out, _ = self.gru(seq_embeddings)
        
        # Causal extraction: take the final hidden state corresponding to the current window
        h_final = gru_out[:, -1, :]  # (Batch, Hidden_Dim)
        
        logits = self.classifier(h_final).squeeze(-1)  # (Batch,)
        return logits

    def forward_raw(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from raw EEG sequences.
        
        Args:
            x_seq: Tensor of shape (Batch, Seq_Len, 23, 1280)
        Returns:
            Logits of shape (Batch,)
        """
        b, seq_len, c, t = x_seq.shape
        x_flat = x_seq.view(b * seq_len, c, t)
        
        embeddings = self.extract_backbone_embedding(x_flat)
        seq_embeddings = embeddings.view(b, seq_len, self.gru_input_dim)
        
        return self.forward_from_embeddings(seq_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flexible forward pass: detects input dimensions.
        (Batch, Seq_Len, 128) -> forward_from_embeddings
        (Batch, Seq_Len, 23, 1280) -> forward_raw
        """
        if x.dim() == 3:
            return self.forward_from_embeddings(x)
        elif x.dim() == 4:
            return self.forward_raw(x)
        else:
            raise ValueError(f"Expected 3D (B, L, 128) or 4D (B, L, 23, 1280) tensor, got shape {tuple(x.shape)}")

    def step(
        self,
        window_embedding: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Incremental step for real-time streaming inference.
        
        Args:
            window_embedding: Spatial embedding of current window (Batch, 128)
            hidden_state: Previous GRU hidden state (1, Batch, Hidden_Dim) or None
        Returns:
            (logit_t, next_hidden_state)
        """
        # Add sequence length 1: (Batch, 1, 128)
        x_step = window_embedding.unsqueeze(1)
        gru_out, next_hidden = self.gru(x_step, hidden_state)
        
        h_t = gru_out[:, -1, :]
        logit_t = self.classifier(h_t).squeeze(-1)
        return logit_t, next_hidden

    def get_parameter_breakdown(self) -> Dict[str, Any]:
        """Calculates precise parameter counts and memory usage."""
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        
        gru_params = sum(p.numel() for p in self.gru.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        total_trainable = gru_params + classifier_params
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "model_name": "CNN_GNN_GRU",
            "backbone_frozen_params": backbone_total,
            "backbone_trainable_params": backbone_trainable,
            "gru_trainable_params": gru_params,
            "classifier_trainable_params": classifier_params,
            "total_trainable_params": total_trainable,
            "total_all_params": total_params,
            "parameter_memory_mb": round(total_params * 4 / (1024 * 1024), 3),
            "trainable_memory_mb": round(total_trainable * 4 / (1024 * 1024), 3)
        }
