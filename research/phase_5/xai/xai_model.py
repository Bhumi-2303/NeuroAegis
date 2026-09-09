"""
NeuroAegis Phase 5: Differentiable XAI Model Wrapper
Wraps the frozen Phase 4B CNN_GNN_GRU model to expose fully differentiable
forward passes for raw EEG input tensors, intermediate node representations,
and graph adjacency structures without modifying any frozen checkpoint weights.
"""

import os
import sys
import json
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from research.phase_4b.cnn_gnn_gru_model import CNN_GNN_GRU


class XAIModelWrapper(nn.Module):
    """
    Differentiable wrapper for the frozen Phase 4B model (1D CNN + Spatial GNN + Causal GRU).
    Enables exact gradient backpropagation from output logits to raw EEG input tensors,
    intermediate node representations, and graph adjacency structures.
    """
    def __init__(
        self,
        frozen_checkpoint_path: str = os.path.join(BASE_DIR, "research/phase_4b/frozen_cnn_gnn_gru.pt"),
        frozen_backbone_path: str = os.path.join(BASE_DIR, "research/phase_4a/frozen_cnn_gnn.pt"),
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.frozen_checkpoint_path = frozen_checkpoint_path
        self.frozen_backbone_path = frozen_backbone_path
        
        # Instantiate frozen base model
        self.base_model = CNN_GNN_GRU(frozen_backbone_path=frozen_backbone_path)
        
        # Load Phase 4B trained weights
        if os.path.exists(frozen_checkpoint_path):
            ckpt = torch.load(frozen_checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.base_model.load_state_dict(state_dict)
            self.loaded = True
        else:
            raise FileNotFoundError(f"Phase 4B checkpoint not found at: {frozen_checkpoint_path}")
            
        # Strictly freeze ALL parameters
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.base_model.eval()
        
        # Store device
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        self.to(self.device)
        self.eval()

    @property
    def backbone(self):
        return self.base_model.backbone

    @property
    def gru(self):
        return self.base_model.gru

    @property
    def classifier(self):
        return self.base_model.classifier

    @property
    def adj_norm(self):
        return self.base_model.backbone.adj_norm

    def forward_from_embeddings(self, seq_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass from precomputed 128-d embeddings.
        Shape: (Batch, Seq_Len, 128) -> (Batch,) logits
        """
        return self.base_model.forward_from_embeddings(seq_embeddings)

    def forward_differentiable(
        self,
        x_seq: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        End-to-end differentiable forward pass taking raw EEG sequences directly to logits.
        
        Args:
            x_seq: Tensor of shape (Batch, Seq_Len, 23, 1280)
            adj_matrix: Optional custom/perturbed normalized adjacency (23, 23).
                       If None, uses self.adj_norm.
        Returns:
            Logits of shape (Batch,)
        """
        b, seq_len, c, t = x_seq.shape
        x_flat = x_seq.view(b * seq_len * c, 1, t)
        
        adj = self.adj_norm if adj_matrix is None else adj_matrix
        
        # 1. 1D Temporal CNN
        temp_feat = self.backbone.temporal_backbone(x_flat)
        temp_feat = temp_feat.view(b * seq_len, c, self.backbone.node_embedding_dim)
        
        # 2. Spatial GNN (eval mode: no dropout)
        h1 = self.backbone.act1(self.backbone.gcn1(temp_feat, adj))
        h2 = self.backbone.act2(self.backbone.gcn2(h1, adj))
        
        # 3. Dual Readout Pooling
        h_mean = h2.mean(dim=1)
        h_max = h2.max(dim=1)[0]
        h_graph = torch.cat([h_mean, h_max], dim=-1)  # (b * seq_len, 128)
        
        # 4. Causal GRU
        seq_embeddings = h_graph.view(b, seq_len, 128)
        gru_out, _ = self.gru(seq_embeddings)
        h_final = gru_out[:, -1, :]  # (b, 64)
        
        # 5. Classifier (eval mode: no dropout)
        logits = self.classifier(h_final).squeeze(-1)
        return logits

    def forward_with_intermediates(
        self,
        x_seq: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning intermediate representations for GNN and node-level XAI.
        """
        b, seq_len, c, t = x_seq.shape
        x_flat = x_seq.view(b * seq_len * c, 1, t)
        
        adj = self.adj_norm if adj_matrix is None else adj_matrix
        
        temp_feat = self.backbone.temporal_backbone(x_flat)
        temp_feat = temp_feat.view(b * seq_len, c, self.backbone.node_embedding_dim)
        
        h1 = self.backbone.act1(self.backbone.gcn1(temp_feat, adj))
        h2 = self.backbone.act2(self.backbone.gcn2(h1, adj))
        
        h_mean = h2.mean(dim=1)
        h_max = h2.max(dim=1)[0]
        h_graph = torch.cat([h_mean, h_max], dim=-1)
        
        seq_embeddings = h_graph.view(b, seq_len, 128)
        gru_out, _ = self.gru(seq_embeddings)
        h_final = gru_out[:, -1, :]
        
        logits = self.classifier(h_final).squeeze(-1)
        
        return {
            "logits": logits,
            "cnn_features": temp_feat,
            "gnn_h1": h1,
            "gnn_h2": h2,
            "seq_embeddings": seq_embeddings,
            "gru_output": gru_out,
            "h_final": h_final
        }

    def verify_prediction_parity(
        self,
        x_seq: torch.Tensor,
        cached_embedding_seq: torch.Tensor,
        tolerance: float = 1e-4
    ) -> Tuple[bool, float]:
        """
        Verifies that forward_differentiable matches forward_from_embeddings
        within numerical floating point precision.
        """
        with torch.no_grad():
            logit_diff = self.forward_differentiable(x_seq)
            logit_cached = self.forward_from_embeddings(cached_embedding_seq)
            
            diff = (logit_diff - logit_cached).abs().max().item()
            is_valid = diff <= tolerance
            return is_valid, diff
