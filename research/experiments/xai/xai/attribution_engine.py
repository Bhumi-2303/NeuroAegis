"""
NeuroAegis Phase 5: Attribution Engine
Implements Integrated Gradients, Gradient x Input, GNN node attribution,
edge sensitivity analysis, and multi-dimensional attribution aggregation
(temporal, channel, sequence-step).
"""

import os
import sys
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from research.experiments.xai.xai.xai_model import XAIModelWrapper


class AttributionEngine:
    """
    Computes rigorous XAI attributions for the frozen CNN + GNN + Causal GRU model.
    """
    def __init__(
        self,
        model: XAIModelWrapper,
        canonical_channels_path: str = os.path.join(BASE_DIR, "research/data/config/chbmit_channel_order.json")
    ):
        self.model = model
        self.device = model.device
        
        # Load authoritative canonical channel order
        if os.path.exists(canonical_channels_path):
            import json
            with open(canonical_channels_path, "r") as f:
                self.channel_names = json.load(f)
        else:
            raise FileNotFoundError(f"Channel config not found at: {canonical_channels_path}")
            
        assert len(self.channel_names) == 23, f"Expected 23 channels, got {len(self.channel_names)}"

    def compute_integrated_gradients(
        self,
        x_seq: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 25
    ) -> Dict[str, Any]:
        """
        Computes Integrated Gradients (Sundararajan et al., 2017) w.r.t raw EEG sequence.
        
        Formula:
            IG_i(x) = (x_i - x'_i) * (1/m) * \sum_{k=1}^m \nabla_{x_i} F(x' + (k/m)*(x - x'))
            
        Args:
            x_seq: (1, 8, 23, 1280) raw EEG tensor
            baseline: (1, 8, 23, 1280) baseline tensor (defaults to zero baseline)
            steps: Number of Riemann interpolation steps (m)
            
        Returns:
            Dict with attribution tensor, completeness delta, and prediction logits.
        """
        self.model.eval()
        x_seq = x_seq.to(self.device).float()
        
        if baseline is None:
            baseline = torch.zeros_like(x_seq)
        else:
            baseline = baseline.to(self.device).float()
            
        with torch.no_grad():
            logit_input = self.model.forward_differentiable(x_seq).item()
            logit_baseline = self.model.forward_differentiable(baseline).item()
            prob_input = torch.sigmoid(torch.tensor(logit_input)).item()
            prob_baseline = torch.sigmoid(torch.tensor(logit_baseline)).item()
            
        delta = x_seq - baseline
        accumulated_grads = torch.zeros_like(x_seq)
        
        # Linear interpolation from baseline to input
        alphas = torch.linspace(1.0 / steps, 1.0, steps, device=self.device)
        
        try:
            for alpha in alphas:
                x_step = baseline + alpha * delta
                x_step.requires_grad_(True)
                
                logit = self.model.forward_differentiable(x_step)
                logit.backward()
                
                with torch.no_grad():
                    if x_step.grad is not None:
                        accumulated_grads += x_step.grad
                self.model.zero_grad(set_to_none=True)
                del x_step, logit
                    
            avg_grads = accumulated_grads / steps
            attribution = (delta * avg_grads).detach()
        finally:
            self.model.zero_grad(set_to_none=True)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        # Completeness check: sum(IG) should approximate logit_input - logit_baseline
        ig_sum = attribution.sum().item()
        target_diff = logit_input - logit_baseline
        completeness_delta = abs(ig_sum - target_diff)
        
        return {
            "attribution": attribution,  # (1, 8, 23, 1280)
            "logit_input": logit_input,
            "logit_baseline": logit_baseline,
            "prob_input": prob_input,
            "prob_baseline": prob_baseline,
            "completeness_delta": completeness_delta,
            "completeness_relative_error": completeness_delta / (abs(target_diff) + 1e-7),
            "method": "Integrated Gradients",
            "steps": steps
        }

    def compute_gradient_x_input(
        self,
        x_seq: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Computes lightweight Gradient x Input attribution.
        
        Formula:
            GI_i(x) = x_i * \nabla_{x_i} F(x)
        """
        self.model.eval()
        x_seq = x_seq.to(self.device).float().clone()
        x_seq.requires_grad_(True)
        
        try:
            logit = self.model.forward_differentiable(x_seq)
            logit.backward()
            
            grad = x_seq.grad.detach() if x_seq.grad is not None else torch.zeros_like(x_seq)
            attribution = (x_seq.detach() * grad)
            
            logit_val = logit.item()
            prob_val = torch.sigmoid(torch.tensor(logit_val)).item()
            del logit
        finally:
            self.model.zero_grad(set_to_none=True)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        return {
            "attribution": attribution,  # (1, 8, 23, 1280)
            "logit_input": logit_val,
            "prob_input": prob_val,
            "method": "Gradient x Input"
        }


    def compute_gnn_node_attribution(
        self,
        x_seq: torch.Tensor
    ) -> np.ndarray:
        """
        Computes attribution w.r.t the 23 GNN node embeddings (H^(2)) before pooling.
        
        Returns:
            np.ndarray of shape (23,) representing spatial node importance.
        """
        self.model.eval()
        x_seq = x_seq.to(self.device).float()
        
        b, seq_len, c, t = x_seq.shape
        x_flat = x_seq.view(b * seq_len * c, 1, t)
        
        with torch.no_grad():
            temp_feat = self.model.backbone.temporal_backbone(x_flat)
            temp_feat = temp_feat.view(b * seq_len, c, self.model.backbone.node_embedding_dim)
            h1 = self.model.backbone.act1(self.model.backbone.gcn1(temp_feat, self.model.adj_norm))
            
        h1.requires_grad_(True)
        h2 = self.model.backbone.act2(self.model.backbone.gcn2(h1, self.model.adj_norm))
        
        h_mean = h2.mean(dim=1)
        h_max = h2.max(dim=1)[0]
        h_graph = torch.cat([h_mean, h_max], dim=-1)
        
        seq_embeddings = h_graph.view(b, seq_len, 128)
        gru_out, _ = self.model.gru(seq_embeddings)
        h_final = gru_out[:, -1, :]
        
        logit = self.model.classifier(h_final).squeeze(-1)
        logit.backward()
        
        # Gradient w.r.t node features
        h2_grad = h2.grad if h2.grad is not None else torch.autograd.grad(logit, h2, retain_graph=True)[0]
        node_attr = (h2 * h2_grad).abs().sum(dim=-1).mean(dim=0).detach().cpu().numpy()
        return node_attr

    def compute_spatial_edge_sensitivity(
        self,
        x_seq: torch.Tensor
    ) -> np.ndarray:
        """
        Computes model sensitivity to edge weights: |d(logit) / d(adj_norm_ij)|.
        Strictly distinguishes graph connectivity from model gradient sensitivity.
        
        Returns:
            np.ndarray of shape (23, 23)
        """
        self.model.eval()
        x_seq = x_seq.to(self.device).float()
        
        adj_var = self.model.adj_norm.clone().detach().requires_grad_(True)
        logit = self.model.forward_differentiable(x_seq, adj_matrix=adj_var)
        logit.backward()
        
        edge_sensitivity = adj_var.grad.abs().detach().cpu().numpy()
        return edge_sensitivity

    def aggregate_channel_attribution(
        self,
        attribution_4d: torch.Tensor
    ) -> pd.DataFrame:
        """
        Aggregates attribution across all time samples and sequence steps for each channel.
        
        Args:
            attribution_4d: (1, 8, 23, 1280)
        Returns:
            DataFrame with columns: [channel_name, attribution_score, normalized_attribution, rank]
        """
        # Sum absolute attribution across steps (dim 1) and time (dim 3)
        # Result shape: (23,)
        attr_cpu = attribution_4d.squeeze(0).cpu().abs()  # (8, 23, 1280)
        channel_scores = attr_cpu.sum(dim=(0, 2)).numpy()  # (23,)
        
        total_score = channel_scores.sum() + 1e-12
        normalized_scores = channel_scores / total_score
        
        df = pd.DataFrame({
            "channel_name": self.channel_names,
            "attribution_score": channel_scores,
            "normalized_attribution": normalized_scores
        })
        
        # Rank: 1 = highest attribution
        df["rank"] = df["attribution_score"].rank(ascending=False, method="min").astype(int)
        df = df.sort_values("rank").reset_index(drop=True)
        return df

    def aggregate_temporal_attribution(
        self,
        attribution_4d: torch.Tensor,
        target_window_only: bool = True
    ) -> np.ndarray:
        """
        Aggregates absolute attribution across channels for each time sample.
        
        Args:
            attribution_4d: (1, 8, 23, 1280)
            target_window_only: If True, returns (1280,) for current window (step 8).
                               If False, returns (8, 1280).
        """
        attr_cpu = attribution_4d.squeeze(0).cpu().abs()  # (8, 23, 1280)
        if target_window_only:
            # Target window is index 7 (the 8th step)
            temp_curve = attr_cpu[7].sum(dim=0).numpy()  # (1280,)
        else:
            temp_curve = attr_cpu.sum(dim=1).numpy()  # (8, 1280)
        return temp_curve

    def aggregate_gru_step_importance(
        self,
        attribution_4d: torch.Tensor
    ) -> pd.DataFrame:
        """
        Aggregates absolute attribution across all channels and samples for each of the 8 sequence steps.
        
        Returns:
            DataFrame with columns: [sequence_step, step_index, temporal_offset_sec, attribution_score, normalized_importance]
        """
        attr_cpu = attribution_4d.squeeze(0).cpu().abs()  # (8, 23, 1280)
        step_scores = attr_cpu.sum(dim=(1, 2)).numpy()  # (8,)
        
        total_score = step_scores.sum() + 1e-12
        normalized_scores = step_scores / total_score
        
        # Stride = 2.5s. Step 8 is target window [0s, 5.0s].
        # Step 7 is [-2.5s, 2.5s], Step 1 is [-17.5s, -12.5s] relative to target start.
        offsets = [round(- (7 - i) * 2.5, 1) for i in range(8)]
        
        df = pd.DataFrame({
            "sequence_step": [f"Step {i+1}" for i in range(8)],
            "step_index": list(range(1, 9)),
            "temporal_offset_sec": offsets,
            "attribution_score": step_scores,
            "normalized_importance": normalized_scores
        })
        return df
