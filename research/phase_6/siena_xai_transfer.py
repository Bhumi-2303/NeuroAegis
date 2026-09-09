"""
siena_xai_transfer.py
──────────────────────
NeuroAegis Research — Phase 6: Cross-Dataset Generalization (Siena)
Evaluates explainability transfer from CHB-MIT to Siena (Experiment: PHASE6_SIENA_XAI_TRANSFER).
Uses the frozen Phase 5 Integrated Gradients engine to compute:
  1. Channel importance ranking across 23 canonical leads
  2. Temporal intra-window attribution curves (1280 samples)
  3. GRU causal sequence-step attributions (8 steps, 22.5s span)
Evaluates on deterministic benchmark cases (True Positive, False Positive, False Negative).
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from research.phase_5.xai.xai_model import XAIModelWrapper
from research.phase_5.xai.attribution_engine import AttributionEngine

RESULTS_DIR = os.path.join(BASE_DIR, "research/phase_6/results")
CHANNEL_ORDER_PATH = os.path.join(BASE_DIR, "research/data/config/chbmit_channel_order.json")


class SienaXAIExplainer:
    def __init__(self, device: torch.device = None):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        self.xai_model = XAIModelWrapper(device=self.device)
        self.xai_model.eval()
        self.engine = AttributionEngine(self.xai_model)
        
        with open(CHANNEL_ORDER_PATH, "r") as f:
            self.channel_order = json.load(f)

    def explain_window(
        self,
        sequence_tensor: np.ndarray,
        steps: int = 25
    ) -> Dict:
        """
        Computes Integrated Gradients attribution for a single (8, 23, 1280) causal sequence.
        
        Returns:
            Dict containing channel attributions, temporal curves, sequence step importance,
            and completeness metrics.
        """
        # Shape to (1, 8, 23, 1280)
        if len(sequence_tensor.shape) == 3:
            sequence_tensor = np.expand_dims(sequence_tensor, axis=0)
            
        x_tensor = torch.from_numpy(sequence_tensor).to(self.device)
        
        # 1. Integrated Gradients
        ig_res = self.engine.compute_integrated_gradients(x_tensor, steps=steps)
        ig_attr = ig_res["attribution"]
        delta = ig_res["completeness_delta"]
        
        # 2. Channel importance aggregation
        df_ch = self.engine.aggregate_channel_attribution(ig_attr)
        ch_importance = {row["channel_name"]: float(row["normalized_attribution"]) for _, row in df_ch.iterrows()}
        ranked_channels = list(df_ch["channel_name"])
        
        # 3. Temporal attribution curve across current window (Step 8, 1280 samples)
        step8_attr = self.engine.aggregate_temporal_attribution(ig_attr, target_window_only=True)
        
        # 4. Sequence step attribution (8 steps)
        df_steps = self.engine.aggregate_gru_step_importance(ig_attr)
        step_importance_share = df_steps["normalized_importance"].values
        
        # 5. Output probability
        with torch.no_grad():
            logit = self.xai_model.forward_differentiable(x_tensor)
            prob = float(torch.sigmoid(logit).cpu().numpy().item())
            
        return {
            "prediction_prob": prob,
            "ig_completeness_delta": float(delta),
            "channel_importance": ch_importance,
            "ranked_channels": ranked_channels,
            "temporal_curve_1280": step8_attr.tolist(),
            "sequence_step_shares": [float(val) for val in step_importance_share]
        }
