"""
NeuroAegis Phase 5: Faithfulness & Sanity Check Engine
Implements:
1. Insertion / Deletion tests (AUDC, AUIC)
2. Model parameter randomization sanity check (Adebayo et al.)
3. Input perturbation tests (high vs random vs low attribution masks)
"""

import os
import sys
import copy
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn as nn

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from research.phase_5.xai.xai_model import XAIModelWrapper
from research.phase_5.xai.attribution_engine import AttributionEngine


class FaithfulnessEngine:
    """
    Evaluates faithfulness, sensitivity, and sanity of XAI attributions.
    """
    def __init__(
        self,
        model: XAIModelWrapper,
        attribution_engine: AttributionEngine
    ):
        self.model = model
        self.engine = attribution_engine
        self.device = model.device

    def run_insertion_deletion_test(
        self,
        x_seq: torch.Tensor,
        attribution_4d: torch.Tensor,
        fractions: List[float] = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]
    ) -> Dict[str, Any]:
        """
        Computes Deletion and Insertion curves across given feature retention/masking fractions.
        
        Deletion: Starts from full input x, progressively zeroes out top-attributed pixels.
        Insertion: Starts from zero baseline, progressively re-inserts top-attributed pixels.
        Compares Top-Attributed against Random and Bottom-Attributed controls.
        """
        self.model.eval()
        x_seq = x_seq.to(self.device).float()
        
        # Flatten attribution across (8, 23, 1280) = 235,520 features
        attr_flat = attribution_4d.squeeze(0).abs().cpu().numpy().flatten()
        x_flat = x_seq.squeeze(0).cpu().numpy().flatten()
        total_feats = len(attr_flat)
        
        # Sort indices by attribution magnitude
        sorted_indices_top = np.argsort(-attr_flat)  # descending (highest first)
        sorted_indices_bot = np.argsort(attr_flat)   # ascending (lowest first)
        
        np.random.seed(42)
        random_indices = np.random.permutation(total_feats)
        
        deletion_curves = {"top": [], "bottom": [], "random": []}
        insertion_curves = {"top": [], "bottom": [], "random": []}
        
        with torch.no_grad():
            orig_logit = self.model.forward_differentiable(x_seq).item()
            orig_prob = torch.sigmoid(torch.tensor(orig_logit)).item()
            
            baseline_seq = torch.zeros_like(x_seq)
            base_logit = self.model.forward_differentiable(baseline_seq).item()
            base_prob = torch.sigmoid(torch.tensor(base_logit)).item()
            
            for frac in fractions:
                k = int(frac * total_feats)
                
                # --- DELETION TEST ---
                # Top: zero out top k
                del_top = x_flat.copy()
                if k > 0:
                    del_top[sorted_indices_top[:k]] = 0.0
                p_del_top = torch.sigmoid(self.model.forward_differentiable(
                    torch.from_numpy(del_top.reshape(1, 8, 23, 1280)).to(self.device)
                )).item()
                deletion_curves["top"].append(p_del_top)
                
                # Bottom: zero out bottom k
                del_bot = x_flat.copy()
                if k > 0:
                    del_bot[sorted_indices_bot[:k]] = 0.0
                p_del_bot = torch.sigmoid(self.model.forward_differentiable(
                    torch.from_numpy(del_bot.reshape(1, 8, 23, 1280)).to(self.device)
                )).item()
                deletion_curves["bottom"].append(p_del_bot)
                
                # Random: zero out random k
                del_rnd = x_flat.copy()
                if k > 0:
                    del_rnd[random_indices[:k]] = 0.0
                p_del_rnd = torch.sigmoid(self.model.forward_differentiable(
                    torch.from_numpy(del_rnd.reshape(1, 8, 23, 1280)).to(self.device)
                )).item()
                deletion_curves["random"].append(p_del_rnd)
                
                # --- INSERTION TEST ---
                # Top: insert top k into baseline
                ins_top = np.zeros_like(x_flat)
                if k > 0:
                    ins_top[sorted_indices_top[:k]] = x_flat[sorted_indices_top[:k]]
                p_ins_top = torch.sigmoid(self.model.forward_differentiable(
                    torch.from_numpy(ins_top.reshape(1, 8, 23, 1280)).to(self.device)
                )).item()
                insertion_curves["top"].append(p_ins_top)
                
                # Bottom: insert bottom k into baseline
                ins_bot = np.zeros_like(x_flat)
                if k > 0:
                    ins_bot[sorted_indices_bot[:k]] = x_flat[sorted_indices_bot[:k]]
                p_ins_bot = torch.sigmoid(self.model.forward_differentiable(
                    torch.from_numpy(ins_bot.reshape(1, 8, 23, 1280)).to(self.device)
                )).item()
                insertion_curves["bottom"].append(p_ins_bot)
                
                # Random: insert random k into baseline
                ins_rnd = np.zeros_like(x_flat)
                if k > 0:
                    ins_rnd[random_indices[:k]] = x_flat[random_indices[:k]]
                p_ins_rnd = torch.sigmoid(self.model.forward_differentiable(
                    torch.from_numpy(ins_rnd.reshape(1, 8, 23, 1280)).to(self.device)
                )).item()
                insertion_curves["random"].append(p_ins_rnd)
                
        # Compute Area Under Curves using trapezoidal integration
        audc_top = float(np.trapz(deletion_curves["top"], fractions))
        audc_bot = float(np.trapz(deletion_curves["bottom"], fractions))
        audc_rnd = float(np.trapz(deletion_curves["random"], fractions))
        
        auic_top = float(np.trapz(insertion_curves["top"], fractions))
        auic_bot = float(np.trapz(insertion_curves["bottom"], fractions))
        auic_rnd = float(np.trapz(insertion_curves["random"], fractions))
        
        return {
            "fractions": fractions,
            "deletion_top": deletion_curves["top"],
            "deletion_bottom": deletion_curves["bottom"],
            "deletion_random": deletion_curves["random"],
            "insertion_top": insertion_curves["top"],
            "insertion_bottom": insertion_curves["bottom"],
            "insertion_random": insertion_curves["random"],
            "audc_top": audc_top,
            "audc_bottom": audc_bot,
            "audc_random": audc_rnd,
            "auic_top": auic_top,
            "auic_bottom": auic_bot,
            "auic_random": auic_rnd,
            "faithful_deletion": audc_top < audc_rnd,  # Deletion of top features drops prob faster
            "faithful_insertion": auic_top > auic_rnd   # Insertion of top features raises prob faster
        }

    def run_cascading_parameter_randomization(
        self,
        x_seq: torch.Tensor,
        original_attr: torch.Tensor
    ) -> pd.DataFrame:
        """
        Adebayo et al. (2018) Cascading Model Parameter Randomization Sanity Check.
        Creates temporary randomized copies of model layers and tests whether
        attributions are sensitive to the learned weights.
        """
        # Baseline channel attribution in canonical order (0 to 22)
        orig_ch = original_attr.squeeze(0).abs().sum(dim=(0, 2)).cpu().numpy()
        
        stages = [
            ("Original (Trained)", []),
            ("Randomize Classifier Head", ["classifier"]),
            ("Randomize Classifier + GRU", ["classifier", "gru"]),
            ("Randomize Classifier + GRU + GNN", ["classifier", "gru", "gcn"]),
            ("Randomize Entire Network (Full Cascading)", ["classifier", "gru", "gcn", "cnn"])
        ]
        
        results = []
        
        for stage_name, modules_to_randomize in stages:
            if not modules_to_randomize:
                # Stage 0: exact match
                results.append({
                    "stage": stage_name,
                    "modules_randomized": "None",
                    "spearman_rho": 1.0,
                    "p_value": 0.0,
                    "top1_match": True,
                    "top3_overlap": 3
                })
                continue
                
            # Create a clean temporary copy of the wrapper
            temp_model = copy.deepcopy(self.model)
            temp_model.eval()
            
            torch.manual_seed(42)
            if "classifier" in modules_to_randomize:
                for m in temp_model.base_model.classifier.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                            
            if "gru" in modules_to_randomize:
                for name, param in temp_model.base_model.gru.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)
                        
            if "gcn" in modules_to_randomize:
                for m in [temp_model.base_model.backbone.gcn1, temp_model.base_model.backbone.gcn2]:
                    nn.init.xavier_uniform_(m.linear.weight)
                    if m.linear.bias is not None:
                        nn.init.zeros_(m.linear.bias)
                        
            if "cnn" in modules_to_randomize:
                for m in temp_model.base_model.backbone.temporal_backbone.modules():
                    if isinstance(m, nn.Conv1d):
                        nn.init.xavier_uniform_(m.weight)
                        
            # Recompute attribution on randomized model
            temp_engine = AttributionEngine(temp_model)
            rand_res = temp_engine.compute_integrated_gradients(x_seq, steps=10)
            rand_ch = rand_res["attribution"].squeeze(0).abs().sum(dim=(0, 2)).cpu().numpy()
            
            # Rank correlation with original in canonical order
            rho, p_val = spearmanr(orig_ch, rand_ch)
            if np.isnan(rho):
                rho = 0.0
                
            top1_orig = np.argmax(orig_ch)
            top1_rand = np.argmax(rand_ch)
            top3_orig = set(np.argsort(-orig_ch)[:3])
            top3_rand = set(np.argsort(-rand_ch)[:3])
            
            results.append({
                "stage": stage_name,
                "modules_randomized": "+".join(modules_to_randomize),
                "spearman_rho": float(round(rho, 4)),
                "p_value": float(round(p_val, 6)) if not np.isnan(p_val) else 1.0,
                "top1_match": bool(top1_orig == top1_rand),
                "top3_overlap": int(len(top3_orig & top3_rand))
            })
            
            del temp_model
            
        return pd.DataFrame(results)

    def run_input_perturbation_test(
        self,
        x_seq: torch.Tensor,
        attribution_4d: torch.Tensor,
        perturbation_ratio: float = 0.20
    ) -> Dict[str, Any]:
        """
        Perturbs top-k vs random-k vs bottom-k attributed regions with localized Gaussian noise
        and records the resulting probability delta.
        """
        self.model.eval()
        x_seq = x_seq.to(self.device).float()
        
        attr_flat = attribution_4d.squeeze(0).abs().cpu().numpy().flatten()
        x_flat = x_seq.squeeze(0).cpu().numpy().flatten()
        total_feats = len(attr_flat)
        k = int(perturbation_ratio * total_feats)
        
        sorted_top = np.argsort(-attr_flat)[:k]
        sorted_bot = np.argsort(attr_flat)[:k]
        np.random.seed(42)
        rnd_idx = np.random.permutation(total_feats)[:k]
        
        # Standard deviation of the input signal
        noise_std = float(np.std(x_flat))
        
        with torch.no_grad():
            orig_prob = torch.sigmoid(self.model.forward_differentiable(x_seq)).item()
            
            # Perturb Top
            x_top = x_flat.copy()
            x_top[sorted_top] += np.random.normal(0.0, noise_std, size=k).astype(np.float32)
            p_top = torch.sigmoid(self.model.forward_differentiable(
                torch.from_numpy(x_top.reshape(1, 8, 23, 1280)).to(self.device)
            )).item()
            
            # Perturb Bottom
            x_bot = x_flat.copy()
            x_bot[sorted_bot] += np.random.normal(0.0, noise_std, size=k).astype(np.float32)
            p_bot = torch.sigmoid(self.model.forward_differentiable(
                torch.from_numpy(x_bot.reshape(1, 8, 23, 1280)).to(self.device)
            )).item()
            
            # Perturb Random
            x_rnd = x_flat.copy()
            x_rnd[rnd_idx] += np.random.normal(0.0, noise_std, size=k).astype(np.float32)
            p_rnd = torch.sigmoid(self.model.forward_differentiable(
                torch.from_numpy(x_rnd.reshape(1, 8, 23, 1280)).to(self.device)
            )).item()
            
        return {
            "perturbation_ratio": perturbation_ratio,
            "original_probability": orig_prob,
            "perturbed_top_prob": p_top,
            "perturbed_bottom_prob": p_bot,
            "perturbed_random_prob": p_rnd,
            "delta_top": abs(orig_prob - p_top),
            "delta_bottom": abs(orig_prob - p_bot),
            "delta_random": abs(orig_prob - p_rnd),
            "is_sensitive": abs(orig_prob - p_top) > abs(orig_prob - p_bot)
        }
