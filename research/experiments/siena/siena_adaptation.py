"""
siena_adaptation.py
────────────────────
NeuroAegis Research — Phase 6: Cross-Dataset Generalization (Siena)
Domain Adaptation & Calibration Module (Experiment: PHASE6_SIENA_ADAPTATION).
Strict Research Isolation Rule:
  - Calibration / fitting is performed ONLY on the Siena Development Cohort (PN00-PN05).
  - The final Siena Evaluation Cohort (PN06-PN17) remains completely untouched during tuning.
  - No model weights are retrained (zero backprop into CNN/GNN/GRU weights).
  - Adaptation consists of:
      1. Platt Scaling / Temperature Calibration (T*)
      2. Optimal Clinical Decision Threshold Selection (tau*)
      3. Recording-level Variance Harmonization
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score
from typing import Dict, Tuple, List, Optional


class SienaDomainAdapter:
    def __init__(self, dev_cohort_names: Optional[List[str]] = None):
        self.dev_cohort_names = dev_cohort_names or ["PN00", "PN01", "PN03", "PN05"]
        self.temperature = 1.0
        self.calibrated_threshold = 0.50
        self.is_fitted = False
        
    def fit_calibration(self, dev_logits: np.ndarray, dev_labels: np.ndarray) -> Dict:
        """
        Fits temperature scaling T* and finds optimal decision threshold tau* on dev cohort.
        T* minimizes negative log likelihood (cross-entropy) with temperature scaling:
            p_i = sigma(z_i / T)
        tau* maximizes F1 score on the development cohort.
        """
        def nll_obj(t_val):
            t = t_val[0]
            scaled = dev_logits / max(t, 1e-4)
            # Binary cross entropy with logits
            # log(1 + exp(-scaled)) for y=1, log(1 + exp(scaled)) for y=0
            p = 1.0 / (1.0 + np.exp(-scaled))
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            loss = -np.mean(dev_labels * np.log(p) + (1.0 - dev_labels) * np.log(1.0 - p))
            return loss

        res = minimize(nll_obj, [1.0], bounds=[(0.1, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        
        # Compute calibrated probabilities on dev cohort
        cal_probs = 1.0 / (1.0 + np.exp(-dev_logits / self.temperature))
        
        # Grid search optimal tau in [0.10, 0.90] to maximize F1
        best_tau = 0.50
        best_f1 = -1.0
        for tau in np.linspace(0.10, 0.90, 81):
            pred = (cal_probs >= tau).astype(int)
            f1 = f1_score(dev_labels, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau
                
        self.calibrated_threshold = float(best_tau)
        self.is_fitted = True
        
        return {
            "optimal_temperature": self.temperature,
            "optimal_threshold": self.calibrated_threshold,
            "dev_f1_zero_shot": float(f1_score(dev_labels, (1.0 / (1.0 + np.exp(-dev_logits)) >= 0.50).astype(int), zero_division=0)),
            "dev_f1_calibrated": float(best_f1),
            "dev_windows": len(dev_labels),
            "dev_positives": int(np.sum(dev_labels))
        }

    def predict_adapted(self, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies learned temperature scaling and calibrated threshold.
        Returns: (calibrated_probs, binary_predictions)
        """
        if not self.is_fitted:
            raise RuntimeError("Adapter must be fitted on dev cohort first!")
        probs = 1.0 / (1.0 + np.exp(-logits / self.temperature))
        preds = (probs >= self.calibrated_threshold).astype(int)
        return probs, preds
