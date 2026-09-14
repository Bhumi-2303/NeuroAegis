#!/usr/bin/env python3
"""
run_baselines.py
────────────────
Evaluates the baseline models on a given multi-channel dataset:
  (a) Voting Baseline: single-channel LightGBM run independently per channel & averaged
  (c) Dense/Concat Baseline: MLP trained on concatenated 23*57 = 1311 features

Output:
  - apps/api/results/voting/evaluation_report_voting.json
  - apps/api/results/concat/evaluation_report_concat.json
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIBOK"] = "TRUE"

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from training.data import MultiChannelEEGDataset, collate_fn, get_patient_splits
from training.evaluate import calculate_sensitivity_at_fp_rate


class ConcatMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_voting_baseline(parquet_path: str, model_path: str, meta_path: str, output_dir: Path):
    """Run Voting Baseline (a): Single-channel LightGBM run independently per channel."""
    print("=" * 70, flush=True)
    print("Evaluating Baseline (a): Voting Baseline (Single-Channel LightGBM Ensemble)", flush=True)
    print("=" * 70, flush=True)

    print(f"Loading LightGBM model from {model_path}...", flush=True)
    lgbm_model = joblib.load(model_path)
    if isinstance(lgbm_model, dict) and "model" in lgbm_model:
        lgbm_model = lgbm_model["model"]

    print(f"Loading selected features from {meta_path}...", flush=True)
    with open(meta_path) as f:
        raw_feats = json.load(f)
    clean_feats = [f.replace("Ch0_", "") for f in raw_feats]

    print(f"Loading dataset from {parquet_path}...", flush=True)
    dataset = MultiChannelEEGDataset(parquet_path, augment=False, normalize=True, selected_features=clean_feats)
    splits = get_patient_splits(dataset)
    print(f"Dataset loaded. Samples: {len(dataset)}, Folds: {len(splits)}", flush=True)

    metrics_per_fold = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        # Shape [N_val, C, 57]
        val_data = dataset.data_array[val_idx]
        n_val_samples, n_channels, n_features = val_data.shape
        
        # Reshape to [N_val * C, 57] for fast batched LightGBM inference
        flat_val = val_data.reshape(-1, n_features)
        
        # Fast C++ predict using booster_ to avoid OpenMP thread lock
        if hasattr(lgbm_model, 'booster_'):
            flat_probs = lgbm_model.booster_.predict(flat_val)
        else:
            flat_probs = lgbm_model.predict_proba(flat_val)[:, 1]
        
        # Reshape back to [N_val, C] and average across channels
        probs_matrix = flat_probs.reshape(n_val_samples, n_channels)
        all_scores = probs_matrix.mean(axis=1)
        y_true = dataset.labels[val_idx]

        if len(np.unique(y_true)) < 2:
            auroc, auprc, sens1, sens025 = 0.5, 0.0, 0.0, 0.0
        else:
            auroc = roc_auc_score(y_true, all_scores)
            auprc = average_precision_score(y_true, all_scores)
            sens1 = calculate_sensitivity_at_fp_rate(y_true, all_scores, target_fp_per_hour=1.0)
            sens025 = calculate_sensitivity_at_fp_rate(y_true, all_scores, target_fp_per_hour=0.25)

        pid = dataset.patient_ids[val_idx[0]]
        metrics_per_fold.append({
            "fold": fold,
            "patient_id": pid,
            "windows": len(y_true),
            "seizures": int(sum(y_true)),
            "auroc": float(auroc),
            "auprc": float(auprc),
            "sens_1fp_hr": float(sens1),
            "sens_025fp_hr": float(sens025),
        })

        print(f"Patient {pid:<6} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | Sens@1FP/h: {sens1:.4f} | Sens@0.25FP/h: {sens025:.4f}")

    mean_auroc = np.mean([m["auroc"] for m in metrics_per_fold])
    mean_auprc = np.mean([m["auprc"] for m in metrics_per_fold])
    mean_sens1 = np.mean([m["sens_1fp_hr"] for m in metrics_per_fold])
    mean_sens025 = np.mean([m["sens_025fp_hr"] for m in metrics_per_fold])

    print("-" * 70)
    print(f"MEAN   | AUROC: {mean_auroc:.4f} | AUPRC: {mean_auprc:.4f} | Sens@1FP/h: {mean_sens1:.4f} | Sens@0.25FP/h: {mean_sens025:.4f}")

    report = {
        "mode": "voting",
        "mean_metrics": {
            "auroc": float(mean_auroc),
            "auprc": float(mean_auprc),
            "sens_1fp_hr": float(mean_sens1),
            "sens_025fp_hr": float(mean_sens025),
        },
        "per_patient": metrics_per_fold,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report_voting.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"✓ Saved Voting report: {report_path}\n")
    return report


def run_concat_baseline(parquet_path: str, output_dir: Path, epochs: int = 30, lr: float = 1e-3):
    """Run Fixed 23-Channel Concat Baseline (c): MLP trained on concatenated channels."""
    print("=" * 70)
    print("Evaluating Baseline (c): Fixed Multi-Channel Dense/Concat Baseline")
    print("=" * 70)

    dataset = MultiChannelEEGDataset(parquet_path, augment=False, normalize=True)
    splits = get_patient_splits(dataset)
    concat_dim = dataset.n_channels * dataset.n_features

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    metrics_per_fold = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)

        train_loader = DataLoader(train_sub, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_sub, batch_size=32, shuffle=False, collate_fn=collate_fn)

        model = ConcatMLP(concat_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        # Compute pos weight
        train_labels = dataset.labels[train_idx]
        n_pos = sum(train_labels)
        n_neg = len(train_labels) - n_pos
        pos_weight = float(n_neg) / max(float(n_pos), 1.0)

        criterion = nn.BCELoss(reduction="none")

        best_val_auprc = -1.0
        best_scores, best_labels = None, None

        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                x = batch["features"].reshape(batch["features"].shape[0], -1).to(device)
                y = batch["labels"].to(device)

                optimizer.zero_grad()
                preds = model(x)
                weights = torch.where(y == 1.0, torch.tensor(pos_weight, device=device), torch.tensor(1.0, device=device))
                loss = (criterion(preds, y) * weights).mean()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["features"].reshape(batch["features"].shape[0], -1).to(device)
                    y = batch["labels"].numpy().flatten()
                    preds = model(x).cpu().numpy().flatten()
                    val_preds.extend(preds)
                    val_targets.extend(y)

            val_targets = np.array(val_targets)
            val_preds = np.array(val_preds)

            if len(np.unique(val_targets)) >= 2:
                auprc = average_precision_score(val_targets, val_preds)
                if auprc > best_val_auprc:
                    best_val_auprc = auprc
                    best_scores = val_preds
                    best_labels = val_targets

        if best_scores is None:
            best_scores = np.zeros(len(val_idx))
            best_labels = dataset.labels[val_idx]

        auroc = roc_auc_score(best_labels, best_scores)
        sens1 = calculate_sensitivity_at_fp_rate(best_labels, best_scores, target_fp_per_hour=1.0)
        sens025 = calculate_sensitivity_at_fp_rate(best_labels, best_scores, target_fp_per_hour=0.25)

        pid = dataset.patient_ids[val_idx[0]]
        metrics_per_fold.append({
            "fold": fold,
            "patient_id": pid,
            "windows": len(best_labels),
            "seizures": int(sum(best_labels)),
            "auroc": float(auroc),
            "auprc": float(best_val_auprc if best_val_auprc > 0 else 0.0),
            "sens_1fp_hr": float(sens1),
            "sens_025fp_hr": float(sens025),
        })

        print(f"Patient {pid:<6} | AUROC: {auroc:.4f} | AUPRC: {max(best_val_auprc, 0):.4f} | Sens@1FP/h: {sens1:.4f} | Sens@0.25FP/h: {sens025:.4f}")

    mean_auroc = np.mean([m["auroc"] for m in metrics_per_fold])
    mean_auprc = np.mean([m["auprc"] for m in metrics_per_fold])
    mean_sens1 = np.mean([m["sens_1fp_hr"] for m in metrics_per_fold])
    mean_sens025 = np.mean([m["sens_025fp_hr"] for m in metrics_per_fold])

    print("-" * 70)
    print(f"MEAN   | AUROC: {mean_auroc:.4f} | AUPRC: {mean_auprc:.4f} | Sens@1FP/h: {mean_sens1:.4f} | Sens@0.25FP/h: {mean_sens025:.4f}")

    report = {
        "mode": "concat",
        "mean_metrics": {
            "auroc": float(mean_auroc),
            "auprc": float(mean_auprc),
            "sens_1fp_hr": float(mean_sens1),
            "sens_025fp_hr": float(mean_sens025),
        },
        "per_patient": metrics_per_fold,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report_concat.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"✓ Saved Concat report: {report_path}\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run Baseline Evaluations")
    parser.add_argument("--data", type=str, default="data/chbmit_multichannel.parquet")
    parser.add_argument("--lgbm-model", type=str, default="apps/api/models/chbmit/lightgbm_patient_wise.pkl")
    parser.add_argument("--lgbm-meta", type=str, default="apps/api/models/chbmit/selected_features_patient_wise.json")
    parser.add_argument("--output-dir", type=str, default="apps/api/results")
    args = parser.parse_args()

    out_base = Path(args.output_dir)
    run_voting_baseline(args.data, args.lgbm_model, args.lgbm_meta, out_base / "voting")
    run_concat_baseline(args.data, out_base / "concat")

if __name__ == "__main__":
    main()
