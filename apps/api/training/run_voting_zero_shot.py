#!/usr/bin/env python3
"""
run_voting_zero_shot.py
───────────────────────
Evaluates the Voting Baseline (a) in the Cross-Dataset Zero-Shot setting (Bonn → CHB-MIT):
1. Trains a single-channel LightGBM classifier on the Bonn dataset (57 features).
2. Runs the model independently per channel on CHB-MIT (23 channels) and averages predictions across channels.
3. Computes both POOLED and PER-PATIENT MACRO-AVERAGED metrics (AUROC, AUPRC, Sens@1FP/h, Sens@0.25FP/h).

Output:
  - results/zero_shot/evaluation_report_voting_zeroshot.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from training.data import MultiChannelEEGDataset, FeatureNormalizer
from training.evaluate import calculate_sensitivity_at_fp_rate


def train_bonn_lgbm(bonn_csv_path: str):
    """Train a single-channel LightGBM classifier on Bonn dataset."""
    print("=" * 70)
    print("Training Single-Channel LightGBM on Bonn Dataset")
    print("=" * 70)

    df = pd.read_csv(bonn_csv_path)
    if "target" in df.columns:
        y = df["target"].values
    elif "label" in df.columns:
        y = (df["label"] == 4).astype(int).values
    else:
        raise ValueError("No label or target column in Bonn CSV")

    meta_cols = {"label", "target", "epoch_id", "set", "patient_id"}
    feature_names = [c for c in df.columns if c not in meta_cols]

    X = df[feature_names].values.astype(np.float32)

    # Normalize Bonn features
    normalizer = FeatureNormalizer()
    X_norm = normalizer.fit_transform(X)

    # Train LightGBM model
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(X_norm, y)
    print(f"✓ Trained LightGBM on Bonn ({X.shape[0]} samples, {X.shape[1]} features)")
    return model, normalizer, feature_names


def evaluate_voting_zero_shot(bonn_csv: str, chbmit_parquet: str, output_dir: Path):
    """Evaluate Bonn-trained LightGBM model zero-shot on CHB-MIT using Voting across 23 channels."""
    model, normalizer, feature_names = train_bonn_lgbm(bonn_csv)

    print("\n" + "=" * 70)
    print("Evaluating Voting Baseline Zero-Shot (Bonn → CHB-MIT)")
    print("=" * 70)

    # Load CHB-MIT dataset using exact Bonn 57 feature names
    dataset = MultiChannelEEGDataset(chbmit_parquet, augment=False, normalize=True, selected_features=feature_names)

    all_preds = []
    all_labels = []
    patient_preds = {}
    patient_labels = {}

    for idx in range(len(dataset)):
        # Shape [C, 57]
        x = dataset.data_array[idx]
        y = dataset.labels[idx]
        pid = dataset.patient_ids[idx]

        # Predict probability per channel independently, then average
        if hasattr(model, "booster_"):
            channel_probs = model.booster_.predict(x)
        else:
            channel_probs = model.predict_proba(x)[:, 1]

        avg_prob = float(np.mean(channel_probs))

        all_preds.append(avg_prob)
        all_labels.append(y)

        patient_preds.setdefault(pid, []).append(avg_prob)
        patient_labels.setdefault(pid, []).append(y)

    y_true_all = np.array(all_labels)
    y_score_all = np.array(all_preds)

    # 1. Pooled Overall Metrics (across all windows concatenated)
    pooled_auroc = float(roc_auc_score(y_true_all, y_score_all))
    pooled_auprc = float(average_precision_score(y_true_all, y_score_all))
    pooled_sens1 = float(calculate_sensitivity_at_fp_rate(y_true_all, y_score_all, target_fp_per_hour=1.0))
    pooled_sens025 = float(calculate_sensitivity_at_fp_rate(y_true_all, y_score_all, target_fp_per_hour=0.25))

    # 2. Per-Patient Breakdown & Macro-Averaged Metrics
    per_patient_metrics = []
    for pid in sorted(patient_preds.keys()):
        p_y = np.array(patient_labels[pid])
        p_scores = np.array(patient_preds[pid])

        if len(np.unique(p_y)) < 2:
            auroc, auprc = 0.5, 0.0
        else:
            auroc = float(roc_auc_score(p_y, p_scores))
            auprc = float(average_precision_score(p_y, p_scores))

        sens1 = float(calculate_sensitivity_at_fp_rate(p_y, p_scores, target_fp_per_hour=1.0))
        sens025 = float(calculate_sensitivity_at_fp_rate(p_y, p_scores, target_fp_per_hour=0.25))

        per_patient_metrics.append({
            "patient_id": pid,
            "windows": len(p_y),
            "seizures": int(sum(p_y)),
            "auroc": auroc,
            "auprc": auprc,
            "sens_1fp_hr": sens1,
            "sens_025fp_hr": sens025,
        })

    macro_auroc = float(np.mean([m["auroc"] for m in per_patient_metrics]))
    macro_auprc = float(np.mean([m["auprc"] for m in per_patient_metrics]))
    macro_sens1 = float(np.mean([m["sens_1fp_hr"] for m in per_patient_metrics]))
    macro_sens025 = float(np.mean([m["sens_025fp_hr"] for m in per_patient_metrics]))

    print(f"\n--- POOLED METRICS ---")
    print(f"Pooled AUROC:       {pooled_auroc:.4f}")
    print(f"Pooled AUPRC:       {pooled_auprc:.4f}")
    print(f"Pooled Sens@1.0FP:  {pooled_sens1:.4f}")
    print(f"Pooled Sens@0.25FP: {pooled_sens025:.4f}")

    print(f"\n--- MACRO-AVERAGED METRICS (Across Patients) ---")
    print(f"Macro AUROC:       {macro_auroc:.4f}")
    print(f"Macro AUPRC:       {macro_auprc:.4f}")
    print(f"Macro Sens@1.0FP:  {macro_sens1:.4f}")
    print(f"Macro Sens@0.25FP: {macro_sens025:.4f}")

    report = {
        "setting": "Cross-Dataset Zero-Shot (Bonn -> CHB-MIT)",
        "model": "Voting Baseline (Single-Channel LightGBM Ensemble)",
        "pooled_metrics": {
            "auroc": pooled_auroc,
            "auprc": pooled_auprc,
            "sens_1fp_hr": pooled_sens1,
            "sens_025fp_hr": pooled_sens025,
        },
        "macro_metrics": {
            "auroc": macro_auroc,
            "auprc": macro_auprc,
            "sens_1fp_hr": macro_sens1,
            "sens_025fp_hr": macro_sens025,
        },
        "per_patient": per_patient_metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "evaluation_report_voting_zeroshot.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n✓ Saved Zero-Shot Voting Report: {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Voting Baseline Zero-Shot (Bonn -> CHB-MIT)")
    parser.add_argument("--bonn-data", type=str, default="data/bonn_features.csv")
    parser.add_argument("--chbmit-data", type=str, default="data/chbmit_multichannel.parquet")
    parser.add_argument("--output-dir", type=str, default="results/zero_shot")
    args = parser.parse_args()

    evaluate_voting_zero_shot(args.bonn_data, args.chbmit_data, Path(args.output_dir))


if __name__ == "__main__":
    main()
