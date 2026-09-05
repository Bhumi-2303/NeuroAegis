#!/usr/bin/env python3
"""
run_calibrated_benchmarks.py
────────────────────────────
Evaluates Zero-Shot Transfer (Bonn -> CHB-MIT) for BOTH Voting Baseline (a)
and Attention Pooling (d) (at lambda=0.01 and lambda=0.001).

Computes BOTH Raw (Uncalibrated) and Rank-Normalized (Calibrated) metrics:
- Pooled AUROC, Pooled AUPRC, Pooled Sens@1.0 FP/h, Pooled Sens@0.25 FP/h
- Macro AUROC, Macro AUPRC, Macro Sens@1.0 FP/h, Macro Sens@0.25 FP/h
"""

import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIBOK"] = "TRUE"

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

import torch
from torch.utils.data import DataLoader
from training.data import MultiChannelEEGDataset, collate_fn, FeatureNormalizer
from training.evaluate import calculate_sensitivity_at_fp_rate
from training.model import AttentionSeizureDetector


def compute_metrics(y_true, y_score):
    """Compute AUROC, AUPRC, Sens@1.0 FP/h, Sens@0.25 FP/h."""
    if len(np.unique(y_true)) < 2:
        return {"auroc": 0.5, "auprc": 0.0, "sens_1fp_hr": 0.0, "sens_025fp_hr": 0.0}
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    sens1 = float(calculate_sensitivity_at_fp_rate(y_true, y_score, target_fp_per_hour=1.0))
    sens025 = float(calculate_sensitivity_at_fp_rate(y_true, y_score, target_fp_per_hour=0.25))
    return {"auroc": auroc, "auprc": auprc, "sens_1fp_hr": sens1, "sens_025fp_hr": sens025}


def eval_voting_zero_shot(bonn_csv: str, chbmit_parquet: str):
    """Evaluate Voting Baseline (a) Zero-Shot with vectorized batched inference."""
    df_bonn = pd.read_csv(bonn_csv)
    y_bonn = df_bonn["target"].values if "target" in df_bonn.columns else (df_bonn["label"] == 4).astype(int).values
    meta_cols = {"label", "target", "epoch_id", "set", "patient_id"}
    feature_names = [c for c in df_bonn.columns if c not in meta_cols]

    X_bonn = df_bonn[feature_names].values.astype(np.float32)
    norm = FeatureNormalizer()
    X_bonn_norm = norm.fit_transform(X_bonn)

    lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, num_leaves=31, random_state=42, n_jobs=1, verbose=-1)
    lgb_model.fit(X_bonn_norm, y_bonn)

    ds = MultiChannelEEGDataset(chbmit_parquet, augment=False, normalize=True, selected_features=feature_names)

    # Batched inference over all samples [N_windows, N_channels, 57] -> [N_windows * N_channels, 57]
    N_windows, N_channels, N_feats = ds.data_array.shape
    flat_data = ds.data_array.reshape(-1, N_feats)

    if hasattr(lgb_model, "booster_"):
        flat_probs = lgb_model.booster_.predict(flat_data)
    else:
        flat_probs = lgb_model.predict_proba(flat_data)[:, 1]

    probs_matrix = flat_probs.reshape(N_windows, N_channels)
    window_preds = probs_matrix.mean(axis=1)

    patient_scores = {}
    patient_y = {}

    for idx in range(N_windows):
        pid = ds.patient_ids[idx]
        y = ds.labels[idx]
        score = window_preds[idx]
        patient_scores.setdefault(pid, []).append(float(score))
        patient_y.setdefault(pid, []).append(int(y))

    return format_results("Voting Baseline (Single-Channel LightGBM Ensemble)", patient_scores, patient_y)


def eval_attention_zero_shot(model_pt_path: str, chbmit_parquet: str, meta_feats_path: str, name: str):
    """Evaluate Attention Pooling model zero-shot with and without rank-calibration."""
    with open(meta_feats_path) as f:
        raw_feats = json.load(f)
    clean_feats = [f.replace("Ch0_", "") for f in raw_feats]

    ds = MultiChannelEEGDataset(chbmit_parquet, augment=False, normalize=True, selected_features=clean_feats)
    model = AttentionSeizureDetector(input_dim=57, embed_dim=32)
    model.load_state_dict(torch.load(model_pt_path))
    model.eval()

    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    patient_scores = {}
    patient_y = {}

    with torch.no_grad():
        for batch in loader:
            preds, _ = model(batch["features"], batch["mask"])
            preds_np = preds.numpy().flatten()
            labels = batch["labels"].numpy().flatten()
            pids = batch["patient_ids"]
            for p, s, l in zip(pids, preds_np, labels):
                patient_scores.setdefault(p, []).append(float(s))
                patient_y.setdefault(p, []).append(int(l))

    return format_results(name, patient_scores, patient_y)


def format_results(model_name: str, patient_scores: dict, patient_y: dict):
    """Compute raw and rank-normalized pooled + macro metrics."""
    raw_pooled_y, raw_pooled_s = [], []
    rank_pooled_y, rank_pooled_s = [], []
    per_patient_raw = []

    for pid in sorted(patient_scores.keys()):
        y = np.array(patient_y[pid])
        s_raw = np.array(patient_scores[pid])
        s_rank = rankdata(s_raw) / len(s_raw)

        raw_pooled_y.extend(y)
        raw_pooled_s.extend(s_raw)
        rank_pooled_y.extend(y)
        rank_pooled_s.extend(s_rank)

        m_patient = compute_metrics(y, s_raw)
        m_patient["patient_id"] = pid
        m_patient["windows"] = len(y)
        m_patient["seizures"] = int(sum(y))
        per_patient_raw.append(m_patient)

    raw_pooled_metrics = compute_metrics(np.array(raw_pooled_y), np.array(raw_pooled_s))
    rank_pooled_metrics = compute_metrics(np.array(rank_pooled_y), np.array(rank_pooled_s))

    macro_metrics = {
        "auroc": float(np.mean([m["auroc"] for m in per_patient_raw])),
        "auprc": float(np.mean([m["auprc"] for m in per_patient_raw])),
        "sens_1fp_hr": float(np.mean([m["sens_1fp_hr"] for m in per_patient_raw])),
        "sens_025fp_hr": float(np.mean([m["sens_025fp_hr"] for m in per_patient_raw])),
    }

    return {
        "model": model_name,
        "raw_pooled": raw_pooled_metrics,
        "rank_calibrated_pooled": rank_pooled_metrics,
        "macro": macro_metrics,
        "per_patient": per_patient_raw,
    }


def main():
    bonn_csv = "data/bonn_features.csv"
    chbmit_parquet = "data/chbmit_multichannel.parquet"
    meta_path = "apps/api/models/chbmit/selected_features_patient_wise.json"

    print("=" * 80)
    print("RUNNING ZERO-SHOT BENCHMARK WITH RANK-CALIBRATION")
    print("=" * 80)

    r_voting = eval_voting_zero_shot(bonn_csv, chbmit_parquet)
    r_attn_001 = eval_attention_zero_shot("results/zero_shot/model_bonn.pt", chbmit_parquet, meta_path, "Attention Pooling (lambda=0.01)")
    r_attn_0001 = eval_attention_zero_shot("results/zero_shot_lambda_0001/model_bonn.pt", chbmit_parquet, meta_path, "Attention Pooling (lambda=0.001)")

    # Print Comparison Table
    print("\n" + "─" * 95)
    print(f"{'Model Architecture':<35} | {'Setting':<16} | {'AUROC':<7} | {'AUPRC':<7} | {'Sens@1FP':<8} | {'Sens@0.25FP':<10}")
    print("─" * 95)

    for r in [r_voting, r_attn_001, r_attn_0001]:
        m_name = r["model"]
        # Raw Pooled
        rp = r["raw_pooled"]
        print(f"{m_name:<35} | {'Raw Pooled':<16} | {rp['auroc']:.4f}  | {rp['auprc']:.4f}  | {rp['sens_1fp_hr']*100:6.2f}%  | {rp['sens_025fp_hr']*100:8.2f}%")
        # Rank Pooled
        rk = r["rank_calibrated_pooled"]
        print(f"{m_name:<35} | {'Rank Calibrated':<16} | {rk['auroc']:.4f}  | {rk['auprc']:.4f}  | {rk['sens_1fp_hr']*100:6.2f}%  | {rk['sens_025fp_hr']*100:8.2f}%")
        # Macro
        ma = r["macro"]
        print(f"{m_name:<35} | {'Macro Average':<16} | {ma['auroc']:.4f}  | {ma['auprc']:.4f}  | {ma['sens_1fp_hr']*100:6.2f}%  | {ma['sens_025fp_hr']*100:8.2f}%")
        print("─" * 95)

    out_json = Path("results/zero_shot/calibrated_zero_shot_comparison.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"voting": r_voting, "attention_lambda_001": r_attn_001, "attention_lambda_0001": r_attn_0001}, f, indent=4)
    print(f"\n✓ Saved calibrated zero-shot report to {out_json}")


if __name__ == "__main__":
    main()
