#!/usr/bin/env python3
"""
train_and_eval_baselines.py
────────────────────────────
Rigorously trains, tunes, and evaluates Classical Machine Learning Baselines
(Random Forest, XGBoost, LightGBM, Linear SVM) against frozen Model C.

Rules & Protocol Constraints:
  1. DO NOT modify Model C.
  2. DO NOT modify the master evaluation protocol (neuroaegis.eval.metrics.evaluate_event_level).
  3. DO NOT use test data for feature selection, threshold selection, hyperparameter tuning, or model selection.
  4. Use CHB-MIT only with the exact same patient-independent split:
       - TRAIN (16): chb04, chb09, chb11–chb24 (10:1 balanced subsampling)
       - VALIDATION (4): chb06, chb07, chb08, chb10
       - TEST (4): chb01, chb02, chb03, chb05
  5. Test set is LOCKED until hyperparameter tuning and threshold selection are finalized on VALIDATION.
  6. Peak RSS must remain < 4.0 GB.

Outputs:
  research/experiments/baselines/
    random_forest/
    xgboost/
    lightgbm/
    svm/
    classical_ml_comparison.csv
    README.md
"""

import os
import sys
import time
import json
import yaml
import glob
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neuroaegis.eval.harness import load_eval_config
from neuroaegis.eval.metrics import evaluate_event_level, apply_false_alarm_protocol, _get_intervals
from neuroaegis.eval.schemas import EvalConfig, EventMetrics
from neuroaegis.guardrails.checks import assert_no_patient_leakage
from neuroaegis.utils.memory import flush_memory, get_peak_rss_mb, get_live_rss_mb

FEATURES_DIR = REPO_ROOT / "data" / "chbmit_features"
BASELINES_DIR = REPO_ROOT / "research" / "experiments" / "baselines"

TRAIN_PATIENTS = [
    "chb04", "chb09", "chb11", "chb12", "chb13", "chb14", "chb15",
    "chb16", "chb17", "chb18", "chb19", "chb20", "chb21", "chb22", "chb23", "chb24"
]
VAL_PATIENTS = ["chb06", "chb07", "chb08", "chb10"]
TEST_PATIENTS = ["chb01", "chb02", "chb03", "chb05"]

FEATURE_COLS = [
    "mean", "median", "std", "variance", "minimum", "maximum", "range", "rms",
    "energy", "absolute_mean", "line_length", "zero_crossings", "skewness", "kurtosis",
    "iqr", "crest_factor", "shape_factor", "impulse_factor", "clearance_factor",
    "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
    "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
    "relative_delta", "relative_theta", "relative_alpha", "relative_beta", "relative_gamma",
    "dominant_frequency", "spectral_entropy", "spectral_centroid", "total_power",
    "wavelet_entropy",
    "wavelet_energy_0", "wavelet_relative_energy_0", "wavelet_mean_0", "wavelet_std_0",
    "wavelet_energy_1", "wavelet_relative_energy_1", "wavelet_mean_1", "wavelet_std_1",
    "wavelet_energy_2", "wavelet_relative_energy_2", "wavelet_mean_2", "wavelet_std_2",
    "wavelet_energy_3", "wavelet_relative_energy_3", "wavelet_mean_3", "wavelet_std_3",
    "wavelet_energy_4", "wavelet_relative_energy_4", "wavelet_mean_4", "wavelet_std_4"
]

MODEL_C_FROZEN_METRICS = {
    "model": "Model C (CNN + GNN + GRU)",
    "status": "FROZEN REFERENCE ONLY",
    "auroc": 0.98970,
    "auprc": 0.80681,
    "sensitivity": 0.83830,
    "specificity": 0.99818,
    "precision": 0.57235,
    "f1_score": 0.68025,
    "balanced_accuracy": 0.91824,
    "event_sensitivity": 0.95455,  # 21 / 22 events
    "detection_delay_sec": 10.57,
    "false_positive_windows": 399,
    "false_alarm_episodes": 399,
    "fa_per_24h": 62.66,
    "accuracy": 0.99772
}


def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT), check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def load_training_data() -> Tuple[np.ndarray, np.ndarray]:
    p_file = FEATURES_DIR / "train_subsampled_ep0.parquet"
    if not p_file.exists():
        raise FileNotFoundError(f"Missing training feature parquet: {p_file}")
    print(f"Loading training features: {p_file.name} ...", flush=True)
    df = pd.read_parquet(p_file)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
    y = df["label_50pct_overlap"].to_numpy(dtype=np.int32, copy=False)
    del df
    flush_memory()
    return X, y


def load_cohort_data(patient_list: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dfs = []
    for pt in patient_list:
        p_file = FEATURES_DIR / f"{pt}.parquet"
        if not p_file.exists():
            raise FileNotFoundError(f"Missing feature file for {pt}: {p_file}")
        print(f"  Loading shard: {p_file.name} ...", flush=True)
        dfs.append(pd.read_parquet(p_file))

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    flush_memory()
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
    y = df["label_50pct_overlap"].to_numpy(dtype=np.int32, copy=False)
    return df, X, y


def generate_roc_pr_plots(y_true: np.ndarray, y_probs: np.ndarray, model_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=300)
    plt.close()

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    pr_auc_val = auc(rec, prec)

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="blue", lw=2, label=f"PR curve (AUPRC = {pr_auc_val:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=300)
    plt.close()


def train_and_evaluate_model(
    model_name: str,
    folder_name: str,
    create_model_fn: Any,
    hyperparam_grid: List[Dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    eval_config: EvalConfig,
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {model_name.upper()}")
    print("=" * 70, flush=True)

    exp_dir = BASELINES_DIR / folder_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_dir = exp_dir / "results"
    figures_dir = exp_dir / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # Preprocessing: Impute + Scale fit STRICTLY on train
    print("Fitting SimpleImputer and StandardScaler strictly on Training split...", flush=True)
    imputer = SimpleImputer(strategy="median", copy=False)
    scaler = StandardScaler(copy=False)

    X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train.copy()))
    X_val_proc = scaler.transform(imputer.transform(X_val.copy()))
    X_test_proc = scaler.transform(imputer.transform(X_test.copy()))

    val_hours = (len(y_val) * 2.5) / 3600.0
    test_hours = 152.82  # Authoritative continuous monitoring hours for test cohort

    # Hyperparameter Tuning on VALIDATION ONLY
    best_val_auprc = -1.0
    best_hyperparams = None
    best_model_obj = None
    best_train_time = 0.0
    best_val_probs = None

    print(f"\n--- Hyperparameter Grid Search on VALIDATION Set ({len(hyperparam_grid)} configs) ---", flush=True)
    for hp_idx, params in enumerate(hyperparam_grid):
        t0 = time.time()
        model = create_model_fn(params)
        model.fit(X_train_proc, y_train)
        t1 = time.time()
        train_time = t1 - t0

        if hasattr(model, "predict_proba"):
            val_probs = model.predict_proba(X_val_proc)[:, 1]
        else:
            val_dfunc = model.decision_function(X_val_proc)
            val_probs = 1.0 / (1.0 + np.exp(-val_dfunc))  # Sigmoid calibration

        val_auprc = float(average_precision_score(y_val, val_probs))
        val_auroc = float(auc(*roc_curve(y_val, val_probs)[:2]))
        print(f"  [Config {hp_idx+1}/{len(hyperparam_grid)}] Val AUPRC: {val_auprc:.5f} | Val AUROC: {val_auroc:.5f} | Fit Time: {train_time:.2f}s | Params: {params}", flush=True)

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_hyperparams = params
            best_model_obj = model
            best_train_time = train_time
            best_val_probs = val_probs

    # Threshold Selection on VALIDATION ONLY (maximize validation F1)
    print("\n--- Tuning Decision Threshold on VALIDATION Set ---", flush=True)
    best_threshold = 0.5
    best_val_f1 = -1.0
    best_val_sens = 0.0
    best_val_spec = 0.0

    for tau in np.arange(0.05, 0.95, 0.05):
        val_ev = evaluate_event_level(
            y_val, best_val_probs, eval_config, total_duration_hours=val_hours, threshold=float(tau)
        )
        if val_ev.f1_score > best_val_f1:
            best_val_f1 = val_ev.f1_score
            best_threshold = float(tau)
            best_val_sens = val_ev.sensitivity
            best_val_spec = val_ev.specificity

    print(f"  Optimal Decision Threshold: τ = {best_threshold:.2f} (Val F1: {best_val_f1:.4f}, Sens: {best_val_sens*100:.2f}%, Spec: {best_val_spec*100:.2f}%)", flush=True)

    # LOCKED TEST EVALUATION
    print("\n--- Running Evaluation on LOCKED TEST Cohort (152.82 Hours) ---", flush=True)
    t0_inf = time.time()
    if hasattr(best_model_obj, "predict_proba"):
        test_probs = best_model_obj.predict_proba(X_test_proc)[:, 1]
    else:
        test_dfunc = best_model_obj.decision_function(X_test_proc)
        test_probs = 1.0 / (1.0 + np.exp(-test_dfunc))
    t1_inf = time.time()
    inf_latency_ms = round(((t1_inf - t0_inf) / len(test_probs)) * 1000.0, 4)

    test_metrics = evaluate_event_level(
        y_test, test_probs, eval_config, total_duration_hours=test_hours, threshold=best_threshold
    )

    y_test_pred = (test_probs >= best_threshold).astype(int)
    tn, fp, fn, tp = (
        (y_test == 0) & (y_test_pred == 0),
        (y_test == 0) & (y_test_pred == 1),
        (y_test == 1) & (y_test_pred == 0),
        (y_test == 1) & (y_test_pred == 1)
    )
    fp_count = int(np.sum(fp))

    pred_events = apply_false_alarm_protocol(y_test_pred, eval_config)
    true_events = [i for i in _get_intervals(y_test, 5.0) if (i[1] - i[0]) >= eval_config.min_seizure_duration_sec]

    matched_preds = set()
    for true_start, true_end in true_events:
        for p_idx, (p_start, p_end) in enumerate(pred_events):
            if p_end > true_start and p_start < true_end:
                if (p_start - true_start) <= eval_config.allowed_delay_sec:
                    matched_preds.add(p_idx)
                    break

    fa_episodes_count = len(pred_events) - len(matched_preds)

    print(f"\n  AUROC:               {test_metrics.auroc:.5f}")
    print(f"  AUPRC:               {test_metrics.auprc:.5f}")
    print(f"  Sensitivity (Win):   {test_metrics.sensitivity*100:.2f}%")
    print(f"  Specificity (Win):   {test_metrics.specificity*100:.2f}%")
    print(f"  Precision (Win):     {test_metrics.precision*100:.2f}%")
    print(f"  F1 Score (Win):      {test_metrics.f1_score:.5f}")
    print(f"  Event Sensitivity:   {test_metrics.event_sensitivity*100:.2f}% ({int(round(test_metrics.event_sensitivity * len(true_events)))}/{len(true_events)})")
    print(f"  Mean Detection Delay: {test_metrics.detection_delay_sec:.2f} s")
    print(f"  False Alarms / 24h:  {test_metrics.fa_per_24h:.2f}")
    print(f"  Inference Latency:   {inf_latency_ms} ms/window")
    print(f"  Peak RSS:            {get_peak_rss_mb():.1f} MB", flush=True)

    # Save Visualizations
    generate_roc_pr_plots(y_test, test_probs, model_name, figures_dir)

    # Save Predictions Parquet
    pred_df = test_df[["window_id", "patient_id", "edf_filename", "window_start_sec", "window_end_sec", "label_50pct_overlap"]].copy()
    pred_df["y_prob"] = test_probs
    pred_df["y_pred"] = y_test_pred
    pred_df.to_parquet(results_dir / "predictions.parquet", index=False)
    del pred_df

    # Save Config YAML
    config_data = {
        "model": model_name,
        "seed": 42,
        "feature_count": len(FEATURE_COLS),
        "git_commit": get_git_commit_hash(),
        "training_time_sec": round(best_train_time, 4),
        "inference_latency_ms": inf_latency_ms,
        "selected_hyperparameters": best_hyperparams,
        "best_val_auprc": round(best_val_auprc, 5),
        "decision_threshold": best_threshold,
        "train_samples": len(y_train),
        "validation_samples": len(y_val),
        "test_samples": len(y_test),
        "train_patients": TRAIN_PATIENTS,
        "validation_patients": VAL_PATIENTS,
        "test_patients": TEST_PATIENTS,
        "eval_config": eval_config.dict()
    }
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    # Save Results JSON & CSV
    results_json = {
        "model": model_name,
        "auroc": round(float(test_metrics.auroc), 5),
        "auprc": round(float(test_metrics.auprc), 5),
        "sensitivity": round(float(test_metrics.sensitivity), 5),
        "specificity": round(float(test_metrics.specificity), 5),
        "precision": round(float(test_metrics.precision), 5),
        "f1_score": round(float(test_metrics.f1_score), 5),
        "balanced_accuracy": round(float((test_metrics.sensitivity + test_metrics.specificity) / 2.0), 5),
        "event_sensitivity": round(float(test_metrics.event_sensitivity), 5),
        "detection_delay_sec": round(float(test_metrics.detection_delay_sec), 2),
        "false_positive_windows": fp_count,
        "false_alarm_episodes": fa_episodes_count,
        "fa_per_24h": round(float(test_metrics.fa_per_24h), 2),
        "accuracy": round(float(test_metrics.accuracy), 5),
        "latency_ms": inf_latency_ms
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(results_json, f, indent=4)

    pd.DataFrame([results_json]).to_csv(results_dir / "metrics.csv", index=False)

    # Save Markdown README
    readme_content = f"""# {model_name} Baseline Experiment

## Overview
Evaluates **{model_name}** trained on 57 multi-domain EEG features (Time, Frequency, Wavelet) using the exact patient-independent CHB-MIT split.

## Dataset & Split
- **Train (16 patients, 10:1 balanced subsampling)**: {', '.join(TRAIN_PATIENTS)} ({len(y_train):,} windows)
- **Validation (4 patients)**: {', '.join(VAL_PATIENTS)} ({len(y_val):,} windows)
- **Test (4 held-out patients, locked)**: {', '.join(TEST_PATIENTS)} ({len(y_test):,} windows, 152.82 hours)

## Features & Preprocessing
- **Feature Count**: 57 multi-domain features per window (channel-averaged).
- **Imputation**: Median imputation fit strictly on Training set.
- **Normalization**: Z-score StandardScaler fit strictly on Training set.

## Hyperparameters & Tuning
- **Selection Criterion**: Max Validation AUPRC ({best_val_auprc:.5f}).
- **Selected Hyperparameters**: `{best_hyperparams}`
- **Optimal Threshold**: $\\tau = {best_threshold:.2f}$ (selected on Validation set by max F1).
- **Training Fit Time**: {best_train_time:.2f} seconds.
- **Inference Latency**: {inf_latency_ms} ms/window.
- **Git Commit**: `{config_data['git_commit']}`

## Results on Held-Out Test Set (152.82 Continuous Hours)

| Metric | Value |
| :--- | :--- |
| **AUROC** | {test_metrics.auroc:.5f} |
| **AUPRC** | {test_metrics.auprc:.5f} |
| **Sensitivity (Window)** | {test_metrics.sensitivity*100:.2f}% |
| **Specificity (Window)** | {test_metrics.specificity*100:.2f}% |
| **Precision** | {test_metrics.precision*100:.2f}% |
| **F1 Score** | {test_metrics.f1_score:.5f} |
| **Balanced Accuracy** | {(test_metrics.sensitivity + test_metrics.specificity) / 2.0 * 100:.2f}% |
| **Event Sensitivity** | {test_metrics.event_sensitivity * 100:.2f}% ({int(round(test_metrics.event_sensitivity * len(true_events)))}/{len(true_events)}) |
| **Mean Detection Delay** | {test_metrics.detection_delay_sec:.2f} s |
| **False-Positive Windows** | {fp_count:,} |
| **False-Alarm Episodes** | {fa_episodes_count:,} |
| **FA / 24h** | {test_metrics.fa_per_24h:.2f} |
| **Inference Latency** | {inf_latency_ms} ms/window |
"""
    with open(exp_dir / "README.md", "w") as f:
        f.write(readme_content)

    return results_json


def compile_summary_comparison():
    print("\n" + "=" * 70)
    print("COMPILING CLASSICAL ML BASELINES COMPARISON")
    print("=" * 70, flush=True)

    comp_rows = [MODEL_C_FROZEN_METRICS]

    models = [
        ("Random Forest", "random_forest"),
        ("XGBoost", "xgboost"),
        ("LightGBM", "lightgbm"),
        ("Linear SVM", "svm"),
    ]

    for display_name, folder in models:
        metrics_file = BASELINES_DIR / folder / "results" / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                data = json.load(f)
            comp_rows.append({
                "model": data["model"],
                "status": "EVALUATED BASELINE",
                "auroc": data["auroc"],
                "auprc": data["auprc"],
                "sensitivity": data["sensitivity"],
                "specificity": data["specificity"],
                "precision": data["precision"],
                "f1_score": data["f1_score"],
                "balanced_accuracy": data["balanced_accuracy"],
                "event_sensitivity": data["event_sensitivity"],
                "detection_delay_sec": data["detection_delay_sec"],
                "false_positive_windows": data["false_positive_windows"],
                "false_alarm_episodes": data["false_alarm_episodes"],
                "fa_per_24h": data["fa_per_24h"],
                "accuracy": data["accuracy"]
            })

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(BASELINES_DIR / "classical_ml_comparison.csv", index=False)
    print(f"✓ Saved comparison table to: {BASELINES_DIR / 'classical_ml_comparison.csv'}")
    print("\nComparison Table:")
    print(comp_df.to_string(index=False))

    # Master README
    readme_text = f"""# Experiment 3: Classical Machine Learning Baselines

Comprehensive benchmark comparing classical machine learning architectures trained on 57 multi-domain engineered EEG features against frozen **Model C** (CNN + Spatial GNN + Causal GRU).

## Evaluated Models
1. **Random Forest** (`RandomForestClassifier`)
2. **XGBoost** (`XGBClassifier`)
3. **LightGBM** (`LGBMClassifier`)
4. **Linear SVM** (`SGDClassifier` with L2 regularization)
5. **Model C (Frozen Benchmark)**: CNN-GNN-GRU end-to-end spatio-temporal network

## Dataset Protocol & Split
- **Train (16 patients, 10:1 subsampled)**: 36,388 windows
- **Validation (4 patients)**: 293,410 windows
- **Test (4 held-out patients, locked)**: 219,909 windows, 22 seizures, 152.82 continuous hours

## Comparison Summary

| Model | Status | AUROC | AUPRC | Sens (%) | Spec (%) | F1 | Event Sens (%) | Delay (s) | FA/24h |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in comp_rows:
        readme_text += f"| **{r['model']}** | {r['status']} | {r['auroc']:.4f} | {r['auprc']:.4f} | {r['sensitivity']*100:.2f}% | {r['specificity']*100:.2f}% | {r['f1_score']:.4f} | {r['event_sensitivity']*100:.2f}% | {r['detection_delay_sec']:.2f}s | {r['fa_per_24h']:.2f} |\n"

    with open(BASELINES_DIR / "README.md", "w") as f:
        f.write(readme_text)
    print(f"✓ Saved Master README to: {BASELINES_DIR / 'README.md'}")


def main():
    parser = argparse.ArgumentParser(description="NeuroAegis Classical ML Baselines Runner")
    parser.add_argument("--model", type=str, default="all", choices=["rf", "random_forest", "xgb", "xgboost", "lgb", "lightgbm", "svm", "summary", "all"])
    args = parser.parse_args()

    if args.model == "summary":
        compile_summary_comparison()
        return

    # Check patient leakage
    assert_no_patient_leakage(set(TRAIN_PATIENTS), set(TEST_PATIENTS))
    assert_no_patient_leakage(set(TRAIN_PATIENTS), set(VAL_PATIENTS))
    assert_no_patient_leakage(set(VAL_PATIENTS), set(TEST_PATIENTS))
    print("✓ Guardrail passed: Strict patient-level isolation verified.")

    eval_config = load_eval_config()

    # Load splits
    print("\nLoading dataset feature shards...")
    X_train, y_train = load_training_data()
    val_df, X_val, y_val = load_cohort_data(VAL_PATIENTS)
    test_df, X_test, y_test = load_cohort_data(TEST_PATIENTS)

    print(f"\nCohort Overview:")
    print(f"  Train:      {len(y_train):,} windows ({int(y_train.sum()):,} positive, {len(y_train)-int(y_train.sum()):,} negative)")
    print(f"  Validation: {len(y_val):,} windows ({int(y_val.sum()):,} positive)")
    print(f"  Test:       {len(y_test):,} windows ({int(y_test.sum()):,} positive)")

    # Model 1: Random Forest
    if args.model in ["rf", "random_forest", "all"]:
        rf_grid = [
            {"n_estimators": 100, "max_depth": 10, "class_weight": "balanced", "random_state": 42, "n_jobs": 4},
            {"n_estimators": 200, "max_depth": 15, "class_weight": "balanced", "random_state": 42, "n_jobs": 4},
            {"n_estimators": 300, "max_depth": 20, "class_weight": "balanced", "random_state": 42, "n_jobs": 4},
            {"n_estimators": 300, "max_depth": None, "class_weight": "balanced_subsample", "random_state": 42, "n_jobs": 4},
        ]
        train_and_evaluate_model(
            model_name="Random Forest",
            folder_name="random_forest",
            create_model_fn=lambda p: RandomForestClassifier(**p),
            hyperparam_grid=rf_grid,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            eval_config=eval_config,
            test_df=test_df
        )
        flush_memory()

    # Model 2: XGBoost
    if args.model in ["xgb", "xgboost", "all"]:
        spw_train = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
        xgb_grid = [
            {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4, "scale_pos_weight": spw_train, "random_state": 42, "n_jobs": 4},
            {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "scale_pos_weight": spw_train, "random_state": 42, "n_jobs": 4},
            {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 6, "scale_pos_weight": spw_train * 0.5, "random_state": 42, "n_jobs": 4},
            {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 8, "scale_pos_weight": spw_train, "random_state": 42, "n_jobs": 4},
        ]
        train_and_evaluate_model(
            model_name="XGBoost",
            folder_name="xgboost",
            create_model_fn=lambda p: XGBClassifier(**p),
            hyperparam_grid=xgb_grid,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            eval_config=eval_config,
            test_df=test_df
        )
        flush_memory()

    # Model 3: LightGBM
    if args.model in ["lgb", "lightgbm", "all"]:
        spw_train = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
        lgb_grid = [
            {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31, "scale_pos_weight": spw_train, "random_state": 42, "n_jobs": 4, "verbosity": -1},
            {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31, "scale_pos_weight": spw_train, "random_state": 42, "n_jobs": 4, "verbosity": -1},
            {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 8, "num_leaves": 63, "scale_pos_weight": spw_train * 0.5, "random_state": 42, "n_jobs": 4, "verbosity": -1},
            {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 8, "num_leaves": 63, "scale_pos_weight": spw_train, "random_state": 42, "n_jobs": 4, "verbosity": -1},
        ]
        train_and_evaluate_model(
            model_name="LightGBM",
            folder_name="lightgbm",
            create_model_fn=lambda p: LGBMClassifier(**p),
            hyperparam_grid=lgb_grid,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            eval_config=eval_config,
            test_df=test_df
        )
        flush_memory()

    # Model 4: Linear SVM
    if args.model in ["svm", "all"]:
        svm_grid = [
            {"loss": "log_loss", "alpha": 1e-4, "penalty": "l2", "class_weight": "balanced", "random_state": 42, "max_iter": 1000},
            {"loss": "log_loss", "alpha": 1e-3, "penalty": "l2", "class_weight": "balanced", "random_state": 42, "max_iter": 1000},
            {"loss": "log_loss", "alpha": 1e-5, "penalty": "l2", "class_weight": "balanced", "random_state": 42, "max_iter": 1000},
            {"loss": "modified_huber", "alpha": 1e-4, "penalty": "l2", "class_weight": "balanced", "random_state": 42, "max_iter": 1000},
        ]
        train_and_evaluate_model(
            model_name="Linear SVM",
            folder_name="svm",
            create_model_fn=lambda p: SGDClassifier(**p),
            hyperparam_grid=svm_grid,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            eval_config=eval_config,
            test_df=test_df
        )
        flush_memory()

    # Final summary
    compile_summary_comparison()


if __name__ == "__main__":
    main()
