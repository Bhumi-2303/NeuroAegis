#!/usr/bin/env python3
"""
retrain_chbmit_patient_wise.py
──────────────────────────────
Retrain LightGBM on chbmit_subset.parquet using patient-wise
Leave-One-Patient-Out Cross-Validation (LOPO-CV), then train a final
model on the full dataset and save it.

Uses the same hyperparameters as the original lightgbm_baseline.pkl
for a fair comparison.

Output:
  - apps/api/models/chbmit/lightgbm_patient_wise.pkl
  - apps/api/models/chbmit/patient_wise_metadata.json
"""

import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "chbmit_subset.parquet"
OUTPUT_DIR = REPO_ROOT / "apps" / "api" / "models" / "chbmit"
OUTPUT_MODEL = OUTPUT_DIR / "lightgbm_patient_wise.pkl"
OUTPUT_META = OUTPUT_DIR / "patient_wise_metadata.json"

# ── Original baseline hyperparameters (extracted from lightgbm_baseline.pkl) ─
LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": -1,
    "num_leaves": 31,
    "min_child_samples": 20,
    "min_child_weight": 0.001,
    "min_split_gain": 0.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "importance_type": "split",
    "verbosity": -1,
    # scale_pos_weight will be computed from data
}

DROP_COLS = ["target", "patient_id", "record", "window_idx"]


def compute_metrics(y_true, y_pred, y_prob):
    """Compute a dict of standard binary classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    # AUC can fail if only one class is present in the fold
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def main():
    print("=" * 70)
    print("NeuroAegis – CHB-MIT Patient-Wise LightGBM Retraining")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_parquet(DATA_PATH)
    print(f"\nDataset: {DATA_PATH}")
    print(f"  Shape: {df.shape}")

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].values
    y = df["target"].values
    groups = df["patient_id"].values

    patients = sorted(df["patient_id"].unique())
    print(f"  Patients: {patients}")
    print(f"  Target distribution: 0={int((y==0).sum())}, 1={int((y==1).sum())}")
    print(f"  Features: {len(feature_cols)}")

    # ── Compute scale_pos_weight from full data ──────────────────────
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"  scale_pos_weight (computed): {scale_pos_weight:.4f}")

    # ── Leave-One-Patient-Out Cross-Validation ───────────────────────
    print("\n" + "-" * 70)
    print("Leave-One-Patient-Out Cross-Validation")
    print("-" * 70)

    logo = LeaveOneGroupOut()
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_patient = groups[test_idx][0]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        n_seizure_test = int(y_test.sum())
        n_seizure_train = int(y_train.sum())

        print(f"\n  Fold {fold_i + 1}: Test patient = {test_patient}")
        print(f"    Train: {len(y_train)} samples ({n_seizure_train} seizures)")
        print(f"    Test:  {len(y_test)} samples ({n_seizure_test} seizures)")

        # Impute + scale
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test_proc = scaler.transform(imputer.transform(X_test))

        # Per-fold scale_pos_weight
        fold_n_neg = int((y_train == 0).sum())
        fold_n_pos = int((y_train == 1).sum())
        fold_spw = fold_n_neg / fold_n_pos if fold_n_pos > 0 else 1.0

        params = LGBM_PARAMS.copy()
        params["scale_pos_weight"] = fold_spw

        model = LGBMClassifier(**params)
        model.fit(X_train_proc, y_train)

        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["test_patient"] = test_patient
        metrics["test_samples"] = len(y_test)
        metrics["test_seizures"] = n_seizure_test
        fold_results.append(metrics)

        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")
        print(f"    ROC AUC:   {metrics['roc_auc']:.4f}" if not np.isnan(metrics['roc_auc']) else f"    ROC AUC:   N/A (single class in test)")
        print(f"    MCC:       {metrics['mcc']:.4f}")

    # ── Aggregate CV metrics ─────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Aggregated LOPO-CV Metrics (macro-average across folds)")
    print("-" * 70)

    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]
    agg = {}
    for k in metric_keys:
        values = [r[k] for r in fold_results if not np.isnan(r[k])]
        agg[k] = {"mean": float(np.mean(values)), "std": float(np.std(values))} if values else {"mean": float("nan"), "std": float("nan")}
        print(f"  {k:12s}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")

    # ── Train final model on ALL data ────────────────────────────────
    print("\n" + "-" * 70)
    print("Training final model on ALL data")
    print("-" * 70)

    imputer_final = SimpleImputer(strategy="median")
    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(imputer_final.fit_transform(X))

    final_params = LGBM_PARAMS.copy()
    final_params["scale_pos_weight"] = scale_pos_weight

    final_model = LGBMClassifier(**final_params)
    final_model.fit(X_all, y)

    print(f"  Model trained on {len(y)} samples.")

    # ── Save model ───────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(final_model, OUTPUT_MODEL)
    joblib.dump(imputer_final, OUTPUT_DIR / "imputer.pkl")
    joblib.dump(scaler_final, OUTPUT_DIR / "scaler.pkl")
    print(f"\n  ✓ Model saved to: {OUTPUT_MODEL}")

    # ── Save metadata ────────────────────────────────────────────────
    metadata = {
        "model": "LightGBM",
        "dataset": "CHB-MIT",
        "split_strategy": "Leave-One-Patient-Out (LOPO-CV)",
        "patients": patients,
        "features": len(feature_cols),
        "feature_names": feature_cols,
        "total_samples": len(y),
        "seizure_samples": int(n_pos),
        "non_seizure_samples": int(n_neg),
        "scale_pos_weight": round(scale_pos_weight, 4),
        "hyperparameters": {k: v for k, v in LGBM_PARAMS.items() if k != "verbosity"},
        "cv_results": {
            "strategy": "Leave-One-Patient-Out",
            "n_folds": len(fold_results),
            "per_fold": fold_results,
            "aggregate": agg,
        },
        "training_date": pd.Timestamp.now().isoformat(),
        "framework": "NeuroAegis",
        "notes": (
            "Retrained with patient-wise split to eliminate data leakage. "
            "Same hyperparameters as original lightgbm_baseline.pkl for fair comparison."
        ),
    }

    with open(OUTPUT_META, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  ✓ Metadata saved to: {OUTPUT_META}")

    # Also update selected_features.json to match the 57-feature parquet
    selected_features_path = OUTPUT_DIR / "selected_features_patient_wise.json"
    with open(selected_features_path, "w") as f:
        json.dump(feature_cols, f, indent=4)
    print(f"  ✓ Feature list saved to: {selected_features_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
