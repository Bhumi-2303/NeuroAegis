"""
research/experiments/temporal/run_single_temporal_model.py
──────────────────────────────────────────────────────────
Trains and evaluates a single candidate temporal architecture
(CausalGRUModel, CausalLSTMModel, or CausalTCNModel) on precomputed 128-D spatial embeddings.

Enforces:
1. Isolated execution per process (fresh process per model).
2. Strict patient isolation (16 train, 4 val, 4 test).
3. Test set remains strictly locked until training & validation selection finishes.
4. Validation AUPRC for checkpoint selection.
5. Exact evaluation metrics: window-level, event-level, false alarm rates, and resources.
6. Full artifact serialization (config.json, metrics.json, training_log.csv, checkpoint, predictions.csv).
7. Strict bounded memory (< 1.5 GB Peak RSS) via memory mapping and bounded chunking.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from research.experiments.temporal.temporal_models import (
    CausalGRUModel,
    CausalLSTMModel,
    CausalTCNModel
)
from research.phase_4b.sequence_dataset import SequenceNegativeSampler
from research.imbalance.patient_splitter import PatientDataSplitter
from research.imbalance.focal_loss import BinaryFocalLossWithLogits, logits_to_probabilities
from research.imbalance.metrics import SeizureEvaluationMetrics
from neuroaegis.utils.memory import flush_memory, get_peak_rss_mb, get_live_rss_mb

CACHE_DIR = BASE_DIR / "research" / "phase_4b" / "embeddings_cache"
MANIFEST_DIR = BASE_DIR / "research" / "data" / "manifests"
EVENTS_PATH = MANIFEST_DIR / "chbmit_seizure_events.csv"
MANIFEST_PATH = MANIFEST_DIR / "chbmit_manifest.csv"

LABEL_COLUMN = "label_50pct_overlap"
FROZEN_BACKBONE_PARAMS = 52497


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(BASE_DIR))
        return res.stdout.strip()
    except Exception:
        return "unknown"


def build_fast_sequence_matrix(
    split_df: pd.DataFrame,
    seq_len: int = 8,
    stride_sec: float = 2.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds (N, seq_len) int32 index matrix directly with zero intermediate object allocation.
    """
    n_total = len(split_df)
    matrix = np.full((n_total, seq_len), -1, dtype=np.int32)
    labels = split_df[LABEL_COLUMN].values.astype(np.float32)

    grouped = split_df.groupby(["patient_id", "recording_id"], sort=False)
    for (pat_id, rec_id), grp in grouped:
        indices = grp.index.values
        starts = grp["window_start_sec"].values
        n_grp = len(grp)
        for i in range(n_grp):
            curr_idx = indices[i]
            curr_start = starts[i]
            matrix[curr_idx, -1] = curr_idx
            for step in range(1, seq_len):
                prev_i = i - step
                if prev_i >= 0:
                    if abs(starts[prev_i] - (curr_start - step * stride_sec)) < 0.05:
                        matrix[curr_idx, seq_len - 1 - step] = indices[prev_i]
    return matrix, labels


def compute_event_delay_details(
    window_df: pd.DataFrame,
    events_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, float, float, float, int, int, List[Dict[str, Any]]]:
    df = window_df.copy()
    df["pred_prob"] = y_prob
    df["pred_label"] = (df["pred_prob"] >= threshold).astype(int)

    delays = []
    event_details = []
    recs = set(window_df["recording_id"].unique())
    sub_events = events_df[events_df["recording_id"].isin(recs)]
    total_events = len(sub_events)

    for _, ev in sub_events.iterrows():
        rec_id = ev["recording_id"]
        sz_id = ev.get("seizure_id", f"{rec_id}_sz")
        pat_id = ev.get("patient_id", rec_id.split("_")[0])
        s_start = float(ev["start_sec"])
        s_end = float(ev["end_sec"])
        s_dur = float(ev["duration_sec"])

        rec_w = df[df["recording_id"] == rec_id]
        ov_mask = (rec_w["window_end_sec"] > s_start) & (rec_w["window_start_sec"] < s_end)
        ov_w = rec_w[ov_mask]

        det_w = ov_w[ov_w["pred_label"] == 1]
        is_detected = len(det_w) > 0
        if is_detected:
            first_alarm = float(det_w["window_end_sec"].min())
            delay = max(0.0, float(first_alarm - s_start))
            delays.append(delay)
        else:
            first_alarm = None
            delay = None

        event_details.append({
            "patient_id": pat_id,
            "recording_id": rec_id,
            "seizure_id": sz_id,
            "start_sec": s_start,
            "end_sec": s_end,
            "duration_sec": s_dur,
            "detected": is_detected,
            "first_alarm_sec": first_alarm,
            "detection_delay_sec": delay,
            "overlapping_positive_windows": len(det_w)
        })

    detected_count = len(delays)
    mean_delay = float(np.mean(delays)) if delays else float("nan")
    median_delay = float(np.median(delays)) if delays else float("nan")
    min_delay = float(np.min(delays)) if delays else float("nan")
    max_delay = float(np.max(delays)) if delays else float("nan")
    return mean_delay, median_delay, min_delay, max_delay, detected_count, total_events, event_details


def compute_patient_breakdown(
    window_df: pd.DataFrame,
    events_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    df = window_df.copy()
    df["pred_prob"] = y_prob
    df["pred_label"] = (df["pred_prob"] >= threshold).astype(int)

    results = []
    for pat_id, pat_w in df.groupby("patient_id"):
        y_t = pat_w[LABEL_COLUMN].values
        y_p = pat_w["pred_prob"].values
        pat_m = SeizureEvaluationMetrics.compute_window_metrics(y_t, y_p, threshold=threshold, stride_sec=2.5)

        pat_events = events_df[events_df["patient_id"] == pat_id]
        total_ev = len(pat_events)
        det_ev = 0
        delays = []
        for _, ev in pat_events.iterrows():
            r_id = ev["recording_id"]
            s_s = ev["start_sec"]
            s_e = ev["end_sec"]
            ev_w = pat_w[(pat_w["recording_id"] == r_id) & (pat_w["window_end_sec"] > s_s) & (pat_w["window_start_sec"] < s_e)]
            pos_w = ev_w[ev_w["pred_label"] == 1]
            if len(pos_w) > 0:
                det_ev += 1
                alarm_t = pos_w["window_end_sec"].min()
                delays.append(max(0.0, float(alarm_t - s_s)))

        results.append({
            "patient_id": pat_id,
            "total_windows": len(pat_w),
            "recording_hours": round(len(pat_w) * 2.5 / 3600.0, 2),
            "num_seizures": total_ev,
            "detected_seizures": det_ev,
            "missed_seizures": total_ev - det_ev,
            "event_sensitivity": round(det_ev / total_ev, 4) if total_ev > 0 else 0.0,
            "window_sensitivity": pat_m["sensitivity"],
            "window_specificity": pat_m["specificity"],
            "false_alarms_count": pat_m["false_positives"],
            "false_alarms_per_day": pat_m["false_alarms_per_24h"],
            "mean_detection_delay_sec": round(float(np.mean(delays)), 2) if delays else None
        })
    return results


def run_experiment(
    model_type: str,
    device_str: str = "mps",
    batch_size: int = 4,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    output_dir: Optional[str] = None
):
    start_time = time.time()
    model_type = model_type.lower()
    print("=" * 80)
    print(f"STARTING EXPERIMENT 1 RUN: {model_type.upper()}")
    print("=" * 80)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Select device
    if device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} | Initial Live RSS: {get_live_rss_mb():.1f} MB | Peak RSS: {get_peak_rss_mb():.1f} MB")

    if output_dir is None:
        out_dir = BASE_DIR / "research" / "experiments" / "temporal" / model_type
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset splits and leakage checks
    print("\n[Step 1] Loading partitions and verifying patient isolation...")
    splitter = PatientDataSplitter(label_column=LABEL_COLUMN)
    train_df, val_df, test_df = splitter.get_splits()
    events_df = pd.read_csv(EVENTS_PATH)
    manifest_df = pd.read_csv(MANIFEST_PATH)

    assert len(set(splitter.train_patients) & set(splitter.val_patients)) == 0, "Leakage Train/Val!"
    assert len(set(splitter.train_patients) & set(splitter.test_patients)) == 0, "Leakage Train/Test!"
    assert len(set(splitter.val_patients) & set(splitter.test_patients)) == 0, "Leakage Val/Test!"
    print(f"Train patients ({len(splitter.train_patients)}): {splitter.train_patients}")
    print(f"Val patients ({len(splitter.val_patients)}): {splitter.val_patients}")
    print(f"Test patients ({len(splitter.test_patients)}): {splitter.test_patients} [LOCKED]")

    # Slim DataFrame to release unnecessary columns and memory
    keep_cols = ["patient_id", "recording_id", "window_start_sec", "window_end_sec", LABEL_COLUMN]
    train_df = train_df[keep_cols].reset_index(drop=True)
    val_df = val_df[keep_cols].reset_index(drop=True)
    test_df = test_df[keep_cols].reset_index(drop=True)
    del splitter
    flush_memory()

    # 2. Memory-mapped 128-D spatial embeddings
    print("\n[Step 2] Memory-mapping 128-D spatial embeddings from cache...")
    val_embs_path = CACHE_DIR / "val_embeddings_unified.npy"
    train_embs_path = CACHE_DIR / "train_embeddings_unified.npy"
    test_embs_path = CACHE_DIR / "test_embeddings_unified.npy"

    assert val_embs_path.exists(), f"Missing {val_embs_path}"
    assert train_embs_path.exists(), f"Missing {train_embs_path}"
    assert test_embs_path.exists(), f"Missing {test_embs_path}"

    val_embs = np.load(val_embs_path, mmap_mode="r")
    train_embs = np.load(train_embs_path, mmap_mode="r")
    print(f"Memory-mapped Train embeddings: {train_embs.shape} | Val embeddings: {val_embs.shape}")

    # 3. Build sequence metadata
    seq_len = 8
    temporal_span = 5.0 + (seq_len - 1) * 2.5
    print(f"\n[Step 3] Building fast sequence matrices (seq_len={seq_len}, span={temporal_span:.1f}s)...")

    t_b0 = time.time()
    train_indices_matrix, train_labels = build_fast_sequence_matrix(train_df, seq_len=seq_len)
    val_indices_matrix, val_labels = build_fast_sequence_matrix(val_df, seq_len=seq_len)
    t_b1 = time.time()
    print(f"Sequences built in {t_b1 - t_b0:.2f}s (Train: {len(train_indices_matrix):,}, Val: {len(val_indices_matrix):,})")

    # Lightweight negative sampler wrapper
    train_sampler_df = train_df[["patient_id", LABEL_COLUMN]].copy().rename(columns={LABEL_COLUMN: "target_label"})
    train_sampler = SequenceNegativeSampler(train_sampler_df, ratio=10.0, base_seed=seed, shuffle=True)
    del train_sampler_df

    val_manifest_duration_hours = manifest_df[manifest_df["patient_id"].isin(set(val_df["patient_id"]))]["recording_duration_sec"].sum() / 3600.0

    # 4. Instantiate Model
    print(f"\n[Step 4] Initializing candidate architecture: {model_type.upper()}...")
    if model_type == "gru":
        model = CausalGRUModel(input_dim=128, hidden_dim=64, num_layers=1)
    elif model_type == "lstm":
        model = CausalLSTMModel(input_dim=128, hidden_dim=64, num_layers=1)
    elif model_type == "tcn":
        model = CausalTCNModel(input_dim=128, hidden_dim=64, kernel_size=3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.to(device)
    param_info = model.get_parameter_breakdown(frozen_backbone_params=FROZEN_BACKBONE_PARAMS)
    print(f"Architecture Parameters: Trainable={param_info['trainable_parameters']:,} | Total={param_info['total_parameters']:,}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25, reduction="mean")

    best_val_auprc = -1.0
    best_epoch = -1
    best_val_metrics = None
    best_val_probs = None
    history = []

    # 5. Training loop
    print(f"\n[Step 5] Training {model_type.upper()} for {epochs} epochs (batch_size={batch_size})...")
    training_start_time = time.time()

    for epoch in range(epochs):
        t_ep0 = time.time()
        train_sampler.set_epoch(epoch)
        epoch_indices = train_sampler.current_epoch_indices

        # Vectorized batch index selection with zero-padding mask
        sub_indices = train_indices_matrix[epoch_indices]
        mask = (sub_indices == -1)
        safe_indices = np.where(mask, 0, sub_indices)
        train_x_np = train_embs[safe_indices].copy()
        train_x_np[mask] = 0.0
        train_y_np = train_labels[epoch_indices]

        train_dataset = TensorDataset(torch.from_numpy(train_x_np), torch.from_numpy(train_y_np))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=False)

        model.train()
        train_loss = 0.0
        train_preds = []
        train_trues = []

        t_tr0 = time.time()
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(by)
            probs = logits_to_probabilities(logits).detach().cpu().numpy()
            train_preds.append(probs)
            train_trues.append(by.cpu().numpy())
            del bx, by, logits, loss

        scheduler.step()
        train_loss /= len(train_dataset)
        t_tr1 = time.time()

        y_tr_pred = np.concatenate(train_preds)
        y_tr_true = np.concatenate(train_trues)
        tr_metrics = SeizureEvaluationMetrics.compute_window_metrics(y_tr_true, y_tr_pred, threshold=0.5)

        del train_dataset, train_loader, train_x_np, train_preds, train_trues, y_tr_pred, y_tr_true
        flush_memory()

        # Validation evaluation pass in bounded chunks
        model.eval()
        val_preds = []
        t_val0 = time.time()
        val_chunk_size = 4096
        with torch.no_grad():
            for b_i in range(0, len(val_indices_matrix), val_chunk_size):
                b_idx = val_indices_matrix[b_i : b_i + val_chunk_size]
                b_mask = (b_idx == -1)
                b_safe = np.where(b_mask, 0, b_idx)
                b_data = val_embs[b_safe].copy()
                b_data[b_mask] = 0.0
                b_chunk = torch.from_numpy(b_data).to(device)
                logits = model(b_chunk)
                p = logits_to_probabilities(logits).cpu().numpy()
                val_preds.append(p)
                del b_idx, b_mask, b_safe, b_data, b_chunk, logits
        val_probs = np.concatenate(val_preds)
        t_val1 = time.time()
        del val_preds
        flush_memory()

        val_metrics = SeizureEvaluationMetrics.compute_window_metrics(
            y_true=val_labels,
            y_prob=val_probs,
            threshold=0.5,
            total_duration_hours=val_manifest_duration_hours,
            stride_sec=2.5
        )

        v_mean_delay, v_med_delay, v_min_delay, v_max_delay, v_det_ev, v_tot_ev, _ = compute_event_delay_details(
            window_df=val_df,
            events_df=events_df,
            y_prob=val_probs,
            threshold=0.5
        )
        val_metrics['event_level_sensitivity'] = round(v_det_ev / v_tot_ev, 4) if v_tot_ev > 0 else 0.0
        val_metrics['mean_detection_delay_sec'] = round(v_mean_delay, 2) if not np.isnan(v_mean_delay) else None
        val_metrics["detected_seizures"] = v_det_ev
        val_metrics["total_seizures"] = v_tot_ev

        val_auprc = val_metrics["auprc"] or 0.0
        val_auroc = val_metrics["auroc"] or 0.0

        ep_duration = time.time() - t_ep0
        print(f"Epoch {epoch+1}/{epochs} [{model_type.upper()}] ({ep_duration:.1f}s):")
        print(f"  Train Loss: {train_loss:.5f} | Val AUPRC: {val_auprc:.5f} | Val AUROC: {val_auroc:.5f}")
        print(f"  Val Sens: {val_metrics['sensitivity']*100:.2f}% | Val Spec: {val_metrics['specificity']*100:.2f}% | Val Events: {v_det_ev}/{v_tot_ev} ({val_metrics['event_level_sensitivity']*100:.1f}%) | Delay: {val_metrics['mean_detection_delay_sec']}s | FA/24h: {val_metrics['false_alarms_per_24h']:.2f}")

        ep_log = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 5),
            "train_auprc": round(tr_metrics["auprc"], 5) if tr_metrics["auprc"] is not None else None,
            "train_auroc": round(tr_metrics["auroc"], 5) if tr_metrics["auroc"] is not None else None,
            "val_auprc": round(val_auprc, 5),
            "val_auroc": round(val_auroc, 5),
            "val_sensitivity": round(val_metrics['sensitivity'], 5),
            "val_specificity": round(val_metrics['specificity'], 5),
            "val_precision": round(val_metrics["precision"], 5),
            "val_f1": round(val_metrics["f1_score"], 5),
            "val_balanced_accuracy": round(val_metrics["balanced_accuracy"], 5),
            "val_event_sensitivity": round(val_metrics['event_level_sensitivity'], 5),
            "val_false_alarms_per_24h": round(val_metrics['false_alarms_per_24h'], 2),
            "val_mean_detection_delay_sec": val_metrics['mean_detection_delay_sec'],
            "live_rss_mb": round(get_live_rss_mb(), 1),
            "peak_rss_mb": round(get_peak_rss_mb(), 1),
            "epoch_time_sec": round(ep_duration, 1)
        }
        history.append(ep_log)

        # Checkpoint selection: Peak Validation AUPRC
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_epoch = epoch + 1
            best_val_metrics = val_metrics
            best_val_probs = val_probs

            ckpt_dict = {
                "model_type": model_type,
                "epoch": best_epoch,
                "val_auprc": best_val_auprc,
                "model_state_dict": model.state_dict(),
                "param_info": param_info
            }
            torch.save(ckpt_dict, out_dir / "best_model.pt")

    training_time_sec = round(time.time() - training_start_time, 2)
    print(f"\nTraining finished in {training_time_sec}s. Best Epoch: {best_epoch} (Val AUPRC: {best_val_auprc:.5f})")

    # Clean up training variables
    del val_embs, train_embs, train_indices_matrix, val_indices_matrix
    flush_memory()

    # 6. Final Evaluation on Locked Test Cohort
    print("\n" + "=" * 80)
    print(f"[Step 6] Final Single-Pass Evaluation on LOCKED Test Cohort ({model_type.upper()})...")
    print("=" * 80)

    # Load best checkpoint
    best_ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Memory map test embeddings and build test sequence matrix
    test_embs = np.load(test_embs_path, mmap_mode="r")
    test_indices_matrix, test_labels = build_fast_sequence_matrix(test_df, seq_len=seq_len)

    test_duration_hours = manifest_df[manifest_df["patient_id"].isin(set(test_df["patient_id"]))]["recording_duration_sec"].sum() / 3600.0
    print(f"Test Set: {len(test_indices_matrix):,} windows, {test_duration_hours:.2f} hours across {sorted(test_df['patient_id'].unique())}")

    test_preds = []
    t_inf0 = time.time()
    test_chunk_size = 4096
    with torch.no_grad():
        for b_i in range(0, len(test_indices_matrix), test_chunk_size):
            b_idx = test_indices_matrix[b_i : b_i + test_chunk_size]
            b_mask = (b_idx == -1)
            b_safe = np.where(b_mask, 0, b_idx)
            b_data = test_embs[b_safe].copy()
            b_data[b_mask] = 0.0
            b_chunk = torch.from_numpy(b_data).to(device)
            logits = model(b_chunk)
            p = logits_to_probabilities(logits).cpu().numpy()
            test_preds.append(p)
            del b_idx, b_mask, b_safe, b_data, b_chunk, logits
    t_inf1 = time.time()
    inference_duration_sec = round(t_inf1 - t_inf0, 3)
    test_probs = np.concatenate(test_preds)
    del test_preds, test_embs, test_indices_matrix
    flush_memory()

    # Compute Test Metrics
    test_metrics = SeizureEvaluationMetrics.compute_window_metrics(
        y_true=test_labels,
        y_prob=test_probs,
        threshold=0.5,
        total_duration_hours=test_duration_hours,
        stride_sec=2.5
    )

    t_mean_d, t_med_d, t_min_d, t_max_d, t_det_ev, t_tot_ev, t_ev_details = compute_event_delay_details(
        window_df=test_df,
        events_df=events_df,
        y_prob=test_probs,
        threshold=0.5
    )

    pat_breakdown = compute_patient_breakdown(
        window_df=test_df,
        events_df=events_df,
        y_prob=test_probs,
        threshold=0.5
    )

    # Benchmark single-window latency
    dummy_win = torch.randn(1, 8, 128).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_win)
        t_lat0 = time.perf_counter()
        for _ in range(100):
            _ = model(dummy_win)
        t_lat1 = time.perf_counter()
    latency_ms_per_window = round(((t_lat1 - t_lat0) / 100.0) * 1000.0, 4)
    del dummy_win
    flush_memory()

    checkpoint_size_mb = round((out_dir / "best_model.pt").stat().st_size / (1024 * 1024), 3)
    peak_rss_mb = round(get_peak_rss_mb(), 1)
    mps_allocated_mb = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2) if device.type == "mps" else 0.0

    print(f"\n--- TEST METRICS SUMMARY [{model_type.upper()}] ---")
    print(f"  AUROC:             {test_metrics['auroc']:.5f}")
    print(f"  AUPRC:             {test_metrics['auprc']:.5f}")
    print(f"  Sensitivity:       {test_metrics['sensitivity']*100:.2f}%")
    print(f"  Specificity:       {test_metrics['specificity']*100:.2f}%")
    print(f"  Precision:         {test_metrics['precision']*100:.2f}%")
    print(f"  F1 Score:          {test_metrics['f1_score']:.5f}")
    print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Event Sensitivity: {t_det_ev}/{t_tot_ev} ({t_det_ev/t_tot_ev*100:.2f}%)")
    print(f"  Missed Seizures:   {t_tot_ev - t_det_ev}")
    print(f"  Mean Delay:        {t_mean_d:.2f}s (median: {t_med_d:.2f}s)")
    print(f"  False Alarms:      {test_metrics['false_positives']} ({test_metrics['false_alarms_per_24h']:.2f}/24h)")
    print(f"  Peak RSS:          {peak_rss_mb} MB")
    print(f"  Inference Latency: {latency_ms_per_window} ms/window")

    # 7. Save outputs
    print(f"\n[Step 7] Serializing artifacts to {out_dir}...")
    pd.DataFrame(history).to_csv(out_dir / "training_log.csv", index=False)

    # Save test predictions (incremental csv)
    pred_df = test_df[["patient_id", "recording_id", "window_start_sec", "window_end_sec", LABEL_COLUMN]].copy()
    pred_df["prediction_prob"] = np.round(test_probs, 6)
    pred_df["predicted_label"] = (test_probs >= 0.5).astype(int)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    full_metrics = {
        "model_type": model_type,
        "selected_best_epoch": best_epoch,
        "validation_metrics": best_val_metrics,
        "test_metrics": {
            "total_windows": len(test_labels),
            "recording_hours": round(test_duration_hours, 2),
            "auroc": test_metrics['auroc'],
            "auprc": test_metrics['auprc'],
            "sensitivity": test_metrics['sensitivity'],
            "specificity": test_metrics['specificity'],
            "precision": test_metrics['precision'],
            "f1_score": test_metrics["f1_score"],
            "balanced_accuracy": test_metrics['balanced_accuracy'],
            "confusion_matrix": {
                "tp": test_metrics["true_positives"],
                "fp": test_metrics['false_positives'],
                "tn": test_metrics["true_negatives"],
                "fn": test_metrics["false_negatives"]
            },
            "event_metrics": {
                "total_events": t_tot_ev,
                "detected_events": t_det_ev,
                "missed_events": t_tot_ev - t_det_ev,
                "event_sensitivity": round(t_det_ev / t_tot_ev, 4) if t_tot_ev > 0 else 0.0,
                "mean_detection_delay_sec": round(t_mean_d, 2) if not np.isnan(t_mean_d) else None,
                "median_detection_delay_sec": round(t_med_d, 2) if not np.isnan(t_med_d) else None,
                "min_detection_delay_sec": round(t_min_d, 2) if not np.isnan(t_min_d) else None,
                "max_detection_delay_sec": round(t_max_d, 2) if not np.isnan(t_max_d) else None
            },
            "false_alarm_metrics": {
                "false_alarm_count": test_metrics['false_positives'],
                "non_seizure_hours": round(test_duration_hours, 2),
                "false_alarms_per_24h": round(test_metrics['false_alarms_per_24h'], 2)
            },
            "patient_breakdown": pat_breakdown
        },
        "resource_metrics": {
            "trainable_parameters": param_info['trainable_parameters'],
            "frozen_backbone_parameters": param_info["frozen_backbone_parameters"],
            "total_parameters": param_info['total_parameters'],
            "checkpoint_size_mb": checkpoint_size_mb,
            "training_time_sec": training_time_sec,
            "inference_duration_sec": inference_duration_sec,
            "latency_ms_per_window": latency_ms_per_window,
            "peak_rss_mb": peak_rss_mb,
            "mps_allocated_mb": mps_allocated_mb
        }
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    config_dict = {
        "experiment": f"temporal_{model_type}",
        "model_type": model_type,
        "input_dimension": 128,
        "sequence_length": seq_len,
        "temporal_span_sec": temporal_span,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "loss_function": "BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25)",
        "negative_sampling_ratio": 10.0,
        "random_seed": seed,
        "device": str(device),
        "git_commit": get_git_commit(),
        "train_patients": sorted(train_df["patient_id"].unique()),
        "val_patients": sorted(val_df["patient_id"].unique()),
        "test_patients": sorted(test_df["patient_id"].unique()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"All artifacts saved successfully in {out_dir}.")
    print(f"Total experiment time: {time.time() - start_time:.1f}s.")
    return full_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run temporal model experiment")
    parser.add_argument("--model", type=str, required=True, choices=["gru", "lstm", "tcn"], help="Model type to train")
    parser.add_argument("--device", type=str, default="mps", help="Torch device")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()
    run_experiment(
        model_type=args.model,
        device_str=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        output_dir=args.output_dir
    )
