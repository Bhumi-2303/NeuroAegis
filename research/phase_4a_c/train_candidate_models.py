"""
NeuroAegis Phase 4A-C: Training Engine for Candidate Graph Thresholds
Trains two independent CNN + GNN models under frozen Phase 4A hyperparameters:
  - Run 1: theta = 0.25 (research/phase_4a_c/theta_025/)
  - Run 2: theta = 0.30 (research/phase_4a_c/theta_030/)
Evaluates all models and the frozen Phase 4A reference (theta = 0.35) on the Validation set ONLY.
Computes:
  - Epoch-wise window metrics (14 metrics per epoch)
  - Full Validation Event Metrics (25 seizures across 4 validation patients)
  - Validation False Alarms per 24 hours (203.76 hours)
Exports:
  - research/phase_4a_c/validation_threshold_comparison.csv
  - Checkpoints and training histories in candidate subdirectories
"""

import os
import sys
import time
import json
import hashlib
import resource
import platform
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

from research.phase_4a.cnn_gnn_model import Baseline1DCNN_GNN
from research.phase_3.data_loader import CHBMITDataPipeline
from research.imbalance.patient_splitter import PatientDataSplitter
from research.imbalance.dynamic_sampler import DynamicNegativeSampler
from research.imbalance.focal_loss import BinaryFocalLossWithLogits, logits_to_probabilities
from research.imbalance.metrics import SeizureEvaluationMetrics

# Invariant Configuration
LABEL_COLUMN = "label_50pct_overlap"
SAMPLING_RATIO = 10.0
BASE_SEED = 42
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 3
TOTAL_VAL_HOURS = 293410 * 2.5 / 3600.0  # ~203.757 hours

MANIFEST_DIR = os.path.join(BASE_DIR, "research/data/manifests")
WINDOW_INDEX_PATH = os.path.join(MANIFEST_DIR, "chbmit_window_index.csv")
EVENTS_PATH = os.path.join(MANIFEST_DIR, "chbmit_seizure_events.csv")
EDF_ROOT_DIR = os.path.join(BASE_DIR, "CHB-MIT Dataset")
PHASE4A_C_DIR = os.path.join(BASE_DIR, "research/phase_4a_c")
PHASE4A_EXP_DIR = os.path.join(BASE_DIR, "research/phase_4a/cnn_gnn/exp_01")
PHASE4A_CHECKPOINT = os.path.join(PHASE4A_EXP_DIR, "best_cnn_gnn.pt")


def get_peak_memory_mb() -> float:
    if platform.system() == "Darwin":
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def update_live_status(
    status_file: str,
    stage: str,
    completed_steps: List[str],
    in_progress: str,
    queued_steps: List[str],
    start_time: float,
    current_epoch: int = 0,
    total_epochs: int = NUM_EPOCHS,
    extra_info: Optional[Dict[str, Any]] = None
):
    elapsed = time.time() - start_time
    status = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "elapsed_seconds": round(elapsed, 1),
        "in_progress": in_progress,
        "completed_steps": completed_steps,
        "queued_steps": queued_steps,
        "peak_memory_mb": round(get_peak_memory_mb(), 2)
    }
    if extra_info:
        status.update(extra_info)
    try:
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)
    except Exception:
        pass


def evaluate_event_latency(
    window_df: pd.DataFrame,
    events_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, int, int, List[Dict[str, Any]]]:
    df = window_df.copy()
    df["pred_prob"] = y_prob
    df["pred_label"] = (df["pred_prob"] >= threshold).astype(int)
    
    delays = []
    event_details = []
    val_recs = set(window_df["recording_id"].unique())
    sub_events = events_df[events_df["recording_id"].isin(val_recs)]
    total_events = len(sub_events)
    
    for _, ev in sub_events.iterrows():
        rec_id = ev["recording_id"]
        sz_id = ev.get("seizure_id", f"{rec_id}_sz")
        pat_id = ev.get("patient_id", rec_id.split("_")[0])
        s_start = ev["start_sec"]
        s_end = ev["end_sec"]
        s_dur = ev["duration_sec"]
        
        rec_w = df[df["recording_id"] == rec_id]
        ov_mask = (rec_w["window_end_sec"] > s_start) & (rec_w["window_start_sec"] < s_end)
        ov_w = rec_w[ov_mask]
        
        det_w = ov_w[ov_w["pred_label"] == 1]
        is_detected = len(det_w) > 0
        if is_detected:
            first_alarm = det_w["window_end_sec"].min()
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
    return round(mean_delay, 2), detected_count, total_events, event_details


def train_single_candidate(
    candidate_name: str,
    threshold: float,
    candidate_dir: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    events_df: pd.DataFrame,
    pipeline: CHBMITDataPipeline,
    sampler: DynamicNegativeSampler,
    X_pos: torch.Tensor,
    y_pos: torch.Tensor,
    device: torch.device
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"TRAINING CANDIDATE: {candidate_name} (theta = {threshold:.2f})")
    print(f"Destination: {candidate_dir}")
    print("=" * 80)
    
    start_time = time.time()
    os.makedirs(candidate_dir, exist_ok=True)
    status_file = os.path.join(candidate_dir, "live_status.json")
    checkpoint_path = os.path.join(candidate_dir, "best_cnn_gnn.pt")
    history_csv_path = os.path.join(candidate_dir, "training_history.csv")
    metrics_json_path = os.path.join(candidate_dir, "validation_metrics.json")
    
    completed_steps = []
    queued_steps = [
        "Load Spatial Adjacency",
        "Initialize Model Architecture",
        "Epoch 1 Training & Validation",
        "Epoch 2 Training & Validation",
        "Epoch 3 Training & Validation",
        "Validation Event & Latency Evaluation"
    ]
    
    update_live_status(status_file, "INIT", completed_steps, "Loading Adjacency", queued_steps, start_time)
    
    # Load adjacency
    adj_path = os.path.join(candidate_dir, "graph_adjacency.csv")
    adj_df = pd.read_csv(adj_path, index_col=0)
    adj_matrix = adj_df.values.astype(np.float32)
    
    # Model setup
    torch.manual_seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    
    model = Baseline1DCNN_GNN(in_channels=23, num_classes=1, adj_matrix=adj_matrix).to(device)
    criterion = BinaryFocalLossWithLogits(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    
    completed_steps.append("Initialize Model Architecture (PASS)")
    queued_steps.remove("Load Spatial Adjacency")
    queued_steps.remove("Initialize Model Architecture")
    
    best_epoch = -1
    best_val_auprc = -1.0
    best_val_prob = None
    best_val_true = None
    history = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        sampler.set_epoch(epoch)
        
        in_prog_str = f"Epoch {epoch+1}/{NUM_EPOCHS}: Sampling 33,080 Dynamic Negatives"
        update_live_status(status_file, "TRAINING", completed_steps, in_prog_str, queued_steps, start_time, epoch+1, NUM_EPOCHS)
        
        print(f"\n--- [{candidate_name}] Epoch {epoch+1}/{NUM_EPOCHS} ---")
        t_load0 = time.time()
        neg_indices = sampler.current_sampled_neg_indices
        X_neg, y_neg = pipeline.load_epoch_windows(
            df=train_df,
            sampled_indices=neg_indices,
            label_column=LABEL_COLUMN,
            shuffle=False
        )
        t_load1 = time.time()
        print(f"Loaded {len(X_neg):,} negative windows in {t_load1 - t_load0:.1f}s")
        
        # Combine and shuffle
        X_epoch = torch.cat([X_pos, X_neg], dim=0)
        y_epoch = torch.cat([y_pos, y_neg], dim=0)
        perm = torch.randperm(len(X_epoch))
        X_epoch = X_epoch[perm]
        y_epoch = y_epoch[perm]
        
        dataset = TensorDataset(X_epoch, y_epoch)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        
        in_prog_str = f"Epoch {epoch+1}/{NUM_EPOCHS}: Training on 36,388 Windows"
        update_live_status(status_file, "TRAINING", completed_steps, in_prog_str, queued_steps, start_time, epoch+1, NUM_EPOCHS)
        
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_train0 = time.time()
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        t_train1 = time.time()
        print(f"Epoch {epoch+1} Training: Loss = {avg_train_loss:.5f} ({t_train1 - t_train0:.1f}s)")
        
        # Validation Evaluation
        in_prog_str = f"Epoch {epoch+1}/{NUM_EPOCHS}: Validation Evaluation (293,410 Windows)"
        update_live_status(status_file, "VALIDATION", completed_steps, in_prog_str, queued_steps, start_time, epoch+1, NUM_EPOCHS)
        
        print(f"Evaluating Validation set (293,410 windows across 82 recordings)...")
        t_val0 = time.time()
        val_true, val_prob = pipeline.evaluate_split(
            model=model,
            split_df=val_df,
            device=device,
            label_column=LABEL_COLUMN,
            batch_size=256
        )
        t_val1 = time.time()
        
        val_metrics = SeizureEvaluationMetrics.compute_window_metrics(
            y_true=val_true,
            y_prob=val_prob,
            threshold=0.5,
            stride_sec=2.5
        )
        
        # Validation loss
        with torch.no_grad():
            t_val_true = torch.from_numpy(val_true).float()
            t_val_prob = torch.from_numpy(val_prob).float()
            eps = 1e-7
            p_clamped = torch.clamp(t_val_prob, eps, 1.0 - eps)
            t_val_logits = torch.log(p_clamped / (1.0 - p_clamped))
            val_loss = criterion(t_val_logits, t_val_true).item()
            
        val_auprc = float(val_metrics["auprc"]) if val_metrics["auprc"] is not None else 0.0
        val_auroc = float(val_metrics["auroc"]) if val_metrics["auroc"] is not None else 0.0
        val_sens = float(val_metrics["sensitivity"])
        val_spec = float(val_metrics["specificity"])
        val_prec = float(val_metrics["precision"])
        val_rec = val_sens
        val_f1 = float(val_metrics["f1_score"])
        val_acc = float(val_metrics["accuracy"])
        
        epoch_dur = round(time.time() - epoch_start, 1)
        peak_mem = round(get_peak_memory_mb(), 2)
        
        print(f"Epoch {epoch+1} Validation ({t_val1 - t_val0:.1f}s):")
        print(f"  Val Loss:        {val_loss:.5f}")
        print(f"  Val AUPRC:       {val_auprc:.5f} (Primary Selection Metric)")
        print(f"  Val AUROC:       {val_auroc:.5f}")
        print(f"  Val Sensitivity: {val_sens*100:.2f}%")
        print(f"  Val Specificity: {val_spec*100:.2f}%")
        print(f"  Val Precision:   {val_prec:.5f}")
        print(f"  Val F1:          {val_f1:.5f}")
        
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 5),
            "val_loss": round(val_loss, 5),
            "validation_accuracy": round(val_acc, 5),
            "validation_precision": round(val_prec, 5),
            "validation_recall": round(val_rec, 5),
            "validation_f1": round(val_f1, 5),
            "validation_auroc": round(val_auroc, 5),
            "validation_auprc": round(val_auprc, 5),
            "validation_sensitivity": round(val_sens, 5),
            "validation_specificity": round(val_spec, 5),
            "learning_rate": current_lr,
            "epoch_duration": epoch_dur,
            "peak_memory": peak_mem
        }
        history.append(epoch_record)
        
        # Checkpoint based on peak validation AUPRC
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_epoch = epoch + 1
            best_val_prob = np.copy(val_prob)
            best_val_true = np.copy(val_true)
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auprc": best_val_auprc,
                "val_auroc": val_auroc,
                "val_sensitivity": val_sens,
                "val_specificity": val_spec,
                "val_f1": val_f1,
                "threshold": threshold,
                "candidate": candidate_name
            }, checkpoint_path)
            print(f"  -> Checkpointed new BEST model at Epoch {best_epoch} (Val AUPRC: {best_val_auprc:.5f})")
            
        step_str = f"Epoch {epoch+1} Training & Validation"
        completed_steps.append(step_str)
        if step_str in queued_steps:
            queued_steps.remove(step_str)
            
    # Save training history
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(history_csv_path, index=False)
    print(f"Saved training history to {history_csv_path}")
    
    # Validation Event Evaluation on best checkpoint
    print(f"\n[Validation Event Evaluation on Best Checkpoint (Epoch {best_epoch})]")
    mean_delay, det_events, tot_events, event_details = evaluate_event_latency(
        window_df=val_df,
        events_df=events_df,
        y_prob=best_val_prob,
        threshold=0.5
    )
    event_sens = det_events / tot_events if tot_events > 0 else 0.0
    
    # Compute confusion matrix and false alarms
    y_pred_binary = (best_val_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(best_val_true, y_pred_binary, labels=[0, 1]).ravel()
    val_fa_per_day = float(round((fp / TOTAL_VAL_HOURS) * 24.0, 2))
    bal_acc = float(round(0.5 * ((tp / (tp + fn)) + (tn / (tn + fp))), 5))
    
    print(f"Validation Event Sensitivity: {det_events}/{tot_events} ({event_sens*100:.2f}%)")
    print(f"Validation False Alarms/Day:  {val_fa_per_day} FA/24h (FP={fp:,} over {TOTAL_VAL_HOURS:.2f} hours)")
    print(f"Validation Mean Delay:        {mean_delay}s")
    print(f"Validation Balanced Accuracy: {bal_acc:.5f}")
    
    # Extract best epoch metrics
    best_record = next(r for r in history if r["epoch"] == best_epoch)
    
    val_summary = {
        "candidate": candidate_name,
        "threshold": float(threshold),
        "best_epoch": int(best_epoch),
        "validation_auprc": float(round(best_record["validation_auprc"], 5)),
        "validation_auroc": float(round(best_record["validation_auroc"], 5)),
        "validation_sensitivity": float(round(best_record["validation_sensitivity"], 5)),
        "validation_specificity": float(round(best_record["validation_specificity"], 5)),
        "validation_precision": float(round(best_record["validation_precision"], 5)),
        "validation_recall": float(round(best_record["validation_recall"], 5)),
        "validation_f1": float(round(best_record["validation_f1"], 5)),
        "validation_accuracy": float(round(best_record["validation_accuracy"], 5)),
        "validation_balanced_accuracy": float(bal_acc),
        "validation_event_sensitivity": float(round(event_sens, 4)),
        "validation_event_detected": int(det_events),
        "validation_event_total": int(tot_events),
        "validation_false_alarms_per_day": float(val_fa_per_day),
        "validation_detection_delay_sec": float(mean_delay),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        },
        "training_duration_sec": float(round(time.time() - start_time, 1)),
        "history": history
    }
    
    with open(metrics_json_path, "w") as f:
        json.dump(val_summary, f, indent=2)
    print(f"Saved validation metrics to {metrics_json_path}")
    
    # Save best probabilities for figure generation
    prob_path = os.path.join(candidate_dir, "best_val_predictions.npz")
    np.savez_compressed(prob_path, y_true=best_val_true, y_prob=best_val_prob)
    print(f"Saved validation predictions to {prob_path}")
    
    completed_steps.append("Validation Event & Latency Evaluation (PASS)")
    update_live_status(status_file, "COMPLETE", completed_steps, "Finished", [], start_time, NUM_EPOCHS, NUM_EPOCHS)
    
    return val_summary


def evaluate_frozen_phase4a_reference(
    val_df: pd.DataFrame,
    events_df: pd.DataFrame,
    pipeline: CHBMITDataPipeline,
    device: torch.device
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("EVALUATING FROZEN PHASE 4A REFERENCE (theta = 0.35) ON VALIDATION SET")
    print("=" * 80)
    
    adj_path = os.path.join(PHASE4A_EXP_DIR, "graph_adjacency.csv")
    adj_df = pd.read_csv(adj_path, index_col=0)
    adj_matrix = adj_df.values.astype(np.float32)
    
    model = Baseline1DCNN_GNN(in_channels=23, num_classes=1, adj_matrix=adj_matrix).to(device)
    checkpoint = torch.load(PHASE4A_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded Phase 4A checkpoint: Epoch {checkpoint['epoch']} (Logged Val AUPRC: {checkpoint['val_auprc']:.5f})")
    
    print(f"Running validation inference on 293,410 windows...")
    val_true, val_prob = pipeline.evaluate_split(
        model=model,
        split_df=val_df,
        device=device,
        label_column=LABEL_COLUMN,
        batch_size=256
    )
    
    val_metrics = SeizureEvaluationMetrics.compute_window_metrics(
        y_true=val_true,
        y_prob=val_prob,
        threshold=0.5,
        stride_sec=2.5
    )
    
    val_auprc = float(val_metrics["auprc"]) if val_metrics["auprc"] is not None else 0.0
    val_auroc = float(val_metrics["auroc"]) if val_metrics["auroc"] is not None else 0.0
    val_sens = float(val_metrics["sensitivity"])
    val_spec = float(val_metrics["specificity"])
    val_prec = float(val_metrics["precision"])
    val_rec = val_sens
    val_f1 = float(val_metrics["f1_score"])
    val_acc = float(val_metrics["accuracy"])
    
    mean_delay, det_events, tot_events, event_details = evaluate_event_latency(
        window_df=val_df,
        events_df=events_df,
        y_prob=val_prob,
        threshold=0.5
    )
    event_sens = det_events / tot_events if tot_events > 0 else 0.0
    
    y_pred_binary = (val_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(val_true, y_pred_binary, labels=[0, 1]).ravel()
    val_fa_per_day = float(round((fp / TOTAL_VAL_HOURS) * 24.0, 2))
    bal_acc = float(round(0.5 * ((tp / (tp + fn)) + (tn / (tn + fp))), 5))
    
    print(f"Frozen Phase 4A (theta=0.35) Validation Results:")
    print(f"  Val AUPRC:            {val_auprc:.5f}")
    print(f"  Val AUROC:            {val_auroc:.5f}")
    print(f"  Val Sensitivity:      {val_sens*100:.2f}%")
    print(f"  Val Specificity:      {val_spec*100:.2f}%")
    print(f"  Val Precision:        {val_prec:.5f}")
    print(f"  Val F1:               {val_f1:.5f}")
    print(f"  Val Balanced Acc:     {bal_acc:.5f}")
    print(f"  Val Event Sens:       {det_events}/{tot_events} ({event_sens*100:.2f}%)")
    print(f"  Val False Alarms/Day: {val_fa_per_day} FA/24h")
    print(f"  Val Mean Delay:       {mean_delay}s")
    
    ref_summary = {
        "candidate": "Frozen Phase 4A (theta=0.35)",
        "threshold": 0.35,
        "best_epoch": int(checkpoint["epoch"]),
        "validation_auprc": float(round(val_auprc, 5)),
        "validation_auroc": float(round(val_auroc, 5)),
        "validation_sensitivity": float(round(val_sens, 5)),
        "validation_specificity": float(round(val_spec, 5)),
        "validation_precision": float(round(val_prec, 5)),
        "validation_recall": float(round(val_rec, 5)),
        "validation_f1": float(round(val_f1, 5)),
        "validation_accuracy": float(round(val_acc, 5)),
        "validation_balanced_accuracy": float(bal_acc),
        "validation_event_sensitivity": float(round(event_sens, 4)),
        "validation_event_detected": int(det_events),
        "validation_event_total": int(tot_events),
        "validation_false_alarms_per_day": float(val_fa_per_day),
        "validation_detection_delay_sec": float(mean_delay),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }
    }
    
    ref_json_path = os.path.join(PHASE4A_C_DIR, "reference_theta035_val_metrics.json")
    with open(ref_json_path, "w") as f:
        json.dump(ref_summary, f, indent=2)
        
    ref_prob_path = os.path.join(PHASE4A_C_DIR, "reference_theta035_val_predictions.npz")
    np.savez_compressed(ref_prob_path, y_true=val_true, y_prob=val_prob)
    
    return ref_summary


def run_experiment():
    print("=" * 80)
    print("NEUROAEGIS PHASE 4A-C: GRAPH THRESHOLD CORRECTION EXPERIMENT")
    print("=" * 80)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[Hardware] Apple Silicon MPS acceleration enabled: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Hardware] CUDA acceleration enabled: {device}")
    else:
        device = torch.device("cpu")
        print(f"[Hardware] Running on CPU: {device}")
        
    # Split verification
    print("\n[Split Verification]")
    splitter = PatientDataSplitter(
        window_index_path=WINDOW_INDEX_PATH,
        seizure_events_path=EVENTS_PATH,
        label_column=LABEL_COLUMN
    )
    train_df, val_df, test_df = splitter.get_splits()
    events_df = pd.read_csv(EVENTS_PATH)
    
    # Assert zero leakage
    assert len(set(splitter.train_patients) & set(splitter.val_patients)) == 0, "Patient leakage!"
    assert len(set(splitter.train_patients) & set(splitter.test_patients)) == 0, "Patient leakage!"
    assert len(set(splitter.val_patients) & set(splitter.test_patients)) == 0, "Patient leakage!"
    print("Patient, recording, and window partitions verified: ZERO LEAKAGE (PASS)")
    
    pipeline = CHBMITDataPipeline(edf_root_dir=EDF_ROOT_DIR)
    sampler = DynamicNegativeSampler(
        train_df=train_df,
        ratio=SAMPLING_RATIO,
        base_seed=BASE_SEED,
        label_column=LABEL_COLUMN,
        shuffle=True
    )
    
    # Pre-cache positive training windows (shared across both candidates)
    print("\n[Data Pipeline] Pre-caching 3,308 positive training windows...")
    t_pos0 = time.time()
    X_pos, y_pos = pipeline.load_epoch_windows(
        df=train_df,
        sampled_indices=sampler.pos_indices,
        label_column=LABEL_COLUMN,
        shuffle=False
    )
    print(f"Pre-cached {len(X_pos):,} positive windows in {time.time() - t_pos0:.1f}s")
    
    # 1. Train Model 1: theta = 0.25
    summary_025 = train_single_candidate(
        candidate_name="Graph A (theta=0.25)",
        threshold=0.25,
        candidate_dir=os.path.join(PHASE4A_C_DIR, "theta_025"),
        train_df=train_df,
        val_df=val_df,
        events_df=events_df,
        pipeline=pipeline,
        sampler=sampler,
        X_pos=X_pos,
        y_pos=y_pos,
        device=device
    )
    
    # 2. Train Model 2: theta = 0.30
    summary_030 = train_single_candidate(
        candidate_name="Graph B (theta=0.30)",
        threshold=0.30,
        candidate_dir=os.path.join(PHASE4A_C_DIR, "theta_030"),
        train_df=train_df,
        val_df=val_df,
        events_df=events_df,
        pipeline=pipeline,
        sampler=sampler,
        X_pos=X_pos,
        y_pos=y_pos,
        device=device
    )
    
    # 3. Evaluate Frozen Reference: theta = 0.35
    summary_035 = evaluate_frozen_phase4a_reference(
        val_df=val_df,
        events_df=events_df,
        pipeline=pipeline,
        device=device
    )
    
    # 4. Generate validation_threshold_comparison.csv
    print("\n" + "=" * 80)
    print("COMPILATION OF VALIDATION THRESHOLD COMPARISON")
    print("=" * 80)
    
    all_summaries = [summary_025, summary_030, summary_035]
    comp_rows = []
    for s in all_summaries:
        comp_rows.append({
            "candidate": s["candidate"],
            "threshold": s["threshold"],
            "best_epoch": s["best_epoch"],
            "validation_auprc": s["validation_auprc"],
            "validation_auroc": s["validation_auroc"],
            "validation_sensitivity": s["validation_sensitivity"],
            "validation_specificity": s["validation_specificity"],
            "validation_precision": s["validation_precision"],
            "validation_recall": s["validation_recall"],
            "validation_f1": s["validation_f1"],
            "validation_accuracy": s["validation_accuracy"],
            "validation_balanced_accuracy": s["validation_balanced_accuracy"],
            "validation_event_sensitivity": s["validation_event_sensitivity"],
            "validation_event_detected": s["validation_event_detected"],
            "validation_event_total": s["validation_event_total"],
            "validation_false_alarms_per_day": s["validation_false_alarms_per_day"],
            "validation_detection_delay_sec": s["validation_detection_delay_sec"],
            "tp": s["confusion_matrix"]["tp"],
            "fp": s["confusion_matrix"]["fp"],
            "tn": s["confusion_matrix"]["tn"],
            "fn": s["confusion_matrix"]["fn"]
        })
        
    df_val_comp = pd.DataFrame(comp_rows)
    val_csv_path = os.path.join(PHASE4A_C_DIR, "validation_threshold_comparison.csv")
    df_val_comp.to_csv(val_csv_path, index=False)
    print(f"Saved validation comparison CSV to {val_csv_path}")
    print("\n" + df_val_comp[["candidate", "threshold", "best_epoch", "validation_auprc", "validation_auroc", "validation_event_sensitivity", "validation_false_alarms_per_day"]].to_string(index=False))

if __name__ == "__main__":
    run_experiment()
