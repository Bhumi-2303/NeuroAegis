"""
NeuroAegis Phase 3: CNN Baseline Training & Evaluation Engine
Trains the 1D CNN baseline on CHB-MIT EEG under frozen Decision 2:
- Dynamic Negative Subsampling (10:1 ratio, seed = base_seed + epoch, base_seed = 42)
- Binary Focal Loss (gamma = 2.0, alpha = 0.25)
- Primary label: label_50pct_overlap (Strategy B >= 50% overlap)
- Strict patient isolation (16 Train / 4 Val / 4 Test)
- Validation-driven checkpoint selection (Peak Validation AUPRC)
- Untouched single-pass Test evaluation (219,909 windows, 152.82 hours)
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

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "research/phase_3"))

from research.phase_3.cnn_model import Baseline1DCNN
from research.phase_3.data_loader import CHBMITDataPipeline
from research.imbalance.patient_splitter import PatientDataSplitter
from research.imbalance.dynamic_sampler import DynamicNegativeSampler
from research.imbalance.focal_loss import BinaryFocalLossWithLogits, logits_to_probabilities
from research.imbalance.metrics import SeizureEvaluationMetrics

WINDOW_INDEX_PATH = os.path.join(BASE_DIR, "research/data/manifests/chbmit_window_index.csv")
EVENTS_PATH = os.path.join(BASE_DIR, "research/data/manifests/chbmit_seizure_events.csv")
MANIFEST_PATH = os.path.join(BASE_DIR, "research/data/manifests/chbmit_manifest.csv")
EDF_ROOT_DIR = os.path.join(BASE_DIR, "CHB-MIT Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "research/phase_3")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_cnn_baseline.pt")
METRICS_JSON_PATH = os.path.join(OUTPUT_DIR, "phase_3_metrics.json")
LIVE_STATUS_PATH = os.path.join(OUTPUT_DIR, "live_status.json")

LABEL_COLUMN = "label_50pct_overlap"
SAMPLING_RATIO = 10.0
BASE_SEED = 42
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 3


def update_live_status(
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
        "completed": completed_steps,
        "in_progress": in_progress,
        "queued": queued_steps,
        "extra_info": extra_info or {}
    }
    with open(LIVE_STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)


def get_file_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def compute_detection_delay(
    window_df: pd.DataFrame,
    events_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, int, int]:
    df = window_df.copy()
    df["pred_prob"] = y_prob
    df["pred_label"] = (df["pred_prob"] >= threshold).astype(int)
    
    delays = []
    test_recs = set(window_df["recording_id"].unique())
    sub_events = events_df[events_df["recording_id"].isin(test_recs)]
    total_events = len(sub_events)
    
    for _, ev in sub_events.iterrows():
        rec_id = ev["recording_id"]
        s_start = ev["start_sec"]
        s_end = ev["end_sec"]
        
        rec_w = df[df["recording_id"] == rec_id]
        ov_mask = (rec_w["window_end_sec"] > s_start) & (rec_w["window_start_sec"] < s_end)
        ov_w = rec_w[ov_mask]
        
        det_w = ov_w[ov_w["pred_label"] == 1]
        if len(det_w) > 0:
            first_alarm = det_w["window_end_sec"].min()
            delay = max(0.0, float(first_alarm - s_start))
            delays.append(delay)
            
    detected_count = len(delays)
    mean_delay = float(np.mean(delays)) if delays else float("nan")
    return round(mean_delay, 2), detected_count, total_events


def run_phase3_training():
    print("=" * 80)
    print("NEUROAEGIS PHASE 3: CNN BASELINE TRAINING & EVALUATION")
    print("=" * 80)
    start_time = time.time()
    
    completed_steps = []
    queued_steps = [
        "Partition Verification & Leakage Checks",
        "Pre-caching Positive Training Windows",
        "Training & Validation Loop (3 Epochs)",
        "Final Untouched Test Evaluation (219,909 windows)",
        "Clinical Metrics & Latency Compilation"
    ]
    
    update_live_status("INITIALIZATION", completed_steps, "Hardware & Data Partition Setup", queued_steps, start_time)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[Hardware] Apple Silicon MPS acceleration enabled: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Hardware] CUDA acceleration enabled: {device}")
    else:
        device = torch.device("cpu")
        print(f"[Hardware] Running on CPU: {device}")
        
    print("\n[Data] Loading and verifying partition boundaries...")
    index_sha_before = get_file_sha256(WINDOW_INDEX_PATH)
    print(f"Master window index SHA256: {index_sha_before}")
    
    splitter = PatientDataSplitter(
        window_index_path=WINDOW_INDEX_PATH,
        seizure_events_path=EVENTS_PATH,
        label_column=LABEL_COLUMN
    )
    
    train_df, val_df, test_df = splitter.get_splits()
    events_df = pd.read_csv(EVENTS_PATH)
    
    p_leak = (len(set(splitter.train_patients) & set(splitter.val_patients)) == 0 and
              len(set(splitter.train_patients) & set(splitter.test_patients)) == 0 and
              len(set(splitter.val_patients) & set(splitter.test_patients)) == 0)
    assert p_leak, "FATAL: Patient leakage detected!"
    
    r_leak = (len(set(train_df["recording_id"]) & set(val_df["recording_id"])) == 0 and
              len(set(train_df["recording_id"]) & set(test_df["recording_id"])) == 0 and
              len(set(val_df["recording_id"]) & set(test_df["recording_id"])) == 0)
    assert r_leak, "FATAL: Recording leakage detected!"
    
    w_leak = (len(set(train_df["window_id"]) & set(val_df["window_id"])) == 0 and
              len(set(train_df["window_id"]) & set(test_df["window_id"])) == 0 and
              len(set(val_df["window_id"]) & set(test_df["window_id"])) == 0)
    assert w_leak, "FATAL: Window leakage detected!"
    
    completed_steps.append("Partition Verification & Leakage Checks (PASS)")
    queued_steps.remove("Partition Verification & Leakage Checks")
    
    print("\n[Model Initialization]")
    torch.manual_seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    
    model = Baseline1DCNN(in_channels=23, num_classes=1).to(device)
    num_params = model.get_num_parameters()
    print(f"Architecture:       1D CNN (Temporal Multi-Scale Conv1D)")
    print(f"Trainable Params:   {num_params:,}")
    
    criterion = BinaryFocalLossWithLogits(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    
    pipeline = CHBMITDataPipeline(edf_root_dir=EDF_ROOT_DIR)
    sampler = DynamicNegativeSampler(
        train_df=train_df,
        ratio=SAMPLING_RATIO,
        base_seed=BASE_SEED,
        label_column=LABEL_COLUMN,
        shuffle=True
    )
    
    print("\n[Data Pipeline] Pre-caching positive training windows...")
    update_live_status("PRE_CACHING", completed_steps, "Pre-caching 3,308 Positive Training Windows", queued_steps, start_time)
    
    t_pos0 = time.time()
    pos_indices = sampler.pos_indices
    X_pos, y_pos = pipeline.load_epoch_windows(
        df=train_df,
        sampled_indices=pos_indices,
        label_column=LABEL_COLUMN,
        shuffle=False,
        progress_callback=lambda cur, tot: print(f"  Pre-caching positives: {cur}/{tot} recordings processed...", flush=True)
    )
    t_pos1 = time.time()
    print(f"Pre-cached {len(X_pos):,} positive training windows in {t_pos1 - t_pos0:.2f}s ({X_pos.element_size() * X_pos.nelement() / 1024 / 1024:.1f} MB)")
    
    completed_steps.append("Pre-caching Positive Training Windows (3,308 windows cached)")
    queued_steps.remove("Pre-caching Positive Training Windows")
    
    print("\n" + "=" * 80)
    print(f"STARTING TRAINING ({NUM_EPOCHS} Epochs, 10:1 Dynamic Negative Sampling)")
    print("=" * 80)
    
    best_epoch = -1
    best_val_auprc = -1.0
    history = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        sampler.set_epoch(epoch)
        
        in_prog_str = f"Epoch {epoch+1}/{NUM_EPOCHS}: Loading 33,080 Dynamic Negative Windows"
        update_live_status("TRAINING", completed_steps, in_prog_str, queued_steps, start_time, epoch+1, NUM_EPOCHS)
        
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        print(f"Loading 33,080 sampled negative windows (seed={42+epoch})...")
        t_load0 = time.time()
        neg_indices = sampler.current_sampled_neg_indices
        X_neg, y_neg = pipeline.load_epoch_windows(
            df=train_df,
            sampled_indices=neg_indices,
            label_column=LABEL_COLUMN,
            shuffle=False,
            progress_callback=lambda cur, tot: print(f"  Sampling negatives: {cur}/{tot} recordings...", flush=True)
        )
        t_load1 = time.time()
        print(f"Loaded {len(X_neg):,} negatives in {t_load1 - t_load0:.1f}s")
        
        in_prog_str = f"Epoch {epoch+1}/{NUM_EPOCHS}: Training 1D CNN on 36,388 Windows"
        update_live_status("TRAINING", completed_steps, in_prog_str, queued_steps, start_time, epoch+1, NUM_EPOCHS)
        
        X_epoch = torch.cat([X_pos, X_neg], dim=0)
        y_epoch = torch.cat([y_pos, y_neg], dim=0)
        
        perm = torch.randperm(len(X_epoch))
        X_epoch = X_epoch[perm]
        y_epoch = y_epoch[perm]
        
        dataset = TensorDataset(X_epoch, y_epoch)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        
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
            
        scheduler.step()
        avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        t_train1 = time.time()
        print(f"Epoch {epoch+1} Training: Loss = {avg_train_loss:.5f} ({t_train1 - t_train0:.1f}s)")
        
        in_prog_str = f"Epoch {epoch+1}/{NUM_EPOCHS}: Full Validation Evaluation (293,410 Windows, 82 Recordings)"
        update_live_status("VALIDATION", completed_steps, in_prog_str, queued_steps, start_time, epoch+1, NUM_EPOCHS)
        
        print(f"Evaluating Validation set (293,410 windows across 82 recordings)...")
        t_val0 = time.time()
        val_true, val_prob = pipeline.evaluate_split(
            model=model,
            split_df=val_df,
            device=device,
            label_column=LABEL_COLUMN,
            batch_size=256,
            progress_callback=lambda cur, tot, r: print(f"  Validation progress: {cur}/{tot} recordings ({r})...", flush=True)
        )
        t_val1 = time.time()
        
        val_metrics = SeizureEvaluationMetrics.compute_window_metrics(
            y_true=val_true,
            y_prob=val_prob,
            threshold=0.5,
            stride_sec=2.5
        )
        
        val_auprc = val_metrics["auprc"] if val_metrics["auprc"] is not None else 0.0
        val_auroc = val_metrics["auroc"] if val_metrics["auroc"] is not None else 0.0
        val_sens = val_metrics["sensitivity"]
        val_spec = val_metrics["specificity"]
        val_f1 = val_metrics["f1_score"]
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS:02d} Complete [{epoch_time:.1f}s] "
              f"Train Loss: {avg_train_loss:.5f} | "
              f"Val AUPRC: {val_auprc:.5f} | "
              f"Val AUROC: {val_auroc:.5f} | "
              f"Val Sens: {val_sens:.4f} | "
              f"Val Spec: {val_spec:.4f} | "
              f"Val F1: {val_f1:.4f}")
              
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 5),
            "val_auprc": round(val_auprc, 5),
            "val_auroc": round(val_auroc, 5),
            "val_sensitivity": round(val_sens, 5),
            "val_specificity": round(val_spec, 5),
            "val_f1": round(val_f1, 5),
            "epoch_duration_sec": round(epoch_time, 2)
        })
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_epoch = epoch + 1
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auprc": best_val_auprc,
                "val_metrics": val_metrics
            }, CHECKPOINT_PATH)
            print(f"  -> Best model checkpoint saved at epoch {best_epoch} with Val AUPRC = {best_val_auprc:.5f}")
            
        completed_steps.append(f"Epoch {epoch+1}/{NUM_EPOCHS} (Loss={avg_train_loss:.4f}, Val AUPRC={val_auprc:.4f})")
        
    training_duration = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE in {training_duration:.2f}s ({training_duration/60:.2f} min)")
    print(f"Best Epoch: {best_epoch} with Validation AUPRC: {best_val_auprc:.5f}")
    print("=" * 80)
    
    queued_steps.remove("Training & Validation Loop (3 Epochs)")
    
    print("\n[Test Evaluation] Evaluating best model on untouched Test partition (219,909 windows, 155 recordings)...")
    in_prog_str = "Final Untouched Test Evaluation (219,909 windows, 155 recordings, 152.82 hours)"
    update_live_status("TEST_EVALUATION", completed_steps, in_prog_str, queued_steps, start_time)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    t_test0 = time.time()
    test_true, test_prob = pipeline.evaluate_split(
        model=model,
        split_df=test_df,
        device=device,
        label_column=LABEL_COLUMN,
        batch_size=256,
        progress_callback=lambda cur, tot, r: print(f"  Test progress: {cur}/{tot} recordings ({r})...", flush=True)
    )
    t_test1 = time.time()
    print(f"Test evaluation completed in {t_test1 - t_test0:.1f}s")
    
    completed_steps.append("Final Untouched Test Evaluation (219,909 windows)")
    queued_steps.remove("Final Untouched Test Evaluation (219,909 windows)")
    
    in_prog_str = "Clinical Metrics & Latency Compilation"
    update_live_status("METRIC_COMPILATION", completed_steps, in_prog_str, queued_steps, start_time)
    
    test_manifest = pd.read_csv(MANIFEST_PATH)
    test_dur_hours = test_manifest[test_manifest["patient_id"].isin(splitter.test_patients)]["recording_duration_sec"].sum() / 3600.0
    
    test_metrics = SeizureEvaluationMetrics.compute_window_metrics(
        y_true=test_true,
        y_prob=test_prob,
        threshold=0.5,
        total_duration_hours=test_dur_hours,
        stride_sec=2.5
    )
    
    event_metrics = SeizureEvaluationMetrics.compute_event_level_sensitivity(
        window_df=test_df,
        y_prob=test_prob,
        threshold=0.5,
        label_column=LABEL_COLUMN
    )
    
    det_delay, det_ev_count, total_ev_count = compute_detection_delay(
        window_df=test_df,
        events_df=events_df,
        y_prob=test_prob,
        threshold=0.5
    )
    
    peak_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 2)
    index_sha_after = get_file_sha256(WINDOW_INDEX_PATH)
    index_match = (index_sha_before == index_sha_after)
    assert index_match, "FATAL: Master window index was modified!"
    
    completed_steps.append("Clinical Metrics & Latency Compilation")
    queued_steps.remove("Clinical Metrics & Latency Compilation")
    
    final_output = {
        "status": "PASS",
        "dataset": "CHB-MIT",
        "channels": 23,
        "window": "5 sec / 1280 samples",
        "stride": "2.5 sec / 640 samples",
        "primary_label": ">=50% overlap",
        "training_sampler": "10:1 dynamic negative sampling",
        "loss": "Binary Focal Loss",
        "gamma": FOCAL_GAMMA,
        "alpha": FOCAL_ALPHA,
        "train_patients": splitter.train_patients,
        "validation_patients": splitter.val_patients,
        "test_patients": splitter.test_patients,
        "patient_leakage": "PASS",
        "recording_leakage": "PASS",
        "window_leakage": "PASS",
        "model": "1D CNN",
        "parameters": num_params,
        "best_epoch": best_epoch,
        "validation_auprc": round(best_val_auprc, 5),
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_sensitivity": test_metrics["sensitivity"],
        "test_specificity": test_metrics["specificity"],
        "test_f1": test_metrics["f1_score"],
        "test_auroc": test_metrics["auroc"],
        "test_auprc": test_metrics["auprc"],
        "event_sensitivity": event_metrics["event_level_sensitivity"],
        "false_alarms_per_day": test_metrics["false_alarms_per_24h"],
        "detection_delay": det_delay,
        "peak_memory_mb": peak_rss_mb,
        "training_time_sec": round(training_duration, 2),
        "training_history": history,
        "master_index_sha256": index_sha_after
    }
    
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(final_output, f, indent=2)
        
    update_live_status("COMPLETE", completed_steps, "Done", [], start_time, NUM_EPOCHS, NUM_EPOCHS, {"final_status": "PASS"})
    print(f"\nSaved metrics to {METRICS_JSON_PATH}")
    print("ALL PHASE 3 TASKS COMPLETED SUCCESSFULLY!")
    return final_output

if __name__ == "__main__":
    run_phase3_training()
