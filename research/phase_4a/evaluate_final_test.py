"""
NeuroAegis Phase 4A-C: Single Final Test Evaluation Engine
Evaluates the frozen CNN + GNN model (theta = 0.30) on the untouched CHB-MIT test set.
Computes:
  - Window-level metrics (accuracy, precision, recall, specificity, F1, balanced acc, AUROC, AUPRC)
  - Seizure event-level latency and sensitivity (22 events across 4 test patients)
  - False alarm rate per 24 hours (152.82 recording hours)
  - Patient-level performance breakdowns
Exports:
  - research/phase_4a/final_test_metrics.json
  - research/phase_4a/final_test_predictions.npz
  - research/phase_4a/final_test_event_details.csv
  - research/phase_4a/final_test_patient_metrics.csv
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from research.phase_4a.cnn_gnn_model import Baseline1DCNN_GNN
from research.phase_3.data_loader import CHBMITDataPipeline
from research.imbalance.patient_splitter import PatientDataSplitter
from research.imbalance.focal_loss import logits_to_probabilities
from research.imbalance.metrics import SeizureEvaluationMetrics

MANIFEST_DIR = os.path.join(BASE_DIR, "research/data/manifests")
WINDOW_INDEX_PATH = os.path.join(MANIFEST_DIR, "chbmit_window_index.csv")
EVENTS_PATH = os.path.join(MANIFEST_DIR, "chbmit_seizure_events.csv")
EDF_ROOT_DIR = os.path.join(BASE_DIR, "CHB-MIT Dataset")

FROZEN_ADJ_PATH = os.path.join(BASE_DIR, "research/phase_4a/frozen_graph_adjacency.csv")
FROZEN_CHECKPOINT_PATH = os.path.join(BASE_DIR, "research/phase_4a/frozen_cnn_gnn.pt")
OUTPUT_METRICS_PATH = os.path.join(BASE_DIR, "research/phase_4a/final_test_metrics.json")
OUTPUT_PREDS_PATH = os.path.join(BASE_DIR, "research/phase_4a/final_test_predictions.npz")
OUTPUT_EVENTS_CSV = os.path.join(BASE_DIR, "research/phase_4a/final_test_event_details.csv")
OUTPUT_PATIENTS_CSV = os.path.join(BASE_DIR, "research/phase_4a/final_test_patient_metrics.csv")

LABEL_COLUMN = "label_50pct_overlap"


def run_final_test_evaluation():
    print("=" * 80)
    print("NEUROAEGIS PHASE 4A-C: SINGLE FINAL TEST EVALUATION (EXACTLY ONCE)")
    print("Model Configuration: Frozen Baseline1DCNN_GNN (theta = 0.30)")
    print("=" * 80)
    
    start_time = time.time()
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Hardware Acceleration]: {device}")
    
    # 1. Dataset Sanity & Leakage Checks
    print("\n[Step 1/5] Verifying test dataset partitions & leakage invariants...")
    splitter = PatientDataSplitter(
        window_index_path=WINDOW_INDEX_PATH,
        seizure_events_path=EVENTS_PATH,
        label_column=LABEL_COLUMN
    )
    train_df, val_df, test_df = splitter.get_splits()
    events_df = pd.read_csv(EVENTS_PATH)
    
    # Sanity checks
    n_test_windows = len(test_df)
    n_pos_windows = int(test_df[LABEL_COLUMN].sum())
    n_neg_windows = n_test_windows - n_pos_windows
    test_patients = sorted(list(test_df["patient_id"].unique()))
    test_recordings = sorted(list(test_df["recording_id"].unique()))
    
    sub_events = events_df[events_df["recording_id"].isin(test_recordings)]
    n_events = len(sub_events)
    test_duration_hours = n_test_windows * 2.5 / 3600.0
    
    print(f"  Test Patients ({len(test_patients)}):   {test_patients}")
    print(f"  Test Recordings ({len(test_recordings)}): {len(test_recordings)} EDFs")
    print(f"  Test Windows:       {n_test_windows:,} (Positives: {n_pos_windows:,}, Negatives: {n_neg_windows:,})")
    print(f"  Test Duration:      {test_duration_hours:.2f} hours")
    print(f"  Test Seizures:      {n_events} annotated events")
    
    assert len(set(splitter.train_patients) & set(splitter.test_patients)) == 0, "Patient leakage!"
    assert len(set(splitter.val_patients) & set(splitter.test_patients)) == 0, "Patient leakage!"
    assert len(set(train_df["recording_id"]) & set(test_df["recording_id"])) == 0, "Recording leakage!"
    assert len(set(train_df["window_id"]) & set(test_df["window_id"])) == 0, "Window leakage!"
    assert n_test_windows == 219909, f"Expected 219,909 test windows, found {n_test_windows}"
    assert n_events == 22, f"Expected 22 test events, found {n_events}"
    print("  -> Dataset sanity check & zero leakage: PASS")
    
    # 2. Model Initialization
    print("\n[Step 2/5] Loading frozen model checkpoint and spatial graph...")
    adj_df = pd.read_csv(FROZEN_ADJ_PATH, index_col=0)
    adj_matrix = adj_df.values.astype(np.float32)
    
    model = Baseline1DCNN_GNN(in_channels=23, num_classes=1, adj_matrix=adj_matrix).to(device)
    checkpoint = torch.load(FROZEN_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded frozen checkpoint from Epoch {checkpoint['epoch']} (Val AUPRC: {checkpoint['val_auprc']:.5f})")
    
    # 3. Test Inference
    print("\n[Step 3/5] Running single-pass inference on 219,909 test windows...")
    t_inf0 = time.time()
    pipeline = CHBMITDataPipeline(edf_root_dir=EDF_ROOT_DIR)
    
    test_true, test_prob = pipeline.evaluate_split(
        model=model,
        split_df=test_df,
        device=device,
        label_column=LABEL_COLUMN,
        batch_size=256
    )
    t_inf1 = time.time()
    print(f"  Completed test inference in {t_inf1 - t_inf0:.1f} seconds")
    
    # Save test predictions
    np.savez_compressed(OUTPUT_PREDS_PATH, y_true=test_true, y_prob=test_prob)
    print(f"  Saved raw test predictions to {OUTPUT_PREDS_PATH}")
    
    # 4. Metric Computation
    print("\n[Step 4/5] Computing comprehensive window, event, and clinical metrics...")
    y_pred_binary = (test_prob >= 0.50).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(test_true, y_pred_binary, labels=[0, 1]).ravel()
    
    acc = float(accuracy_score(test_true, y_pred_binary))
    prec = float(precision_score(test_true, y_pred_binary, zero_division=0))
    rec = float(recall_score(test_true, y_pred_binary, zero_division=0))
    sens = rec
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(test_true, y_pred_binary, zero_division=0))
    bal_acc = float(0.5 * (sens + spec))
    
    auroc = float(roc_auc_score(test_true, test_prob))
    auprc = float(average_precision_score(test_true, test_prob))
    
    # False alarms per 24 hours
    fa_count = int(fp)
    # Total non-seizure recording hours
    non_sz_hours = (n_neg_windows * 2.5) / 3600.0
    fa_per_24h = float(round((fa_count / non_sz_hours) * 24.0, 2))
    
    # Event-level evaluation
    test_df_eval = test_df.copy()
    test_df_eval["pred_prob"] = test_prob
    test_df_eval["pred_label"] = y_pred_binary
    
    delays = []
    event_details = []
    
    for _, ev in sub_events.iterrows():
        rec_id = ev["recording_id"]
        sz_id = ev["seizure_id"]
        pat_id = ev["patient_id"]
        s_start = ev["start_sec"]
        s_end = ev["end_sec"]
        s_dur = ev["duration_sec"]
        
        rec_w = test_df_eval[test_df_eval["recording_id"] == rec_id]
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
            "true_onset_sec": s_start,
            "true_end_sec": s_end,
            "duration_sec": s_dur,
            "detected": bool(is_detected),
            "first_detection_time_sec": first_alarm,
            "detection_delay_sec": delay,
            "positive_windows_count": len(det_w),
            "overlapping_windows_count": len(ov_w)
        })
        
    df_events_out = pd.DataFrame(event_details)
    df_events_out.to_csv(OUTPUT_EVENTS_CSV, index=False)
    print(f"  Saved event details table to {OUTPUT_EVENTS_CSV}")
    
    n_detected_events = len(delays)
    n_missed_events = n_events - n_detected_events
    event_sens = float(n_detected_events / n_events) if n_events > 0 else 0.0
    
    mean_delay = float(np.mean(delays)) if delays else float("nan")
    median_delay = float(np.median(delays)) if delays else float("nan")
    min_delay = float(np.min(delays)) if delays else float("nan")
    max_delay = float(np.max(delays)) if delays else float("nan")
    
    # Patient-level breakdown
    pat_records = []
    for pat_id in test_patients:
        pat_mask = (test_df["patient_id"] == pat_id).values
        pat_true = test_true[pat_mask]
        pat_prob = test_prob[pat_mask]
        pat_pred = y_pred_binary[pat_mask]
        
        p_tn, p_fp, p_fn, p_tp = confusion_matrix(pat_true, pat_pred, labels=[0, 1]).ravel()
        p_sens = float(p_tp / (p_tp + p_fn)) if (p_tp + p_fn) > 0 else 0.0
        p_spec = float(p_tn / (p_tn + p_fp)) if (p_tn + p_fp) > 0 else 0.0
        p_hours = len(pat_true) * 2.5 / 3600.0
        p_fa_rate = float(round((p_fp / p_hours) * 24.0, 2)) if p_hours > 0 else 0.0
        
        pat_evs = df_events_out[df_events_out["patient_id"] == pat_id]
        p_tot_ev = len(pat_evs)
        p_det_ev = int(pat_evs["detected"].sum())
        p_ev_sens = float(p_det_ev / p_tot_ev) if p_tot_ev > 0 else 0.0
        p_delays = pat_evs[pat_evs["detected"]]["detection_delay_sec"].dropna().values
        p_mean_delay = float(np.mean(p_delays)) if len(p_delays) > 0 else float("nan")
        
        pat_records.append({
            "patient_id": pat_id,
            "num_recordings": int(test_df[test_df["patient_id"] == pat_id]["recording_id"].nunique()),
            "total_windows": len(pat_true),
            "recording_hours": round(p_hours, 2),
            "num_seizures": p_tot_ev,
            "detected_seizures": p_det_ev,
            "event_sensitivity": round(p_ev_sens, 4),
            "window_sensitivity": round(p_sens, 5),
            "window_specificity": round(p_spec, 5),
            "false_alarms_count": int(p_fp),
            "false_alarms_per_day": p_fa_rate,
            "mean_detection_delay_sec": round(p_mean_delay, 2) if not np.isnan(p_mean_delay) else None
        })
        
    df_pat_out = pd.DataFrame(pat_records)
    df_pat_out.to_csv(OUTPUT_PATIENTS_CSV, index=False)
    print(f"  Saved patient-level metrics to {OUTPUT_PATIENTS_CSV}")
    
    # 5. Output Summary Metrics JSON
    test_metrics = {
        "phase": "4A-C",
        "model": "Baseline1DCNN_GNN (Frozen)",
        "graph_threshold": 0.30,
        "test_dataset": "CHB-MIT",
        "test_patients": test_patients,
        "test_windows_total": n_test_windows,
        "test_duration_hours": round(test_duration_hours, 2),
        "test_accuracy": round(acc, 5),
        "test_precision": round(prec, 5),
        "test_sensitivity": round(sens, 5),
        "test_specificity": round(spec, 5),
        "test_f1": round(f1, 5),
        "test_balanced_accuracy": round(bal_acc, 5),
        "test_auroc": round(auroc, 5),
        "test_auprc": round(auprc, 5),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        },
        "event_metrics": {
            "total_seizure_events": n_events,
            "detected_seizure_events": n_detected_events,
            "missed_seizure_events": n_missed_events,
            "event_sensitivity": round(event_sens, 4),
            "mean_detection_delay_sec": round(mean_delay, 2) if not np.isnan(mean_delay) else None,
            "median_detection_delay_sec": round(median_delay, 2) if not np.isnan(median_delay) else None,
            "min_detection_delay_sec": round(min_delay, 2) if not np.isnan(min_delay) else None,
            "max_detection_delay_sec": round(max_delay, 2) if not np.isnan(max_delay) else None
        },
        "false_alarm_metrics": {
            "false_alarm_count": fa_count,
            "non_seizure_recording_hours": round(non_sz_hours, 2),
            "false_alarms_per_24h": fa_per_24h
        },
        "patient_metrics": pat_records,
        "evaluation_duration_sec": round(time.time() - start_time, 1)
    }
    
    with open(OUTPUT_METRICS_PATH, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"  Saved final test metrics JSON to {OUTPUT_METRICS_PATH}")
    
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Accuracy:            {acc*100:.2f}%")
    print(f"Precision:           {prec:.5f}")
    print(f"Sensitivity (Recall):{sens*100:.2f}%")
    print(f"Specificity:         {spec*100:.2f}%")
    print(f"F1 Score:            {f1:.5f}")
    print(f"Balanced Accuracy:   {bal_acc:.5f}")
    print(f"AUROC:               {auroc:.5f}")
    print(f"AUPRC:               {auprc:.5f}")
    print(f"Event Sensitivity:   {n_detected_events}/{n_events} ({event_sens*100:.2f}%)")
    print(f"False Alarms/24h:    {fa_per_24h} FA/24h (FP={fa_count:,})")
    print(f"Mean Delay:          {mean_delay:.2f}s (Median: {median_delay:.2f}s)")
    print("=" * 80)

if __name__ == "__main__":
    run_final_test_evaluation()
