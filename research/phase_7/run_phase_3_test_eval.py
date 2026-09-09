"""
NeuroAegis Phase 7: Evaluate frozen Phase 3 CNN baseline on the untouched Test set.
Saves predictions to research/phase_7/results/phase_3_test_predictions.npz
and computes patient-level and event-level metrics.
"""
import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
sys.path.insert(0, BASE_DIR)

from research.phase_3.cnn_model import Baseline1DCNN
from research.phase_3.data_loader import CHBMITDataPipeline
from research.imbalance.patient_splitter import PatientDataSplitter
from research.imbalance.metrics import SeizureEvaluationMetrics

WINDOW_INDEX_PATH = os.path.join(BASE_DIR, "research/data/manifests/chbmit_window_index.csv")
EVENTS_PATH = os.path.join(BASE_DIR, "research/data/manifests/chbmit_seizure_events.csv")
MANIFEST_PATH = os.path.join(BASE_DIR, "research/data/manifests/chbmit_manifest.csv")
EDF_ROOT_DIR = os.path.join(BASE_DIR, "CHB-MIT Dataset")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "research/phase_3/best_cnn_baseline.pt")
OUTPUT_NPZ = os.path.join(BASE_DIR, "research/phase_7/results/phase_3_test_predictions.npz")
OUTPUT_PATIENT_CSV = os.path.join(BASE_DIR, "research/phase_7/results/phase_3_patient_results.csv")
OUTPUT_EVENT_CSV = os.path.join(BASE_DIR, "research/phase_7/results/phase_3_event_results.csv")

def run():
    t0 = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    splitter = PatientDataSplitter(
        window_index_path=WINDOW_INDEX_PATH,
        seizure_events_path=EVENTS_PATH,
        label_column="label_50pct_overlap"
    )
    train_df, val_df, test_df = splitter.get_splits()
    events_df = pd.read_csv(EVENTS_PATH)
    manifest_df = pd.read_csv(MANIFEST_PATH)

    pipeline = CHBMITDataPipeline(edf_root_dir=EDF_ROOT_DIR)
    model = Baseline1DCNN(in_channels=23, num_classes=1).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Evaluating on {len(test_df)} test windows across {test_df['recording_id'].nunique()} recordings...")
    y_true, y_prob = pipeline.evaluate_split(
        model=model,
        split_df=test_df,
        device=device,
        batch_size=256,
        progress_callback=lambda cur, tot, r: print(f"  [{cur}/{tot}] {r} done...", flush=True) if cur % 25 == 0 or cur == tot else None
    )

    np.savez_compressed(OUTPUT_NPZ, y_true=y_true, y_prob=y_prob, window_ids=test_df["window_id"].values)
    print(f"Saved {OUTPUT_NPZ} ({os.path.getsize(OUTPUT_NPZ)/(1024*1024):.2f} MB)")

    # Compute overall metrics
    test_dur_hours = manifest_df[manifest_df["patient_id"].isin(splitter.test_patients)]["recording_duration_sec"].sum() / 3600.0
    metrics = SeizureEvaluationMetrics.compute_window_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
        total_duration_hours=test_dur_hours,
        stride_sec=2.5
    )
    print("Overall Test Metrics:", metrics)

    # Compute patient-level metrics
    test_df_eval = test_df.copy()
    test_df_eval["y_prob"] = y_prob
    test_df_eval["pred_label"] = (y_prob >= 0.5).astype(int)

    pat_rows = []
    for pat_id in ["chb01", "chb02", "chb03", "chb05"]:
        pdf = test_df_eval[test_df_eval["patient_id"] == pat_id]
        dur_h = manifest_df[manifest_df["patient_id"] == pat_id]["recording_duration_sec"].sum() / 3600.0
        pat_events = events_df[events_df["patient_id"] == pat_id]
        n_events = len(pat_events)
        
        # Event sensitivity and delays
        det_events = 0
        delays = []
        for _, ev in pat_events.iterrows():
            rec_id = ev["recording_id"]
            s_start = ev["start_sec"]
            s_end = ev["end_sec"]
            rw = pdf[(pdf["recording_id"] == rec_id) & (pdf["window_end_sec"] > s_start) & (pdf["window_start_sec"] < s_end)]
            det_w = rw[rw["pred_label"] == 1]
            if len(det_w) > 0:
                det_events += 1
                alarm_time = det_w["window_end_sec"].min()
                delays.append(max(0.0, float(alarm_time - s_start)))

        w_sens = float((pdf[pdf["label_50pct_overlap"] == 1]["pred_label"] == 1).mean()) if (pdf["label_50pct_overlap"] == 1).sum() > 0 else 0.0
        w_spec = float((pdf[pdf["label_50pct_overlap"] == 0]["pred_label"] == 0).mean())
        fp_count = int(((pdf["label_50pct_overlap"] == 0) & (pdf["pred_label"] == 1)).sum())
        fa_per_day = (fp_count / dur_h) * 24.0 if dur_h > 0 else 0.0
        mean_delay = float(np.mean(delays)) if delays else np.nan

        pat_rows.append({
            "patient_id": pat_id,
            "total_windows": len(pdf),
            "recording_hours": round(dur_h, 2),
            "num_seizures": n_events,
            "detected_seizures": det_events,
            "missed_seizures": n_events - det_events,
            "event_sensitivity": round(det_events / n_events, 4) if n_events > 0 else 0.0,
            "window_sensitivity": round(w_sens, 5),
            "window_specificity": round(w_spec, 5),
            "false_alarms_count": fp_count,
            "false_alarms_per_day": round(fa_per_day, 2),
            "mean_detection_delay_sec": round(mean_delay, 2) if not np.isnan(mean_delay) else None
        })

    df_pat = pd.DataFrame(pat_rows)
    df_pat.to_csv(OUTPUT_PATIENT_CSV, index=False)
    print("\nPatient-Level Breakdown:")
    print(df_pat.to_string())

    # Event-level breakdown
    event_rows = []
    for _, ev in events_df[events_df["patient_id"].isin(["chb01", "chb02", "chb03", "chb05"])].iterrows():
        s_id = ev["seizure_id"]
        rec_id = ev["recording_id"]
        p_id = ev["patient_id"]
        s_start = ev["start_sec"]
        s_end = ev["end_sec"]
        dur = ev["duration_sec"]
        rw = test_df_eval[(test_df_eval["recording_id"] == rec_id) & (test_df_eval["window_end_sec"] > s_start) & (test_df_eval["window_start_sec"] < s_end)]
        det_w = rw[rw["pred_label"] == 1]
        is_det = len(det_w) > 0
        first_alarm = float(det_w["window_end_sec"].min()) if is_det else None
        delay = float(max(0.0, first_alarm - s_start)) if is_det else None
        event_rows.append({
            "patient_id": p_id,
            "recording_id": rec_id,
            "seizure_id": s_id,
            "start_sec": s_start,
            "end_sec": s_end,
            "duration_sec": dur,
            "detected": is_det,
            "first_alarm_sec": first_alarm,
            "detection_delay_sec": delay,
            "overlapping_positive_windows": len(det_w)
        })
    df_event = pd.DataFrame(event_rows)
    df_event.to_csv(OUTPUT_EVENT_CSV, index=False)
    print(f"\nSaved {OUTPUT_EVENT_CSV} with {len(df_event)} events.")
    print(f"Total time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    run()
