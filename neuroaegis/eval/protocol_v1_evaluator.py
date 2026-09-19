"""
NeuroAegis Protocol V1.0 Authoritative Evaluator
Standardized, leakage-free, reproducible evaluation engine for seizure detection.
"""

import math
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score
)


def evaluate_window_level(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes standard window-level classification metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    if len(y_true) != len(y_prob):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_prob ({len(y_prob)})")
        
    y_pred = (y_prob >= threshold).astype(int)
    
    # Confusion Matrix
    unique_labels = np.unique(y_true)
    if len(unique_labels) == 2 or (0 in unique_labels and 1 in unique_labels):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    elif 0 in unique_labels:
        # Only negative class present
        tn = int((y_pred == 0).sum())
        fp = int((y_pred == 1).sum())
        fn = 0
        tp = 0
    else:
        # Only positive class present
        tn = 0
        fp = 0
        fn = int((y_pred == 0).sum())
        tp = int((y_pred == 1).sum())
        
    pos_samples = tp + fn
    neg_samples = tn + fp
    total_samples = len(y_true)
    
    sens = float(tp / pos_samples) if pos_samples > 0 else 0.0
    spec = float(tn / neg_samples) if neg_samples > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2.0 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
    acc = float((tp + tn) / total_samples) if total_samples > 0 else 0.0
    bal_acc = float((sens + spec) / 2.0)
    
    # AUROC and AUPRC
    try:
        auroc = float(roc_auc_score(y_true, y_prob)) if len(unique_labels) > 1 else float("nan")
    except Exception:
        auroc = float("nan")
        
    try:
        auprc = float(average_precision_score(y_true, y_prob)) if len(unique_labels) > 1 else float("nan")
    except Exception:
        auprc = float("nan")
        
    return {
        "threshold": float(threshold),
        "total_windows": int(total_samples),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity": round(sens, 5),
        "specificity": round(spec, 5),
        "precision": round(prec, 5),
        "f1_score": round(f1, 5),
        "balanced_accuracy": round(bal_acc, 5),
        "accuracy": round(acc, 5),
        "auroc": round(auroc, 5) if not math.isnan(auroc) else None,
        "auprc": round(auprc, 5) if not math.isnan(auprc) else None
    }


def _get_intervals_from_binary_stream(
    binary_seq: np.ndarray,
    stride_sec: float = 2.5,
    win_dur_sec: float = 5.0
) -> List[Tuple[float, float]]:
    """
    Extracts contiguous active time intervals from a stepped binary window stream.
    """
    intervals = []
    in_event = False
    start_idx = 0
    
    for i, val in enumerate(binary_seq):
        if val == 1 and not in_event:
            in_event = True
            start_idx = i
        elif val == 0 and in_event:
            in_event = False
            start_time = start_idx * stride_sec
            end_time = (i - 1) * stride_sec + win_dur_sec
            intervals.append((start_time, end_time))
            
    if in_event:
        start_time = start_idx * stride_sec
        end_time = (len(binary_seq) - 1) * stride_sec + win_dur_sec
        intervals.append((start_time, end_time))
        
    return intervals


def apply_alarm_protocol_v1(
    y_pred_binary: np.ndarray,
    stride_sec: float = 2.5,
    win_dur_sec: float = 5.0,
    smooth_size: int = 3,
    merge_gap_sec: float = 15.0,
    min_dur_sec: float = 5.0
) -> List[Tuple[float, float]]:
    """
    Applies the Authoritative Protocol V1 alarm post-processing:
    1. Temporal Majority Filtering
    2. Contiguous Interval Extraction
    3. Gap-based Merging (<= merge_gap_sec)
    4. Minimum Duration Filtering (>= min_dur_sec)
    """
    # 1. Temporal Majority Filtering
    pad = smooth_size // 2
    smoothed = np.copy(y_pred_binary)
    if pad > 0 and len(y_pred_binary) >= smooth_size:
        for i in range(pad, len(y_pred_binary) - pad):
            window_slice = y_pred_binary[i - pad : i + pad + 1]
            smoothed[i] = 1 if np.sum(window_slice) > pad else 0
            
    # 2. Extract Raw Intervals
    raw_intervals = _get_intervals_from_binary_stream(smoothed, stride_sec, win_dur_sec)
    
    # 3. Merge Nearby Intervals
    merged_intervals = []
    for intv in raw_intervals:
        if not merged_intervals:
            merged_intervals.append(intv)
        else:
            last = merged_intervals[-1]
            gap = intv[0] - last[1]
            if gap <= merge_gap_sec:
                merged_intervals[-1] = (last[0], max(last[1], intv[1]))
            else:
                merged_intervals.append(intv)
                
    # 4. Filter by Minimum Duration
    final_intervals = [
        intv for intv in merged_intervals
        if (intv[1] - intv[0]) >= min_dur_sec
    ]
    return final_intervals


def evaluate_stream_protocol_v1(
    pred_df: pd.DataFrame,
    events_df: pd.DataFrame,
    total_monitoring_hours: float,
    threshold: float = 0.50,
    stride_sec: float = 2.5,
    win_dur_sec: float = 5.0,
    smooth_size: int = 3,
    merge_gap_sec: float = 15.0,
    min_dur_sec: float = 5.0,
    allowed_delay_sec: float = 30.0,
    label_col: str = "label_50pct_overlap"
) -> Dict[str, Any]:
    """
    Standardized Protocol V1 evaluation across all recordings and patients.
    """
    df = pred_df.copy()
    
    # Auto-detect probability column
    prob_col = None
    for candidate in ["predicted_probability", "prediction_prob", "y_prob", "raw_probability", "prob"]:
        if candidate in df.columns:
            prob_col = candidate
            break
    if prob_col is None:
        raise KeyError(f"Predictions DataFrame must contain a valid probability column. Found columns: {list(df.columns)}")
    probs = df[prob_col].values.astype(float)
    
    # Ensure recording_id is present
    if "recording_id" not in df.columns:
        if "edf_filename" in df.columns:
            df["recording_id"] = df["edf_filename"].astype(str).str.replace(".edf", "", regex=False)
        elif "recording" in df.columns:
            df["recording_id"] = df["recording"]
        else:
            raise KeyError("Predictions DataFrame must contain 'recording_id' or 'edf_filename'.")
        
    labels = df[label_col].values.astype(int)
    
    # 1. Window-Level Evaluation
    win_metrics = evaluate_window_level(labels, probs, threshold=threshold)
    
    # 2. Recording-Level Alarm Generation and Event Matching
    recordings = df["recording_id"].unique()
    if events_df is None or len(events_df) == 0 or "recording_id" not in events_df.columns:
        sub_events = pd.DataFrame(columns=["recording_id", "seizure_id", "start_sec", "end_sec"])
    else:
        sub_events = events_df[events_df["recording_id"].isin(recordings)].sort_values(["recording_id", "start_sec"]).reset_index(drop=True)
    total_events = len(sub_events)
    
    total_alarms = 0
    matched_alarm_indices = []
    unmatched_alarms = 0
    detected_event_count = 0
    onset_delays = []
    completion_delays = []
    
    event_match_records = []
    
    for rec_id, rec_group in df.groupby("recording_id", sort=False):
        rec_probs = rec_group[prob_col].values.astype(float)
        rec_binary = (rec_probs >= threshold).astype(int)
        
        # Per-recording alarm post-processing
        rec_alarms = apply_alarm_protocol_v1(
            rec_binary,
            stride_sec=stride_sec,
            win_dur_sec=win_dur_sec,
            smooth_size=smooth_size,
            merge_gap_sec=merge_gap_sec,
            min_dur_sec=min_dur_sec
        )
        total_alarms += len(rec_alarms)
        
        rec_events = sub_events[sub_events["recording_id"] == rec_id]
        rec_matched_alarms = set()
        
        for _, ev in rec_events.iterrows():
            s_st = float(ev["start_sec"])
            s_en = float(ev["end_sec"])
            s_id = ev.get("seizure_id", f"{rec_id}_sz")
            
            ev_detected = False
            first_onset_delay = None
            first_comp_delay = None
            
            for a_idx, (a_st, a_en) in enumerate(rec_alarms):
                # Overlap check
                if a_en > s_st and a_st < s_en:
                    onset_d = max(0.0, a_st - s_st)
                    if onset_d <= allowed_delay_sec:
                        ev_detected = True
                        rec_matched_alarms.add(a_idx)
                        if first_onset_delay is None:
                            first_onset_delay = onset_d
                            first_comp_delay = max(0.0, a_en - s_st)
                            
            if ev_detected:
                detected_event_count += 1
                onset_delays.append(first_onset_delay)
                completion_delays.append(first_comp_delay)
                
            event_match_records.append({
                "recording_id": rec_id,
                "seizure_id": s_id,
                "start_sec": s_st,
                "end_sec": s_en,
                "detected": ev_detected,
                "onset_delay_sec": round(first_onset_delay, 2) if first_onset_delay is not None else None,
                "completion_delay_sec": round(first_comp_delay, 2) if first_comp_delay is not None else None
            })
            
        unmatched_in_rec = len(rec_alarms) - len(rec_matched_alarms)
        unmatched_alarms += unmatched_in_rec
        
    # Rate calculations
    raw_fp = win_metrics["fp"]
    raw_fp_per_24h = (raw_fp / total_monitoring_hours) * 24.0 if total_monitoring_hours > 0 else 0.0
    clinical_fa_per_24h = (unmatched_alarms / total_monitoring_hours) * 24.0 if total_monitoring_hours > 0 else 0.0
    
    event_sens = (detected_event_count / total_events) if total_events > 0 else 1.0
    mean_onset_delay = float(np.mean(onset_delays)) if onset_delays else None
    median_onset_delay = float(np.median(onset_delays)) if onset_delays else None
    std_onset_delay = float(np.std(onset_delays)) if onset_delays else None
    
    mean_comp_delay = float(np.mean(completion_delays)) if completion_delays else None
    median_comp_delay = float(np.median(completion_delays)) if completion_delays else None
    
    # Model collapse detection (BENDR special case check)
    is_collapsed = bool(win_metrics["specificity"] == 0.0 and win_metrics["sensitivity"] == 1.0)
    
    return {
        "protocol_version": "v1.0",
        "threshold": float(threshold),
        "monitoring_hours": round(float(total_monitoring_hours), 4),
        "total_windows": win_metrics["total_windows"],
        "window_metrics": win_metrics,
        "event_metrics": {
            "total_events": int(total_events),
            "detected_events": int(detected_event_count),
            "missed_events": int(total_events - detected_event_count),
            "event_sensitivity": round(float(event_sens), 5),
            "mean_onset_delay_sec": round(mean_onset_delay, 2) if mean_onset_delay is not None else None,
            "median_onset_delay_sec": round(median_onset_delay, 2) if median_onset_delay is not None else None,
            "std_onset_delay_sec": round(std_onset_delay, 2) if std_onset_delay is not None else None,
            "mean_completion_delay_sec": round(mean_comp_delay, 2) if mean_comp_delay is not None else None,
            "median_completion_delay_sec": round(median_comp_delay, 2) if median_comp_delay is not None else None,
            "event_details": event_match_records
        },
        "alarm_metrics": {
            "raw_fp_windows": int(raw_fp),
            "raw_fp_windows_per_24h": round(float(raw_fp_per_24h), 2),
            "total_alarm_episodes": int(total_alarms),
            "clinical_false_alarm_episodes": int(unmatched_alarms),
            "clinical_fa_episodes_per_24h": round(float(clinical_fa_per_24h), 2),
            "is_collapsed_alert_state": is_collapsed
        }
    }
