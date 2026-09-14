from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Any

__all__ = [
    "evaluate_event_level",
    "evaluate_patient_aggregate",
]

def _compute_total_duration_sec(timestamps: np.ndarray, window_sec: float) -> float:
    """Computes total recording duration from timestamps."""
    if len(timestamps) == 0:
        return 0.0
    return float(timestamps[-1] - timestamps[0] + window_sec)

def _compute_non_seizure_duration_hours(total_sec: float, true_events: list[tuple[float, float]]) -> float:
    """Computes non-seizure duration in hours for FA/24h calculation."""
    seizure_sec = sum(max(0.0, end - start) for start, end in true_events)
    return max(0.0, total_sec - seizure_sec) / 3600.0

def _extract_events(binary_preds: np.ndarray, timestamps: np.ndarray, window_sec: float) -> list[tuple[float, float]]:
    """Extracts contiguous event intervals from binary predictions.
    
    Args:
        binary_preds: np.ndarray of shape (N,) containing 0s and 1s.
        timestamps: np.ndarray of shape (N,) containing start times.
        window_sec: float duration of each window.
        
    Returns:
        List of (start_sec, end_sec) intervals representing contiguous predicted events.
    """
    events = []
    in_event = False
    start_time = 0.0
    
    for i in range(len(binary_preds)):
        if binary_preds[i] == 1 and not in_event:
            in_event = True
            start_time = timestamps[i]
        elif binary_preds[i] == 0 and in_event:
            in_event = False
            end_time = timestamps[i-1] + window_sec
            events.append((float(start_time), float(end_time)))
            
    if in_event and len(binary_preds) > 0:
        end_time = timestamps[-1] + window_sec
        events.append((float(start_time), float(end_time)))
        
    return events

def _binarize_and_postprocess(
    y_pred_probs: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    min_duration_sec: float,
    merge_gap_sec: float,
    window_sec: float
) -> np.ndarray:
    """Returns post-processed binary predictions by merging close events and filtering short ones."""
    binary = (y_pred_probs >= threshold).astype(int)
    if len(binary) == 0:
        return binary

    # Extract raw events
    events = _extract_events(binary, timestamps, window_sec)
    
    # 1. Merge predicted events separated by <= merge_gap_sec
    merged_events = []
    for event in events:
        if not merged_events:
            merged_events.append(event)
        else:
            prev_start, prev_end = merged_events[-1]
            start, end = event
            if start - prev_end <= merge_gap_sec:
                merged_events[-1] = (prev_start, max(prev_end, end))
            else:
                merged_events.append(event)

    # 2. Remove predicted events shorter than min_duration_sec
    filtered_events = [ev for ev in merged_events if (ev[1] - ev[0]) >= min_duration_sec]

    # Reconstruct binary array based on filtered events
    post_binary = np.zeros_like(binary)
    for start, end in filtered_events:
        # A window is considered positive if its start falls within the event bounds
        mask = (timestamps >= start) & (timestamps < end)
        post_binary[mask] = 1
    
    return post_binary

def _match_events(
    true_events: list[tuple[float, float]],
    pred_events: list[tuple[float, float]],
    max_delay_sec: float
) -> tuple[list, list, list]:
    """Matches true events and predicted events.
    
    Args:
        true_events: list of (start, end) true events.
        pred_events: list of (start, end) predicted events.
        max_delay_sec: maximum allowable detection delay from event start.
        
    Returns:
        tuple of (matched_true, unmatched_true, false_alarm_pred).
        matched_true contains tuples of (start, end, delay).
    """
    matched_true = []
    unmatched_true = []
    matched_pred_indices = set()
    
    for t_start, t_end in true_events:
        matched = False
        for i, (p_start, p_end) in enumerate(pred_events):
            if i in matched_pred_indices:
                continue
            
            # Check overlap
            overlap = max(0.0, min(t_end, p_end) - max(t_start, p_start)) > 0
            if overlap:
                delay = max(0.0, p_start - t_start)
                if delay <= max_delay_sec:
                    matched = True
                    matched_pred_indices.add(i)
                    matched_true.append((t_start, t_end, delay))
                    break
                    
        if not matched:
            unmatched_true.append((t_start, t_end))
            
    false_alarm_pred = [p for i, p in enumerate(pred_events) if i not in matched_pred_indices]
    
    return matched_true, unmatched_true, false_alarm_pred

def evaluate_event_level(
    y_true_events: list[tuple[float, float]],
    y_pred_probs: np.ndarray,
    timestamps: np.ndarray,
    patient_id: str,
    config: Any
) -> dict:
    """Evaluates event-level seizure detection metrics.
    
    Args:
        y_true_events: list of (start_sec, end_sec) tuples marking true seizure boundaries.
        y_pred_probs: np.ndarray of predicted seizure probabilities (one per window).
        timestamps: np.ndarray of window start times in seconds.
        patient_id: string identifier.
        config: config dictionary or object containing evaluation thresholds and post-processing params.
        
    Returns:
        dict containing ordered evaluation metrics.
    """
    # Extract config values
    if isinstance(config, dict):
        threshold = config.get("threshold", 0.5)
        min_duration_sec = config["post_processing"]["min_duration_sec"]
        merge_gap_sec = config["post_processing"]["merge_gap_sec"]
        max_delay_sec = config["post_processing"]["max_delay_sec"]
        window_sec = config["window_sec"]
    else:
        threshold = getattr(config, "threshold", 0.5)
        min_duration_sec = config.post_processing.min_duration_sec
        merge_gap_sec = config.post_processing.merge_gap_sec
        max_delay_sec = config.post_processing.max_delay_sec
        window_sec = config.window_sec

    # Generate sample-level ground truth binary array
    y_true_sample = np.zeros_like(y_pred_probs)
    for start, end in y_true_events:
        # Window is positive if it overlaps any true event
        mask = (timestamps + window_sec > start) & (timestamps < end)
        y_true_sample[mask] = 1

    # Raw binarization for sample-level metrics (before event post-processing)
    raw_binary = (y_pred_probs >= threshold).astype(int)
    
    # Calculate confusion matrix components
    if len(y_true_sample) > 0:
        cm = confusion_matrix(y_true_sample, raw_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = float(f1_score(y_true_sample, raw_binary, zero_division=0))
    
    if len(np.unique(y_true_sample)) > 1:
        auroc = float(roc_auc_score(y_true_sample, y_pred_probs))
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_sample, y_pred_probs)
        auprc = float(auc(recall_curve, precision_curve))
    else:
        auroc = float('nan')
        auprc = float('nan')

    # Post-process for event-level metrics
    binary_preds = _binarize_and_postprocess(
        y_pred_probs, timestamps, threshold, min_duration_sec, merge_gap_sec, window_sec
    )
    pred_events = _extract_events(binary_preds, timestamps, window_sec)
    
    # Match events
    matched_true, unmatched_true, false_alarms = _match_events(y_true_events, pred_events, max_delay_sec)
    
    event_sensitivity = len(matched_true) / len(y_true_events) if len(y_true_events) > 0 else 0.0
    
    total_sec = _compute_total_duration_sec(timestamps, window_sec)
    non_seizure_hours = _compute_non_seizure_duration_hours(total_sec, y_true_events)
    fa_per_24h = (len(false_alarms) / non_seizure_hours * 24.0) if non_seizure_hours > 0 else 0.0

    # Detection delay
    delays = [delay for _, _, delay in matched_true]
    detection_delay_sec = float(np.mean(delays)) if len(delays) > 0 else float('nan')

    return {
        "sensitivity": float(sensitivity),
        "event_sensitivity": float(event_sensitivity),
        "fa_per_24h": float(fa_per_24h),
        "auprc": float(auprc),
        "f1": float(f1),
        "specificity": float(specificity),
        "precision": float(precision),
        "auroc": float(auroc),
        "detection_delay_sec": float(detection_delay_sec),
        "accuracy": float(accuracy)
    }

def evaluate_patient_aggregate(per_window_results: list[dict]) -> dict:
    """Aggregate per-patient results into dataset-level summary with mean ± std.
    
    Args:
        per_window_results: list of dictionaries containing metrics for each patient/window.
        
    Returns:
        dict with aggregated metrics across all patients.
    """
    if not per_window_results:
        return {}
        
    keys = per_window_results[0].keys()
    aggregated = {}
    
    for key in keys:
        if key == "patient_id":
            continue
        values = [res[key] for res in per_window_results if res[key] is not None and not np.isnan(res[key])]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            aggregated[f"{key}_mean"] = float(mean_val)
            aggregated[f"{key}_std"] = float(std_val)
        else:
            aggregated[f"{key}_mean"] = float('nan')
            aggregated[f"{key}_std"] = float('nan')
            
    return aggregated
