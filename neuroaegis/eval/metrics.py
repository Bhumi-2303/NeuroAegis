import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from neuroaegis.eval.schemas import EvalConfig, EventMetrics
from neuroaegis.guardrails.checks import false_alarms_per_24h

def _get_intervals(binary_sequence: np.ndarray, window_size_sec: float) -> List[Tuple[float, float]]:
    intervals = []
    in_event = False
    start_idx = 0
    for i, val in enumerate(binary_sequence):
        if val == 1 and not in_event:
            in_event = True
            start_idx = i
        elif val == 0 and in_event:
            in_event = False
            intervals.append((start_idx * window_size_sec, i * window_size_sec))
    if in_event:
        intervals.append((start_idx * window_size_sec, len(binary_sequence) * window_size_sec))
    return intervals

def apply_false_alarm_protocol(y_pred_binary: np.ndarray, config: EvalConfig, window_size_sec: float = 5.0) -> List[Tuple[float, float]]:
    # 1. Temporal Smoothing (Median Filter equivalent for binary: sliding window majority)
    pad = config.smoothing_window_size // 2
    smoothed = np.copy(y_pred_binary)
    if pad > 0 and len(y_pred_binary) >= config.smoothing_window_size:
        for i in range(pad, len(y_pred_binary) - pad):
            smoothed[i] = 1 if np.sum(y_pred_binary[i-pad:i+pad+1]) > pad else 0
            
    # Convert to time intervals
    raw_intervals = _get_intervals(smoothed, window_size_sec)
    
    # 2. Merge nearby alarms
    merged_intervals = []
    for interval in raw_intervals:
        if not merged_intervals:
            merged_intervals.append(interval)
        else:
            last = merged_intervals[-1]
            gap = interval[0] - last[1]
            if gap <= config.merge_gap_sec:
                merged_intervals[-1] = (last[0], interval[1])
            else:
                merged_intervals.append(interval)
                
    # 3. Remove detections shorter than min_duration_sec
    final_intervals = [
        intv for intv in merged_intervals 
        if (intv[1] - intv[0]) >= config.min_predicted_event_duration_sec
    ]
    return final_intervals

def evaluate_event_level(
    y_true: np.ndarray, 
    y_pred_probs: np.ndarray, 
    config: EvalConfig, 
    total_duration_hours: float,
    window_size_sec: float = 5.0,
    threshold: float = 0.5
) -> EventMetrics:
    y_pred_binary = (y_pred_probs >= threshold).astype(int)
    
    # Window-level metrics
    auprc = float(average_precision_score(y_true, y_pred_probs)) if len(np.unique(y_true)) > 1 else 0.0
    auroc = float(roc_auc_score(y_true, y_pred_probs)) if len(np.unique(y_true)) > 1 else 0.0
    sens = float(recall_score(y_true, y_pred_binary, zero_division=0))
    prec = float(precision_score(y_true, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred_binary))
    
    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    # Event-level parsing
    true_intervals = _get_intervals(y_true, window_size_sec)
    # Filter true events by min seizure duration
    true_events = [i for i in true_intervals if (i[1] - i[0]) >= config.min_seizure_duration_sec]
    
    # Predicted events (applying the false-alarm protocol)
    pred_events = apply_false_alarm_protocol(y_pred_binary, config, window_size_sec)
    
    # Matching
    matched_preds = set()
    detected_trues = 0
    delays = []
    
    for true_start, true_end in true_events:
        detected = False
        for p_idx, (p_start, p_end) in enumerate(pred_events):
            # Overlap check
            if p_end > true_start and p_start < true_end:
                delay = p_start - true_start
                if delay <= config.allowed_delay_sec:
                    detected = True
                    matched_preds.add(p_idx)
                    delays.append(max(0.0, delay))
                    if config.multi_alarm_policy == "first_only":
                        break
        if detected:
            detected_trues += 1
            
    # Any predicted event not in matched_preds is a False Alarm
    # Raw FP count is an intermediate, only FA/24h is reported
    raw_false_alarms = len(pred_events) - len(matched_preds)
    fa_24h = false_alarms_per_24h(raw_false_alarms, total_duration_hours)
    
    event_sensitivity = (detected_trues / len(true_events)) if true_events else 1.0
    avg_delay = float(np.mean(delays)) if delays else 0.0
    
    return EventMetrics(
        fa_per_24h=fa_24h,
        event_sensitivity=event_sensitivity,
        sensitivity=sens,
        auprc=auprc,
        f1_score=f1,
        specificity=spec,
        precision=prec,
        auroc=auroc,
        detection_delay_sec=avg_delay,
        accuracy=acc
    )
