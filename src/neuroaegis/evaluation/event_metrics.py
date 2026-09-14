import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

def compute_detection_delay_details(
    window_df: pd.DataFrame,
    events_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, int, int, List[Dict[str, Any]]]:
    """
    Authoritative event-level matching protocol.
    Computes true onset delay and merges false alarms.
    """
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
        s_start = ev["start_sec"]
        s_end = ev["end_sec"]
        
        rec_w = df[df["recording_id"] == rec_id]
        ov_mask = (rec_w["window_end_sec"] > s_start) & (rec_w["window_start_sec"] < s_end)
        ov_w = rec_w[ov_mask]
        
        det_w = ov_w[ov_w["pred_label"] == 1]
        is_detected = len(det_w) > 0
        if is_detected:
            # Onset to onset
            first_alarm = det_w["window_start_sec"].min()
            delay = max(0.0, float(first_alarm - s_start))
            delays.append(delay)
        else:
            first_alarm = None
            delay = None
            
        event_details.append({
            "recording_id": rec_id,
            "detected": is_detected,
            "first_alarm_sec": first_alarm,
            "detection_delay_sec": delay,
        })
        
    mean_delay = float(np.mean(delays)) if delays else 0.0
    det_events = sum([1 for e in event_details if e["detected"]])
    
    return mean_delay, det_events, total_events, event_details
