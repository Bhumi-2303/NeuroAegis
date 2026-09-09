"""
NeuroAegis Seizure Detection Clinical Metrics Suite
Phase: Class Imbalance Handling Strategy

Implements clinically meaningful seizure detection evaluation metrics:
Sensitivity, Specificity, Precision, F1, Balanced Accuracy, AUROC, AUPRC,
and False Alarms per 24 Hours (FA/24h).
"""

from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve


class SeizureEvaluationMetrics:
    """
    Computes comprehensive window-level and clinical event-level evaluation metrics.
    Emphasizes clinically actionable metrics (Sensitivity, AUPRC, False Alarms/24h)
    over naive raw accuracy.
    """
    
    @staticmethod
    def compute_window_metrics(
        y_true: Union[np.ndarray, list],
        y_prob: Union[np.ndarray, list],
        threshold: float = 0.5,
        total_duration_hours: Optional[float] = None,
        stride_sec: float = 2.5
    ) -> Dict[str, Any]:
        """
        Computes window-level binary classification and clinical metrics.
        
        Args:
            y_true: Array-like of ground truth binary labels (0 or 1).
            y_prob: Array-like of predicted seizure probabilities in [0, 1].
            threshold: Decision threshold for converting probabilities to binary predictions.
            total_duration_hours: Total hours of EEG recorded. If None, estimated from sample count & stride.
            stride_sec: Window temporal stride in seconds (default: 2.5s).
            
        Returns:
            Dictionary containing computed clinical metrics.
        """
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)
        
        if len(y_true) != len(y_prob):
            raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_prob ({len(y_prob)})")
            
        y_pred = (y_prob >= threshold).astype(int)
        
        # Confusion matrix elements
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        
        total_samples = len(y_true)
        pos_samples = tp + fn
        neg_samples = tn + fp
        
        # Core clinical metrics
        sensitivity = tp / pos_samples if pos_samples > 0 else 0.0
        specificity = tn / neg_samples if neg_samples > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        if (precision + sensitivity) > 0:
            f1_score = 2.0 * (precision * sensitivity) / (precision + sensitivity)
        else:
            f1_score = 0.0
            
        balanced_accuracy = (sensitivity + specificity) / 2.0
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
        
        # Area under curve metrics
        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            # Handles edge cases where only one class is present
            auroc = float("nan")
            
        try:
            auprc = float(average_precision_score(y_true, y_prob))
        except ValueError:
            auprc = float("nan")
            
        # Total monitoring duration in hours
        if total_duration_hours is None:
            # Each window stride represents stride_sec of independent continuous progress
            total_duration_hours = (total_samples * stride_sec) / 3600.0
            
        # False alarms per 24 hours calculation
        if total_duration_hours > 0:
            false_alarms_per_24h = (fp / total_duration_hours) * 24.0
        else:
            false_alarms_per_24h = 0.0
            
        return {
            "threshold": threshold,
            "total_windows": total_samples,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "sensitivity": round(sensitivity, 5),
            "specificity": round(specificity, 5),
            "precision": round(precision, 5),
            "f1_score": round(f1_score, 5),
            "balanced_accuracy": round(balanced_accuracy, 5),
            "accuracy": round(accuracy, 5),
            "auroc": round(auroc, 5) if not np.isnan(auroc) else None,
            "auprc": round(auprc, 5) if not np.isnan(auprc) else None,
            "total_duration_hours": round(total_duration_hours, 2),
            "false_alarms_per_24h": round(false_alarms_per_24h, 2)
        }
        
    @staticmethod
    def compute_event_level_sensitivity(
        window_df: pd.DataFrame,
        y_prob: np.ndarray,
        threshold: float = 0.5,
        label_column: str = "label_any_overlap"
    ) -> Dict[str, Any]:
        """
        Computes seizure event-level sensitivity.
        An event is counted as detected if at least one window overlapping the event
        is predicted as positive (probability >= threshold).
        """
        df = window_df.copy()
        df["pred_prob"] = y_prob
        df["pred_label"] = (df["pred_prob"] >= threshold).astype(int)
        
        # Filter to ictal windows with event IDs
        ictal_df = df[df[label_column] == 1]
        
        event_detection = {}
        for _, row in ictal_df.iterrows():
            e_ids = str(row.get("seizure_event_ids", "")).split(";")
            for e_id in e_ids:
                e_id = e_id.strip()
                if not e_id:
                    continue
                if e_id not in event_detection:
                    event_detection[e_id] = False
                if row["pred_label"] == 1:
                    event_detection[e_id] = True
                    
        total_events = len(event_detection)
        detected_events = sum(1 for detected in event_detection.values() if detected)
        event_sensitivity = detected_events / total_events if total_events > 0 else 0.0
        
        return {
            "total_seizure_events": total_events,
            "detected_seizure_events": detected_events,
            "event_level_sensitivity": round(event_sensitivity, 4)
        }
