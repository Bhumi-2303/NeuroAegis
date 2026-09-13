from pydantic import BaseModel
from typing import Optional, Dict

class EvalConfig(BaseModel):
    min_seizure_duration_sec: float
    min_predicted_event_duration_sec: float
    merge_gap_sec: float
    allowed_delay_sec: float
    multi_alarm_policy: str
    smoothing_window_size: int

class EventMetrics(BaseModel):
    """
    Contract 2.4: Evaluation Metrics Schema
    """
    fa_per_24h: float
    event_sensitivity: float
    sensitivity: float
    auprc: float
    f1_score: float
    specificity: float
    precision: float
    auroc: float
    detection_delay_sec: float
    accuracy: float
    
    def to_ordered_dict(self) -> Dict[str, float]:
        """
        Enforces metric priority order in every report/print/log.
        """
        return {
            "fa_per_24h": self.fa_per_24h,
            "event_sensitivity": self.event_sensitivity,
            "sensitivity": self.sensitivity,
            "auprc": self.auprc,
            "f1_score": self.f1_score,
            "specificity": self.specificity,
            "precision": self.precision,
            "auroc": self.auroc,
            "detection_delay_sec": self.detection_delay_sec,
            "accuracy": self.accuracy
        }
