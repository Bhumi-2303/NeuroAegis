from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import numpy as np
from dataclasses import dataclass

class BandpassConfig(BaseModel):
    low_freq_hz: float = 0.5
    high_freq_hz: float = 40.0

class WindowingConfig(BaseModel):
    size_sec: float = 5.0
    overlap_sec: float = 0.0

class LabelingConfig(BaseModel):
    overlap_threshold_percent: float = 50.0

class PreprocessingConfig(BaseModel):
    sampling_rate_hz: int = 256
    bandpass: BandpassConfig = BandpassConfig()
    notch_filter_hz: Optional[float] = None
    common_channels: List[str] = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ"]
    windowing: WindowingConfig = WindowingConfig()
    labeling: LabelingConfig = LabelingConfig()

@dataclass
class WindowSample:
    """
    Contract 2.1: WindowSample records emitted by the data pipeline.
    """
    patient_id: str
    recording_id: str
    window_start_sec: float
    window_end_sec: float
    data: np.ndarray  # Shape: (n_channels, n_samples)
    label: int        # 1 for seizure, 0 for non-seizure
    dataset_source: str
