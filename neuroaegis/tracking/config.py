import yaml
from pydantic import BaseModel, StrictInt, StrictFloat, StrictStr, StrictBool, ValidationError
from typing import Dict, Any, List

class BandpassConfig(BaseModel):
    low_freq_hz: StrictFloat
    high_freq_hz: StrictFloat

class WindowingConfig(BaseModel):
    size_sec: StrictFloat
    overlap_sec: StrictFloat

class LabelingConfig(BaseModel):
    overlap_threshold_percent: StrictFloat

class ExperimentConfig(BaseModel):
    """
    Contract 2.2 strict schema.
    Every field must be present and typed correctly.
    """
    experiment_name: StrictStr
    seed: StrictInt
    
    # Preprocessing block
    sampling_rate_hz: StrictInt
    bandpass: BandpassConfig
    windowing: WindowingConfig
    labeling: LabelingConfig
    common_channels: List[StrictStr]
    
    # Model block
    model_name: StrictStr
    cnn_hidden_dim: StrictInt
    gnn_hidden_dim: StrictInt
    gru_hidden_dim: StrictInt
    graph_method: StrictStr
    loss: StrictStr
    
    # Training block
    batch_size: StrictInt
    epochs: StrictInt
    learning_rate: StrictFloat

def load_and_validate_config(config_path: str) -> ExperimentConfig:
    """
    Loads and validates config. Refuses to start if missing or typed incorrectly.
    """
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
        
    try:
        return ExperimentConfig(**data)
    except ValidationError as e:
        raise ValueError(f"Config validation failed! Run refused. Details: {e}")

def dict_diff(d1: Dict[str, Any], d2: Dict[str, Any], path: str = "") -> Dict[str, Any]:
    diffs = {}
    all_keys = set(d1.keys()).union(set(d2.keys()))
    for k in all_keys:
        full_path = f"{path}.{k}" if path else k
        if k not in d1:
            diffs[full_path] = (None, d2[k])
        elif k not in d2:
            diffs[full_path] = (d1[k], None)
        else:
            if isinstance(d1[k], dict) and isinstance(d2[k], dict):
                sub_diffs = dict_diff(d1[k], d2[k], full_path)
                diffs.update(sub_diffs)
            elif d1[k] != d2[k]:
                diffs[full_path] = (d1[k], d2[k])
    return diffs

def diff_configs(config_path1: str, config_path2: str):
    """
    Diff two configs to confirm only intended components changed (e.g. ablation).
    """
    with open(config_path1, 'r') as f:
        c1 = yaml.safe_load(f)
    with open(config_path2, 'r') as f:
        c2 = yaml.safe_load(f)
        
    diffs = dict_diff(c1, c2)
    return diffs
