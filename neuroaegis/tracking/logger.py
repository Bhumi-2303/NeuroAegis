import json
import os
import time
from datetime import datetime
import torch
from neuroaegis.tracking.config import ExperimentConfig

class ExperimentLogger:
    def __init__(self, log_file: str = "experiment_logs.jsonl"):
        self.log_file = log_file

    def log_run(
        self, 
        config: ExperimentConfig, 
        param_count: int, 
        training_time_sec: float
    ):
        """
        Logs every config field + param count + hardware + time.
        """
        if torch.backends.mps.is_available():
            hardware = "MPS"
        elif torch.cuda.is_available():
            hardware = f"CUDA ({torch.cuda.get_device_name(0)})"
        else:
            hardware = "CPU"
            
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "hardware": hardware,
            "param_count": param_count,
            "training_time_sec": training_time_sec,
            "config": config.model_dump()
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
