import torch
import torch.nn as nn
from typing import List, Any
import numpy as np
import pandas as pd

from neuroaegis.models import ModelOutput
from neuroaegis.adaptation import build_adapted_model
from neuroaegis.eval import evaluate_event_level, EvalConfig

class StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(18 * 128, 1)
        
    def forward(self, x):
        # x shape: (batch, channels, seq_len) -> (batch, 18, 128)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        logits = self.fc(x_flat)
        return ModelOutput(logits=logits)

def synthetic_siena_loader(patients: List[str], split: str):
    # Returns a list of 5 batches per patient
    data = []
    for _ in patients:
        for _ in range(5):
            X = torch.randn(8, 18, 128)
            # Imbalance logic for testing BCE/Calibration
            y = torch.randint(0, 2, (8, 1)).float()
            # If calibration, maybe shift X to simulate domain shift
            if split == "calibration":
                X = X * 2.0 + 1.0 # different mean/std
            elif split == "test":
                X = X * 2.5 + 1.5 
            data.append((X, y))
    return data

def test_adaptation_pipeline():
    print("Initializing Stub Model...")
    model = StubModel()
    
    calibration_patients = ["siena_calib_01", "siena_calib_02"]
    test_patients = ["siena_test_01", "siena_test_02"]
    
    variants = ["A", "B", "C", "D"]
    variant_names = {
        "A": "Direct Transfer",
        "B": "Normalization",
        "C": "Calibration",
        "D": "Harmonization + Calib"
    }
    
    # Eval Config stub
    eval_config = EvalConfig(
        min_seizure_duration_sec=10.0,
        min_predicted_event_duration_sec=5.0,
        merge_gap_sec=15.0,
        allowed_delay_sec=30.0,
        multi_alarm_policy="first_only",
        smoothing_window_size=3
    )
    
    results = []
    
    for var in variants:
        print(f"\nBuilding Adapted Model for Variant {var} ({variant_names[var]})...")
        adapted_model = build_adapted_model(
            model=model,
            variant=var,
            calibration_patients=calibration_patients,
            test_patients=test_patients,
            data_loader_factory=synthetic_siena_loader
        )
        
        # Test evaluating
        test_loader = synthetic_siena_loader(test_patients, "test")
        y_true_all = []
        y_pred_probs_all = []
        total_hours = 0.0
        
        adapted_model.eval()
        for X_batch, y_batch in test_loader:
            with torch.no_grad():
                out = adapted_model(X_batch)
                probs = torch.sigmoid(out.logits).cpu().numpy().flatten()
                
            y_true_all.extend(y_batch.cpu().numpy().flatten())
            y_pred_probs_all.extend(probs)
            total_hours += (X_batch.size(0) * 5.0) / 3600.0
            
        metrics = evaluate_event_level(
            np.array(y_true_all),
            np.array(y_pred_probs_all),
            eval_config,
            total_duration_hours=total_hours
        )
        
        # Extract subset of metrics for the domain adaptation table spec
        results.append({
            "Model": "StubCNN",
            "Adaptation Variant": variant_names[var],
            "Sensitivity": f"{metrics.sensitivity:.4f}",
            "AUPRC": f"{metrics.auprc:.4f}",
            "F1": f"{metrics.f1_score:.4f}",
            "FA/24h": f"{metrics.fa_per_24h:.4f}"
        })
        
    print("\n" + "="*80)
    print("CHB-MIT -> Siena Domain Adaptation Results")
    print("="*80)
    
    # We will build a manual markdown table to avoid pandas tabulate error
    columns = ["Model", "Adaptation Variant", "Sensitivity", "AUPRC", "F1", "FA/24h"]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    
    print(header)
    print(separator)
    for row in results:
        print("| " + " | ".join([row[c] for c in columns]) + " |")
        
if __name__ == "__main__":
    test_adaptation_pipeline()
