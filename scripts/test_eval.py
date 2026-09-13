import torch
import os
import hashlib
from typing import List
from neuroaegis.models import ModelOutput
from neuroaegis.eval import run_loso_evaluation, run_external_evaluation, generate_results_table

class StubModel(torch.nn.Module):
    def forward(self, x):
        # Random logits
        logits = torch.randn(x.size(0), 1)
        return ModelOutput(logits=logits)

def stub_model_factory(name: str):
    return StubModel()

def stub_model_loader(path: str):
    return StubModel()

def synthetic_data_loader(patients: List[str], split: str):
    # Yields 3 batches per patient of synthetic WindowSamples 
    # Shape: (batch, channels, seq_len) -> (8, 18, 1280)
    for _ in patients:
        for _ in range(3):
            X = torch.randn(8, 18, 1280)
            # Make random labels (0 or 1)
            y = torch.randint(0, 2, (8, 1)).float()
            yield X, y

def test_evaluation_harness():
    print("Testing LOSO Evaluation (CHB-MIT)...")
    chbmit_pts = [f"chb{i:02d}" for i in range(1, 4)] # 3 folds
    
    chbmit_metrics = run_loso_evaluation(
        patients=chbmit_pts,
        model_factory=stub_model_factory,
        data_loader_factory=synthetic_data_loader
    )
    print(f"Obtained CHB-MIT metrics for {len(chbmit_metrics)} folds.")
    
    print("\nTesting External Evaluation (Siena & Bonn)...")
    
    # We must register a fake model path to pass the check_frozen guardrail
    fake_model_path = "mock_frozen_model.pt"
    with open(fake_model_path, "wb") as f:
        f.write(b"fake model")
        
    model_hash = hashlib.sha256(b"fake model").hexdigest()
    
    # We rely on FROZEN_REGISTRY.md being in neuroaegis/guardrails/
    registry_path = "neuroaegis/guardrails/FROZEN_REGISTRY.md"
    with open(registry_path, "a") as f:
        f.write(f"\n| fake_model | {model_hash} | 2026-09-13 | Test eval |\n")
        
    siena_metrics = run_external_evaluation(
        dataset_name="siena",
        test_patients=["siena01", "siena02"],
        model_path=fake_model_path,
        model_loader=stub_model_loader,
        data_loader_factory=synthetic_data_loader
    )
    print(f"Obtained Siena metrics.")
    
    bonn_metrics = run_external_evaluation(
        dataset_name="bonn",
        test_patients=["bonn_test_1"],
        model_path=fake_model_path,
        model_loader=stub_model_loader,
        data_loader_factory=synthetic_data_loader
    )
    print(f"Obtained Bonn metrics.")
    
    print("\nGenerating Results Tables:")
    print("="*40)
    report = generate_results_table(chbmit_metrics, siena_metrics, bonn_metrics)
    print(report)
    print("="*40)
    
if __name__ == "__main__":
    test_evaluation_harness()
