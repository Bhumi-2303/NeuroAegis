import yaml
import os
import torch
import numpy as np
from typing import List, Callable, Any
from neuroaegis.eval.schemas import EvalConfig, EventMetrics
from neuroaegis.eval.metrics import evaluate_event_level
from neuroaegis.guardrails.checks import assert_no_patient_leakage, check_frozen

def load_eval_config(config_path: str = "frozen_eval_config.yaml") -> EvalConfig:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Frozen eval config {config_path} missing. Cannot evaluate.")
    
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
        
    try:
        return EvalConfig(**data)
    except Exception as e:
        raise ValueError(f"Eval config is missing required parameters or is malformed: {e}")

def run_loso_evaluation(
    patients: List[str], 
    model_factory: Callable[[str], Any], 
    data_loader_factory: Callable[[List[str], str], Any], 
    config_path: str = "frozen_eval_config.yaml"
) -> List[EventMetrics]:
    """
    Leave-One-Subject-Out harness for CHB-MIT.
    """
    eval_config = load_eval_config(config_path)
    fold_metrics = []
    
    for i, test_pt in enumerate(patients):
        test_patients = {test_pt}
        train_patients = set(patients) - test_patients
        
        # Guardrail
        assert_no_patient_leakage(train_patients, test_patients)
        
        # Mock load model and data
        model = model_factory(f"fold_{i}")
        test_loader = data_loader_factory(list(test_patients), "test")
        
        y_true_all = []
        y_pred_probs_all = []
        total_hours = 0.0
        
        # Inference
        for X_batch, y_batch in test_loader:
            with torch.no_grad():
                out = model(X_batch)
                probs = torch.sigmoid(out.logits).cpu().numpy().flatten()
                
            y_true_all.extend(y_batch.cpu().numpy().flatten())
            y_pred_probs_all.extend(probs)
            # Rough proxy for duration: assume each window is 5 sec
            total_hours += (X_batch.size(0) * 5.0) / 3600.0
            
        metrics = evaluate_event_level(
            np.array(y_true_all),
            np.array(y_pred_probs_all),
            eval_config,
            total_duration_hours=total_hours
        )
        fold_metrics.append(metrics)
        
    return fold_metrics

def run_external_evaluation(
    dataset_name: str,
    test_patients: List[str],
    model_path: str,
    model_loader: Callable[[str], Any],
    data_loader_factory: Callable[[List[str], str], Any],
    config_path: str = "frozen_eval_config.yaml"
) -> List[EventMetrics]:
    """
    External evaluation for Siena or Bonn.
    Must check registry lock before running.
    """
    # Guardrail: Must be registered
    check_frozen(model_path)
    
    eval_config = load_eval_config(config_path)
    
    model = model_loader(model_path)
    test_loader = data_loader_factory(test_patients, "test")
    
    y_true_all = []
    y_pred_probs_all = []
    total_hours = 0.0
    
    # Inference
    for X_batch, y_batch in test_loader:
        with torch.no_grad():
            out = model(X_batch)
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
    
    # Return as list of 1 to be compatible with aggregator
    return [metrics]
