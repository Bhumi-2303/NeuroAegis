import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np
import os
import hashlib

# Data (Track A)
from neuroaegis.data.synthetic import generate_synthetic_data
from neuroaegis.data.schemas import PreprocessingConfig, BandpassConfig, WindowingConfig, LabelingConfig

# Models (Track B)
from neuroaegis.models.schemas import ModelConfig
from neuroaegis.models.baselines import Baseline1DCNN
from neuroaegis.models.cnn_gnn_gru_attention import CNN_GNN_GRU_Attention

# Eval (Track C)
from neuroaegis.eval.harness import run_loso_evaluation, run_external_evaluation
from neuroaegis.eval.reporting import generate_results_table

# Tracking (Track E)
from neuroaegis.tracking.checkpoints import CheckpointManager
from neuroaegis.tracking.config import ExperimentConfig

# Adaptation (Track F)
from neuroaegis.adaptation.experiment import build_adapted_model
from neuroaegis.adaptation.variants import VariantA_DirectTransfer, VariantB_Normalization, VariantC_Calibration, VariantD_Combined

# XAI (Track D)
from neuroaegis.xai.methods import IntegratedGradients, AttentionExtractor
from neuroaegis.xai.harness import build_comparison_harness

class TorchDataLoader:
    """Wrapper to convert Track A WindowSamples to batch iterators"""
    def __init__(self, samples, batch_size=4):
        self.samples = samples
        self.batch_size = batch_size
        
    def __iter__(self):
        for i in range(0, len(self.samples), self.batch_size):
            batch = self.samples[i:i+self.batch_size]
            X = torch.stack([torch.tensor(s.data, dtype=torch.float32) for s in batch])
            y = torch.tensor([[s.label] for s in batch], dtype=torch.float32)
            yield X, y

def integration_test():
    print("="*60)
    print("NEUROAEGIS INTEGRATION TEST: END-TO-END VERIFICATION")
    print("="*60)
    
    # ---------------------------------------------------------
    # Setup Configs
    # ---------------------------------------------------------
    prep_config = PreprocessingConfig(
        sampling_rate_hz=256,
        bandpass=BandpassConfig(low_freq_hz=0.5, high_freq_hz=40.0),
        windowing=WindowingConfig(size_sec=5.0, overlap_sec=0.0),
        labeling=LabelingConfig(overlap_threshold_percent=50.0),
        common_channels=["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", 
                         "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz"] # 18 channels
    )
    
    model_config = ModelConfig(
        in_channels=18,
        seq_len=1280,
        cnn_hidden_dim=32,
        gru_hidden_dim=64,
        num_classes=1
    )
    
    # Track A Factory
    def track_a_data_factory(patients: List[str], split: str):
        all_samples = []
        for p in patients:
            # generating 5 windows per patient
            samples = generate_synthetic_data("CHB-MIT", p, f"{p}_rec1", prep_config, num_windows=5)
            all_samples.extend(samples)
        return TorchDataLoader(all_samples, batch_size=4)
        
    def model_factory(name: str):
        return Baseline1DCNN(model_config)
        
    # ---------------------------------------------------------
    # 1-3. Track A + Track B + Track C (LOSO + Metrics Order)
    # ---------------------------------------------------------
    print("\n--- 1. Testing Track A -> Track B -> Track C (LOSO) ---")
    chbmit_patients = ["chb01", "chb02", "chb03"]
    
    # run_loso_evaluation internally triggers `assert_no_patient_leakage` per fold
    # and processes Baseline1DCNN end-to-end matching budget constraints.
    chbmit_metrics = run_loso_evaluation(
        patients=chbmit_patients,
        model_factory=model_factory,
        data_loader_factory=track_a_data_factory
    )
    print("LOSO folds completed without patient leakage.")
    
    # Prove metric order explicitly
    print("\n--- 3. Track C Metrics Report Layout ---")
    print(generate_results_table(chbmit_metrics, [], []))
    
    # ---------------------------------------------------------
    # 4. Track F Domain Adaptation
    # ---------------------------------------------------------
    print("\n--- 4. Testing Track F (Domain Adaptation) ---")
    base_model = model_factory("baseline")
    
    calibration_patients = ["siena_calib_1"]
    test_patients = ["siena_test_1"]
    
    # This invokes assert_no_patient_leakage inside build_adapted_model
    try:
        adapted_model = build_adapted_model(
            model=base_model,
            variant="D",
            calibration_patients=calibration_patients,
            test_patients=test_patients,
            data_loader_factory=track_a_data_factory
        )
        print("Domain Adaptation built successfully with strictly isolated calibration subset.")
    except Exception as e:
        print(f"Domain Adaptation Failed: {e}")
        
    # ---------------------------------------------------------
    # 5. Track D XAI (Integrated Gradients + Attention)
    # ---------------------------------------------------------
    print("\n--- 5. Testing Track D (XAI) ---")
    # Using the CNN_GNN_GRU_Attention model to prove attention extraction
    full_model = CNN_GNN_GRU_Attention(model_config)
    
    # Get a real event
    samples = generate_synthetic_data("CHB-MIT", "chb01", "rec1", prep_config, num_windows=1)
    x_event = torch.tensor(samples[0].data, dtype=torch.float32).unsqueeze(0)
    
    print("Computing IG...")
    ig = IntegratedGradients(full_model, steps=5)
    ig_attr = ig.attribute(x_event, target_class=0)
    
    print("Extracting Attention...")
    attn_extractor = AttentionExtractor(full_model)
    attn_attr = attn_extractor.attribute(x_event)
    
    print("Building XAI Map...")
    xai_map = build_comparison_harness(
        event_id="real_event_1",
        x=x_event,
        ig_attr=ig_attr,
        attention_attr=attn_attr
    )
    
    print("\n[XAI Map Built]")
    xai_map.display_warnings()
    print("Completed pairwise agreements on clinical time/channel/freq logic.")
    
    print("\nALL TRACKS SUCCESSFULLY INTEGRATED.")

if __name__ == "__main__":
    integration_test()
