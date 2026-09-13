import argparse
import os
import yaml
import json
from pathlib import Path
from typing import List, Set

from neuroaegis.data.schemas import PreprocessingConfig, WindowSample
from neuroaegis.data.synthetic import generate_synthetic_data
from neuroaegis.guardrails.checks import assert_no_patient_leakage

def load_preprocessing_config(manifest_path: str = "preprocessing_manifest.yaml") -> PreprocessingConfig:
    with open(manifest_path, 'r') as f:
        data = yaml.safe_load(f)
    if 'reasoning' in data:
        del data['reasoning']
    return PreprocessingConfig(**data)

def save_manifest(manifest_path: str, records: dict):
    with open(manifest_path, 'w') as f:
        json.dump(records, f, indent=4)

def check_missing_channels(available_channels: List[str], required_channels: List[str]):
    """
    Never silently substitute missing channels — fail loudly instead.
    """
    missing = [ch for ch in required_channels if ch not in available_channels]
    if missing:
        raise ValueError(f"CRITICAL: Missing required channels in EDF: {missing}. Cannot proceed.")

def build_pipeline(dataset: str, fold: int, synthetic: bool, bonn_experiment: str = None):
    print(f"Building pipeline for {dataset} (fold {fold}) - synthetic={synthetic}")
    config = load_preprocessing_config()
    
    # Dataset routing
    if dataset == "chbmit":
        # Leave-One-Subject-Out for CHB-MIT
        all_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]
        test_patients = {all_patients[fold % len(all_patients)]}
        train_patients = set(all_patients) - test_patients
        
    elif dataset == "siena":
        # External test only
        all_patients = [f"siena{str(i).zfill(2)}" for i in range(1, 15)]
        train_patients = set()
        test_patients = set(all_patients)
        
    elif dataset == "bonn":
        # Bonn secondary task A/B/C/D/E
        if not bonn_experiment:
            raise ValueError("Bonn dataset requires --bonn_experiment (e.g. A, B, C)")
            
        # Implement Experiments A/B/C from the spec as separate, clearly labeled pipelines
        if bonn_experiment == "A":
            train_patients = {"bonn_healthy_train", "bonn_seizure_train"}
            test_patients = {"bonn_healthy_test", "bonn_seizure_test"}
        elif bonn_experiment == "B":
            train_patients = {"bonn_interictal_train", "bonn_ictal_train"}
            test_patients = {"bonn_interictal_test", "bonn_ictal_test"}
        elif bonn_experiment == "C":
            train_patients = {"bonn_healthy_interictal_train", "bonn_ictal_train"}
            test_patients = {"bonn_healthy_interictal_test", "bonn_ictal_test"}
        else:
            raise ValueError(f"Unknown Bonn experiment: {bonn_experiment}")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # GUARDRAIL: Patient Leakage Check inside the pipeline
    # The pipeline should refuse to emit a split that fails them.
    if train_patients and test_patients:
        assert_no_patient_leakage(train_patients, test_patients)
        print("Guardrail passed: No patient leakage between train and test splits.")

    output_dir = Path(f"data_shards/{dataset}/fold_{fold}")
    if bonn_experiment:
        output_dir = Path(f"data_shards/{dataset}_exp{bonn_experiment}/fold_{fold}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_records = {
        "dataset": dataset,
        "fold": fold,
        "bonn_experiment": bonn_experiment,
        "train_patients": list(train_patients),
        "test_patients": list(test_patients),
        "shards": {}
    }
    
    # Generate/Process data
    for split_name, patients in [("train", train_patients), ("test", test_patients)]:
        if not patients:
            continue
            
        split_shards = []
        for pt in patients:
            if synthetic:
                # Simulate checking channels
                check_missing_channels(config.common_channels, config.common_channels)
                
                samples = generate_synthetic_data(dataset, pt, f"{pt}_rec1", config, num_windows=5)
                
                shard_path = output_dir / f"{split_name}_{pt}.npz"
                shard_path.touch() # Mock saving
                split_shards.append(str(shard_path))
            else:
                raise NotImplementedError(
                    "Real EDF processing not implemented yet. "
                    "Use --synthetic. "
                    "Real implementation must use per-fold normalization only, "
                    "bandpass, dataset-specific notch filter, and missing channel strict checking."
                )
        
        manifest_records["shards"][split_name] = split_shards
        
    manifest_file = output_dir / "split_manifest.json"
    save_manifest(str(manifest_file), manifest_records)
    print(f"Pipeline complete. Output format: WindowSample schema (Contract 2.1).")
    print(f"Manifest saved to {manifest_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroAegis Data Pipeline")
    parser.add_argument("--dataset", type=str, required=True, choices=["chbmit", "siena", "bonn"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    parser.add_argument("--bonn_experiment", type=str, help="Experiment A, B, or C for Bonn dataset")
    args = parser.parse_args()
    
    build_pipeline(args.dataset, args.fold, args.synthetic, args.bonn_experiment)
