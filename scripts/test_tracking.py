import torch
import torch.nn as nn
import os
import yaml
import json
from neuroaegis.tracking import load_and_validate_config, diff_configs, ExperimentLogger, seed_everything, CheckpointManager

def create_dummy_config(path: str, name: str, seed: int, model_name: str = "cnn"):
    data = {
        "experiment_name": name,
        "seed": seed,
        "sampling_rate_hz": 256,
        "bandpass": {"low_freq_hz": 0.5, "high_freq_hz": 40.0},
        "windowing": {"size_sec": 5.0, "overlap_sec": 0.0},
        "labeling": {"overlap_threshold_percent": 50.0},
        "common_channels": ["Fp1", "Fp2"],
        "model_name": model_name,
        "cnn_hidden_dim": 64,
        "gnn_hidden_dim": 128,
        "gru_hidden_dim": 256,
        "graph_method": "correlation",
        "loss": "weighted_bce",
        "batch_size": 8,
        "epochs": 1,
        "learning_rate": 0.001
    }
    with open(path, "w") as f:
        yaml.dump(data, f)

def test_tracking_pipeline():
    print("Creating dummy configs...")
    c1_path = "dummy_config1.yaml"
    c2_path = "dummy_config2.yaml"
    
    create_dummy_config(c1_path, "exp1_cnn", 42, "cnn")
    create_dummy_config(c2_path, "exp2_cnn_gru", 42, "cnn_gru") # Ablation diff target
    
    print("Testing Seed Everything...")
    seed_everything(42)
    
    print("\nTesting Config Validation...")
    config1 = load_and_validate_config(c1_path)
    print(f"Loaded config: {config1.experiment_name}")
    
    print("\nTesting Experiment Logger...")
    logger = ExperimentLogger("dummy_logs.jsonl")
    logger.log_run(config1, param_count=500000, training_time_sec=120.5)
    
    with open("dummy_logs.jsonl", "r") as f:
        last_log = json.loads(f.readlines()[-1])
        print(f"Logged parameter count: {last_log['param_count']}")
        print(f"Logged hardware: {last_log['hardware']}")
    
    print("\nTesting Checkpoint Manager...")
    model = nn.Linear(10, 2)
    manager = CheckpointManager(save_dir="dummy_checkpoints", registry_path="dummy_checkpoints/registry.json")
    ckpt_path = manager.save_checkpoint(model, epoch=1, config=config1)
    
    with open("dummy_checkpoints/registry.json", "r") as f:
        registry = json.load(f)
        for h, metadata in registry.items():
            print(f"Registry Hash: {h[:12]}...")
            print(f" -> Frozen: {metadata['frozen']}")
            print(f" -> Model Name: {metadata['config']['model_name']}")
    
    print("\nTesting Config Diff CLI...")
    diffs = diff_configs(c1_path, c2_path)
    print("Differences between exp1 and exp2:")
    for k, (v1, v2) in diffs.items():
        print(f"  {k}: {v1} -> {v2}")

    # Cleanup
    os.remove(c1_path)
    os.remove(c2_path)
    os.remove("dummy_logs.jsonl")
    
if __name__ == "__main__":
    test_tracking_pipeline()
