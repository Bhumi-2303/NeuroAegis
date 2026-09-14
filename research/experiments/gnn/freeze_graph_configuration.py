"""
NeuroAegis Phase 4A-C: Freeze Selected Graph Configuration & Model Checkpoint
Confirms theta = 0.30 as the validation-selected configuration.
Generates:
  - research/experiments/gnn/frozen_graph_config.json
  - research/experiments/gnn/frozen_graph_adjacency.csv
  - artifacts/checkpoints/frozen_cnn_gnn.pt
"""

import os
import sys
import json
import shutil
import time
import subprocess
import pandas as pd
import torch

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
CHANNEL_ORDER_PATH = os.path.join(BASE_DIR, "research/data/config/chbmit_channel_order.json")
SRC_ADJ_PATH = os.path.join(BASE_DIR, "research/experiments/gnn_candidates/theta_030/graph_adjacency.csv")
SRC_CHECKPOINT_PATH = os.path.join(BASE_DIR, "research/experiments/gnn_candidates/theta_030/best_cnn_gnn.pt")

DST_CONFIG_PATH = os.path.join(BASE_DIR, "research/experiments/gnn/frozen_graph_config.json")
DST_ADJ_PATH = os.path.join(BASE_DIR, "research/experiments/gnn/frozen_graph_adjacency.csv")
DST_CHECKPOINT_PATH = os.path.join(BASE_DIR, "artifacts/checkpoints/frozen_cnn_gnn.pt")

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR).decode().strip()
    except Exception:
        return "UNKNOWN"

def freeze_configuration():
    print("=" * 80)
    print("FREEZING SELECTED GRAPH CONFIGURATION (theta = 0.30)")
    print("=" * 80)
    
    with open(CHANNEL_ORDER_PATH) as f:
        channels = json.load(f)
        
    df_adj = pd.read_csv(SRC_ADJ_PATH, index_col=0)
    df_adj.to_csv(DST_ADJ_PATH)
    print(f"Saved frozen adjacency matrix to {DST_ADJ_PATH}")
    
    # Copy checkpoint
    shutil.copyfile(SRC_CHECKPOINT_PATH, DST_CHECKPOINT_PATH)
    print(f"Copied frozen checkpoint to {DST_CHECKPOINT_PATH}")
    
    # Verify checkpoint
    ckpt = torch.load(DST_CHECKPOINT_PATH, map_location="cpu")
    print(f"Verified checkpoint: Epoch {ckpt['epoch']}, Val AUPRC = {ckpt['val_auprc']:.5f}, Val AUROC = {ckpt['val_auroc']:.5f}")
    
    frozen_cfg = {
        "phase": "4A-C",
        "dataset": "CHB-MIT",
        "num_nodes": len(channels),
        "channel_order": channels,
        "adjacency_method": "Thresholded Pearson Correlation with Self-Loops",
        "correlation_method": "Pearson cross-correlation on unlabelled training EEG recordings",
        "threshold": 0.30,
        "num_edges": 40,
        "density": 0.1581,
        "num_components": 2,
        "largest_component_size": 19,
        "smallest_component_size": 4,
        "occipital_leads_disconnected": True,
        "self_loops": True,
        "directed": False,
        "edge_weight_normalization": "Symmetric Kipf-Welling (D^-0.5 * (A + I) * D^-0.5)",
        "selection_split": "validation",
        "selection_metric_primary": "AUPRC (0.00159)",
        "selection_metric_secondary": "AUROC (0.25887)",
        "validation_metrics": {
            "validation_auprc": ckpt["val_auprc"],
            "validation_auroc": ckpt["val_auroc"],
            "validation_sensitivity": ckpt["val_sensitivity"],
            "validation_specificity": ckpt["val_specificity"],
            "validation_f1": ckpt["val_f1"]
        },
        "random_seed": 42,
        "git_commit": get_git_commit(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    with open(DST_CONFIG_PATH, "w") as f:
        json.dump(frozen_cfg, f, indent=2)
    print(f"Saved frozen configuration to {DST_CONFIG_PATH}")

if __name__ == "__main__":
    freeze_configuration()
