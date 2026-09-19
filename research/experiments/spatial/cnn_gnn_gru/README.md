# CNN + GNN + GRU (Model C Frozen)

## Overview
Ablation study stage evaluating **CNN + GNN + GRU (Model C Frozen)** on the quarantined continuous CHB-MIT test set (152.82 continuous hours).

## Configuration
- **Active Modules**: `Conv1D + GNN + Causal GRU`
- **Parameters**: 91,858
- **FLOPs**: 43.1 MFLOPs
- **Decision Threshold**: $\tau = 0.50$

## Test Set Results
- **AUROC**: 0.98970
- **AUPRC**: 0.80681
- **Window Sensitivity**: 83.83%
- **Window Specificity**: 99.82%
- **Window Precision**: 57.23%
- **F1 Score**: 0.68025
- **Event Sensitivity**: 86.36% (19/22)
- **Mean Detection Delay**: 12.89 s
- **False Alarms / 24h**: 8.01 (51 episodes)
- **Inference Latency**: 0.01 ms/window
