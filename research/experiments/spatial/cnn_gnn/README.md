# CNN + GNN (Spatial Topology)

## Overview
Ablation study stage evaluating **CNN + GNN (Spatial Topology)** on the quarantined continuous CHB-MIT test set (152.82 continuous hours).

## Configuration
- **Active Modules**: `Conv1D + GNN`
- **Parameters**: 52,497
- **FLOPs**: 42.8 MFLOPs
- **Decision Threshold**: $\tau = 0.50$

## Test Set Results
- **AUROC**: 0.19431
- **AUPRC**: 0.00492
- **Window Sensitivity**: 4.24%
- **Window Specificity**: 99.42%
- **Window Precision**: 2.07%
- **F1 Score**: 0.02784
- **Event Sensitivity**: 18.18% (4/22)
- **Mean Detection Delay**: 10.00 s
- **False Alarms / 24h**: 32.82 (209 episodes)
- **Inference Latency**: 0.82 ms/window
