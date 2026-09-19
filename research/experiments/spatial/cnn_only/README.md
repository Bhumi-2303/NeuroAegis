# CNN-only (1D CNN Temporal)

## Overview
Ablation study stage evaluating **CNN-only (1D CNN Temporal)** on the quarantined continuous CHB-MIT test set (152.82 continuous hours).

## Configuration
- **Active Modules**: `Conv1D Temporal`
- **Parameters**: 173,601
- **FLOPs**: 135.2 MFLOPs
- **Decision Threshold**: $\tau = 0.50$

## Test Set Results
- **AUROC**: 0.36389
- **AUPRC**: 0.04148
- **Window Sensitivity**: 16.01%
- **Window Specificity**: 94.35%
- **Window Precision**: 0.82%
- **F1 Score**: 0.01553
- **Event Sensitivity**: 31.82% (7/22)
- **Mean Detection Delay**: 7.14 s
- **False Alarms / 24h**: 329.80 (2,100 episodes)
- **Inference Latency**: 0.34 ms/window
