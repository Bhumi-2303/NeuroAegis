# Experiment 2 — EEG-Specific Deep Learning Architectures

## Objective
Evaluate standard lightweight EEG-specific deep learning architectures trained directly on 23-channel raw windowed EEG signals (1,280 samples = 5.0 seconds at 256 Hz) without spatial graph inductive biases or multi-window temporal recurrence. Compare performance against the frozen NeuroAegis spatio-temporal Model C benchmark.

## Evaluated Architectures (< 500,000 Parameters)
1. **EEGNet** (Lawhern et al., 2018): 2,113 parameters. Compact 2D convolutional network employing temporal convolution, depthwise spatial convolution, and separable convolution.
2. **ShallowConvNet** (Schirrmeister et al., 2017): 41,041 parameters. Inspired by Filter Bank Common Spatial Patterns (FBCSP), using temporal and spatial convolutions followed by squaring, average pooling, and logarithmic bandpower transformation.
3. **Lightweight DeepConvNet** (Schirrmeister et al., 2017): 178,776 parameters. 4-block deep convolutional network with temporal + spatial first block followed by progressive 1D convolutions and max pooling.
4. **Lightweight 1D CNN** (NeuroAegis Multi-Scale Baseline): 173,601 parameters. Multi-scale 1D temporal convolutions processing channels independently before channel fusion.
5. **Model C Reference (Frozen)**: 91,858 parameters. Combined spatial 1D-CNN + Graph Neural Network (theta=0.30) + Causal GRU sequence model.

## Experimental Protocol & Scientific Safeguards
- **Input Data**: Raw normalized 23-channel EEG windows of shape (23, 1280).
- **Partitions**:
  - TRAIN (16 patients): chb04, chb09, chb11-chb24
  - VALIDATION (4 patients): chb06, chb07, chb08, chb10 (293,410 windows)
  - TEST (4 patients): chb01, chb02, chb03, chb05 (219,909 windows, 22 seizures, 152.82 hours) — LOCKED until validation checkpoint selection.
- **Training Setup**:
  - Batch size: 4
  - Epochs: 3
  - Loss function: Binary Focal Loss (gamma=2.0, alpha=0.25)
  - Negative sampling: 10:1 ratio per epoch with deterministic seeds
  - Optimizer: AdamW (lr=1e-3, weight_decay=1e-4) with CosineAnnealingLR
  - Hardware: Apple Silicon M4 (PyTorch MPS)
  - Memory: Memory-mapped epoch datasets and bounded streaming test evaluation.

---

## Summary Comparative Table

| Model | Params | Val AUPRC | Test AUROC | Test AUPRC | Window Sens | Window Spec | Event Sens | Mean Delay | False Alarms/24h | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EEGNet** | **2,113** | 0.06439 | 0.82506 | 0.04037 | 59.50% | 81.42% | **22/22 (100%)** | **5.57s** | 6,399.24 | **0.29 ms** |
| **ShallowConvNet** | 41,041 | 0.07115 | 0.84763 | 0.03746 | 54.32% | 87.59% | 21/22 (95.5%) | 5.43s | 4,274.28 | **0.25 ms** |
| **DeepConvNet** | 178,776 | 0.01950 | 0.85335 | 0.07248 | 47.72% | 89.91% | 20/22 (90.9%) | 16.45s | 3,476.02 | 0.34 ms |
| **Lightweight 1D CNN** | 173,601 | **0.08425** | **0.87227** | **0.08721** | 45.05% | **93.77%** | **22/22 (100%)** | 7.05s | **2,147.11** | 0.34 ms |
| **Model C (Frozen)** | 91,858 | **0.42002** | **0.98970** | **0.80681** | **83.83%** | **99.82%** | 21/22 (95.5%) | 10.57s | **62.66** | 0.01 ms |

---

## Key Clinical & Scientific Insights
1. **The Representation Gap (Single-Window vs Spatio-Temporal Graph)**:
   - Standalone single-window EEG architectures (EEGNet, ShallowConvNet, DeepConvNet, 1D CNN) achieved test AUROCs between 0.825 and 0.872 and AUPRCs between 0.037 and 0.087.
   - In stark contrast, Model C (which integrates spatial graph connectivity theta=0.30 and 8-window causal GRU temporal recurrence over 22.5s) achieves an AUROC of 0.98970 and AUPRC of 0.80681 (> 9x higher AUPRC).
   - This empirically demonstrates that single-window 2D convolutions cannot capture long-range pre-ictal temporal evolution or patient-invariant graph topologies necessary for clean seizure discrimination.

2. **False Alarm Profiles**:
   - Without graph spatial filtering and recurrent temporal context, standalone EEG models suffer extreme false alarm rates in continuous 24h monitoring (ranging from 2,147 to 6,399 false alarms per 24 hours).
   - Model C suppresses false alarms down to 62.66 / 24h (a 34-fold to 102-fold reduction), confirming that temporal recurrence acts as a critical temporal evidence accumulator that rejects transient non-seizure artifacts.

3. **Relative Comparison Among EEG Architectures**:
   - **Lightweight 1D CNN** was the superior performer among all pure EEG architectures: highest test AUROC (0.87227), highest test AUPRC (0.08721), lowest false alarm rate (2,147.11 / 24h), and 100% event sensitivity (22/22 seizures detected).
   - **EEGNet** demonstrated extreme parameter efficiency (2,113 parameters) and detected 22/22 seizures with rapid 5.57s detection delay, but suffered high false alarm rates (6,399 / 24h) due to low specificity (81.42%).
   - **ShallowConvNet** (41,041 params) improved specificity over EEGNet by 6.17% and reduced false alarms by 33% through its log-power filter bank structure.
   - **DeepConvNet** achieved higher specificity (89.91%) but struggled with detection latency (16.45s mean delay) and missed 2 seizures (90.91% event sensitivity).
