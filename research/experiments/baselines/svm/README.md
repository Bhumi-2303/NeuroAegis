# Linear SVM Baseline Experiment

## Overview
Evaluates **Linear SVM** trained on 57 multi-domain EEG features (Time, Frequency, Wavelet) using the exact patient-independent CHB-MIT split.

## Dataset & Split
- **Train (16 patients, 10:1 balanced subsampling)**: chb04, chb09, chb11, chb12, chb13, chb14, chb15, chb16, chb17, chb18, chb19, chb20, chb21, chb22, chb23, chb24 (36,388 windows)
- **Validation (4 patients)**: chb06, chb07, chb08, chb10 (293,410 windows)
- **Test (4 held-out patients, locked)**: chb01, chb02, chb03, chb05 (219,909 windows, 152.82 hours)

## Features & Preprocessing
- **Feature Count**: 57 multi-domain features per window (channel-averaged).
- **Imputation**: Median imputation fit strictly on Training set.
- **Normalization**: Z-score StandardScaler fit strictly on Training set.

## Hyperparameters & Tuning
- **Selection Criterion**: Max Validation AUPRC (0.00340).
- **Selected Hyperparameters**: `{'loss': 'log_loss', 'alpha': 0.001, 'penalty': 'l2', 'class_weight': 'balanced', 'random_state': 42, 'max_iter': 1000}`
- **Optimal Threshold**: $\tau = 0.10$ (selected on Validation set by max F1).
- **Training Fit Time**: 0.26 seconds.
- **Inference Latency**: 0.0001 ms/window.
- **Git Commit**: `f6b32ee20036e86aa626aa11217b80e999fdfc2e`

## Results on Held-Out Test Set (152.82 Continuous Hours)

| Metric | Value |
| :--- | :--- |
| **AUROC** | 0.65417 |
| **AUPRC** | 0.01189 |
| **Sensitivity (Window)** | 72.53% |
| **Specificity (Window)** | 44.10% |
| **Precision** | 0.38% |
| **F1 Score** | 0.00747 |
| **Balanced Accuracy** | 58.32% |
| **Event Sensitivity** | 86.36% (19/22) |
| **Mean Detection Delay** | 1.32 s |
| **False-Positive Windows** | 122,565 |
| **False-Alarm Episodes** | 3,286 |
| **FA / 24h** | 516.06 |
| **Inference Latency** | 0.0001 ms/window |
