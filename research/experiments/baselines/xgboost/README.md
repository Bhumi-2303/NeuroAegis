# XGBoost Baseline Experiment

## Overview
Evaluates **XGBoost** trained on 57 multi-domain EEG features (Time, Frequency, Wavelet) using the exact patient-independent CHB-MIT split.

## Dataset & Split
- **Train (16 patients, 10:1 balanced subsampling)**: chb04, chb09, chb11, chb12, chb13, chb14, chb15, chb16, chb17, chb18, chb19, chb20, chb21, chb22, chb23, chb24 (36,388 windows)
- **Validation (4 patients)**: chb06, chb07, chb08, chb10 (293,410 windows)
- **Test (4 held-out patients, locked)**: chb01, chb02, chb03, chb05 (219,909 windows, 152.82 hours)

## Features & Preprocessing
- **Feature Count**: 57 multi-domain features per window (channel-averaged).
- **Imputation**: Median imputation fit strictly on Training set.
- **Normalization**: Z-score StandardScaler fit strictly on Training set.

## Hyperparameters & Tuning
- **Selection Criterion**: Max Validation AUPRC (0.00572).
- **Selected Hyperparameters**: `{'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 4, 'scale_pos_weight': 10.0, 'random_state': 42, 'n_jobs': 4}`
- **Optimal Threshold**: $\tau = 0.65$ (selected on Validation set by max F1).
- **Training Fit Time**: 0.31 seconds.
- **Inference Latency**: 0.0002 ms/window.
- **Git Commit**: `f6b32ee20036e86aa626aa11217b80e999fdfc2e`

## Results on Held-Out Test Set (152.82 Continuous Hours)

| Metric | Value |
| :--- | :--- |
| **AUROC** | 0.71778 |
| **AUPRC** | 0.00980 |
| **Sensitivity (Window)** | 1.41% |
| **Specificity (Window)** | 99.76% |
| **Precision** | 1.69% |
| **F1 Score** | 0.01541 |
| **Balanced Accuracy** | 50.59% |
| **Event Sensitivity** | 0.00% (0/22) |
| **Mean Detection Delay** | 0.00 s |
| **False-Positive Windows** | 522 |
| **False-Alarm Episodes** | 75 |
| **FA / 24h** | 11.78 |
| **Inference Latency** | 0.0002 ms/window |
