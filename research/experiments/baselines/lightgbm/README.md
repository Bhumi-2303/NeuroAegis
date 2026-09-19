# LightGBM Baseline Experiment

## Overview
Evaluates **LightGBM** trained on 57 multi-domain EEG features (Time, Frequency, Wavelet) using the exact patient-independent CHB-MIT split.

## Dataset & Split
- **Train (16 patients, 10:1 balanced subsampling)**: chb04, chb09, chb11, chb12, chb13, chb14, chb15, chb16, chb17, chb18, chb19, chb20, chb21, chb22, chb23, chb24 (36,388 windows)
- **Validation (4 patients)**: chb06, chb07, chb08, chb10 (293,410 windows)
- **Test (4 held-out patients, locked)**: chb01, chb02, chb03, chb05 (219,909 windows, 152.82 hours)

## Features & Preprocessing
- **Feature Count**: 57 multi-domain features per window (channel-averaged).
- **Imputation**: Median imputation fit strictly on Training set.
- **Normalization**: Z-score StandardScaler fit strictly on Training set.

## Hyperparameters & Tuning
- **Selection Criterion**: Max Validation AUPRC (0.00620).
- **Selected Hyperparameters**: `{'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31, 'scale_pos_weight': 10.0, 'random_state': 42, 'n_jobs': 4, 'verbosity': -1}`
- **Optimal Threshold**: $\tau = 0.45$ (selected on Validation set by max F1).
- **Training Fit Time**: 0.81 seconds.
- **Inference Latency**: 0.0027 ms/window.
- **Git Commit**: `f6b32ee20036e86aa626aa11217b80e999fdfc2e`

## Results on Held-Out Test Set (152.82 Continuous Hours)

| Metric | Value |
| :--- | :--- |
| **AUROC** | 0.69973 |
| **AUPRC** | 0.00692 |
| **Sensitivity (Window)** | 6.91% |
| **Specificity (Window)** | 98.75% |
| **Precision** | 1.58% |
| **F1 Score** | 0.02569 |
| **Balanced Accuracy** | 52.83% |
| **Event Sensitivity** | 4.55% (1/22) |
| **Mean Detection Delay** | 30.00 s |
| **False-Positive Windows** | 2,745 |
| **False-Alarm Episodes** | 344 |
| **FA / 24h** | 54.02 |
| **Inference Latency** | 0.0027 ms/window |
