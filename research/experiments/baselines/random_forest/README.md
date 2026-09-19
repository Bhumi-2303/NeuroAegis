# Random Forest Baseline Experiment

## Overview
Evaluates **Random Forest** trained on 57 multi-domain EEG features (Time, Frequency, Wavelet) using the exact patient-independent CHB-MIT split.

## Dataset & Split
- **Train (16 patients, 10:1 balanced subsampling)**: chb04, chb09, chb11, chb12, chb13, chb14, chb15, chb16, chb17, chb18, chb19, chb20, chb21, chb22, chb23, chb24 (36,388 windows)
- **Validation (4 patients)**: chb06, chb07, chb08, chb10 (293,410 windows)
- **Test (4 held-out patients, locked)**: chb01, chb02, chb03, chb05 (219,909 windows, 152.82 hours)

## Features & Preprocessing
- **Feature Count**: 57 multi-domain features per window (channel-averaged).
- **Imputation**: Median imputation fit strictly on Training set.
- **Normalization**: Z-score StandardScaler fit strictly on Training set.

## Hyperparameters & Tuning
- **Selection Criterion**: Max Validation AUPRC (0.00548).
- **Selected Hyperparameters**: `{'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': 4}`
- **Optimal Threshold**: $\tau = 0.40$ (selected on Validation set by max F1).
- **Training Fit Time**: 6.61 seconds.
- **Inference Latency**: 0.001 ms/window.
- **Git Commit**: `f6b32ee20036e86aa626aa11217b80e999fdfc2e`

## Results on Held-Out Test Set (152.82 Continuous Hours)

| Metric | Value |
| :--- | :--- |
| **AUROC** | 0.62465 |
| **AUPRC** | 0.00407 |
| **Sensitivity (Window)** | 1.88% |
| **Specificity (Window)** | 98.66% |
| **Precision** | 0.41% |
| **F1 Score** | 0.00667 |
| **Balanced Accuracy** | 50.27% |
| **Event Sensitivity** | 0.00% (0/22) |
| **Mean Detection Delay** | 0.00 s |
| **False-Positive Windows** | 2,947 |
| **False-Alarm Episodes** | 513 |
| **FA / 24h** | 80.57 |
| **Inference Latency** | 0.001 ms/window |
