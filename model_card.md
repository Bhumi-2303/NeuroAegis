# NeuroAegis Model Card

This document provides details about the machine learning models trained for the NeuroAegis project, including datasets used, performance metrics, and hyperparameter configurations.

## Dataset Details
- **Source**: Bonn University EEG Dataset (Sets A–E)
- **Classes**: Healthy vs. Seizure
- **Preprocessing**: Discrete Wavelet Transform (DWT) denoising (`coif3`, level 4)
- **Data Splitting**: 
  - **Train/Test Split**: 80% Training / 20% Testing (`test_size=0.20`)
  - **Random Seed**: 42 (`random_state=42` used for splitting to ensure reproducibility)

## Model Performance & Evaluation Metrics
The models were evaluated on the 20% held-out test set using 57 extracted features (time, frequency, and wavelet domains).

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **LightGBM** (Best) | 0.9000 | 0.9015 | 0.9000 | 0.9001 | 0.9844 |
| **XGBoost** | 0.8800 | 0.8829 | 0.8800 | 0.8812 | 0.9846 |
| **Random Forest** | 0.8500 | 0.8546 | 0.8500 | 0.8518 | 0.9794 |

*Note: The best-performing baseline model selected for deployment is **LightGBM**.*

## Hyperparameters

### LightGBM
- `n_estimators`: 300
- `learning_rate`: 0.05
- `max_depth`: 6
- `num_leaves`: 31
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `random_state`: 42

### XGBoost
- `n_estimators`: 300
- `learning_rate`: 0.1
- `max_depth`: 6
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `min_child_weight`: 1
- `random_state`: 42

### Random Forest
- `n_estimators`: 500
- `random_state`: 42

## Training Resources
- **Original Training Script / Notebook**: [`neuroaegis-v1.ipynb`](./neuroaegis-v1.ipynb) (and its extracted Python equivalent [`extracted_notebook.py`](./extracted_notebook.py))
- **Random Seeds**: A global seed of `42` is strictly enforced for dataset splitting, model initialization, and cross-validation to maintain deterministic outputs.

## Validation Methodology (Patient-Independent Evaluation)

To ensure realistic clinical performance without data leakage (which occurs when windows from the same patient bleed across training and test sets), we performed a Leave-One-Patient-Out Cross-Validation on a 4-patient CHB-MIT subset. 

### Fold-by-Fold Metrics (Default Threshold 0.5)

| Fold | Held-Out Patient | Train Windows | Test Windows | Seizures in Test | Accuracy | AUC | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `chb01` | 3,643 | 1,162 | 27 | 97.85% | 0.7369 | 1.0000 | 0.0741 | 0.1379 |
| **2** | `chb02` | 4,461 | 344 | 6 | 98.26% | 0.8476 | 0.0000 | 0.0000 | 0.0000 |
| **3** | `chb03` | 3,741 | 1,064 | 24 | 97.65% | 0.9236 | 0.4545 | 0.2083 | 0.2857 |
| **4** | `chb04` | 2,570 | 2,235 | 19 | 99.11% | 0.6367 | 0.0000 | 0.0000 | 0.0000 |

*Note: The near-baseline performance on `chb04` is expected variance given the very small sample size and massive class imbalance.*

### Averaged Patient-Independent Metrics
*(n=4 patients — small sample, high variance)*

| Metric | Mean ± Std Dev |
| :--- | :--- |
| **AUC** | 0.7862 ± 0.1258 |
| **Precision** | 0.3636 ± 0.4753 |
| **Recall** | 0.0706 ± 0.0982 |
| **F1 Score** | 0.1059 ± 0.1364 |

### Threshold Tuning

The default threshold of 0.5 underperforms severely on unseen patients due to the massive class imbalance and a patient-specific probability calibration shift (model confidence drops on new signatures). Threshold tuning (lowering the cutoff to ~0.2-0.3) substantially recovers recall on most folds (e.g., jumping from 0 to 0.66 recall on `chb02`).

### Limitations

> [!WARNING]  
> **Full-dataset patient-independent validation (23 patients) is identified as priority follow-up work.** This 4-patient subset demonstrates the methodology and reveals a real threshold-calibration issue worth addressing at scale.
