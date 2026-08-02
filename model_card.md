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
