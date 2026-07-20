
# NeuroAegis Final Experiment Report

## Dataset

Bonn University EEG Dataset

- Samples: 500
- Classes: 5
- Selected Features: 56

---

## Best Model

LightGBM

Test Accuracy: **90.00%**

---

## Top SHAP Features

| Feature                   |   Mean_SHAP |
|:--------------------------|------------:|
| relative_beta             |    0.521781 |
| relative_alpha            |    0.361351 |
| hjorth_mobility           |    0.329709 |
| theta_power               |    0.257535 |
| wavelet_relative_energy_2 |    0.231548 |
| hjorth_complexity         |    0.202722 |
| alpha_power               |    0.198469 |
| wavelet_energy_3          |    0.198232 |
| wavelet_energy_1          |    0.17843  |
| skewness                  |    0.165315 |

---

## Pipeline

Dataset
→ Feature Engineering
→ Feature Selection
→ Train/Test Split
→ Random Forest
→ XGBoost
→ LightGBM
→ Hyperparameter Optimization
→ SHAP Explainability
→ Final Model Training
→ Deployment Pipeline

---

## Deployment Model

final_lightgbm_full_dataset.pkl

