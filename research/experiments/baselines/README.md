# Experiment 3: Classical Machine Learning Baselines

Comprehensive benchmark comparing classical machine learning architectures trained on 57 multi-domain engineered EEG features against frozen **Model C** (CNN + Spatial GNN + Causal GRU).

## Evaluated Models
1. **Random Forest** (`RandomForestClassifier`)
2. **XGBoost** (`XGBClassifier`)
3. **LightGBM** (`LGBMClassifier`)
4. **Linear SVM** (`SGDClassifier` with L2 regularization)
5. **Model C (Frozen Benchmark)**: CNN-GNN-GRU end-to-end spatio-temporal network

## Dataset Protocol & Split
- **Train (16 patients, 10:1 subsampled)**: 36,388 windows
- **Validation (4 patients)**: 293,410 windows
- **Test (4 held-out patients, locked)**: 219,909 windows, 22 seizures, 152.82 continuous hours

## Comparison Summary

| Model | Status | AUROC | AUPRC | Sens (%) | Spec (%) | F1 | Event Sens (%) | Delay (s) | FA/24h |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model C (CNN + GNN + GRU)** | FROZEN REFERENCE ONLY | 0.9897 | 0.8068 | 83.83% | 99.82% | 0.6803 | 95.45% | 10.57s | 62.66 |
| **Random Forest** | EVALUATED BASELINE | 0.6247 | 0.0041 | 1.88% | 98.66% | 0.0067 | 0.00% | 0.00s | 80.57 |
| **XGBoost** | EVALUATED BASELINE | 0.7178 | 0.0098 | 1.41% | 99.76% | 0.0154 | 0.00% | 0.00s | 11.78 |
| **LightGBM** | EVALUATED BASELINE | 0.6997 | 0.0069 | 6.91% | 98.75% | 0.0257 | 4.54% | 30.00s | 54.02 |
| **Linear SVM** | EVALUATED BASELINE | 0.6542 | 0.0119 | 72.53% | 44.10% | 0.0075 | 86.36% | 1.32s | 516.06 |
