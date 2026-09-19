# NeuroAegis Protocol V1.0 — Master Model Comparison

**Evaluation Date**: 2026-09-19 08:39:23 UTC  
**Quarantined Test Cohort**: CHB-MIT (`chb01, chb02, chb03, chb05`), 155 EDFs, 152.82 hours, 22 seizures, 219,909 windows.  
**Standard**: Protocol V1.0 (Fixed threshold $\tau=0.50$, 3-window majority, 15s merge, 5s min duration).  

---

## Master Performance Table

| Model | Suite | Parameters | AUROC | AUPRC | Win Sens | Win Spec | Win F1 | Event Sens | Mean Onset Delay | Raw FP / 24h | Clinical FA / 24h |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN-only (1D CNN)** | Spatial Ablation | 173,601 | 0.36389 | 0.04148 | 16.01% | 94.35% | 0.01553 | 40.91% (9/22) | 3.78s | 1946.56 | 228.03 |
| **CNN + Spatial GNN (θ=0.30)** | Spatial Ablation | 52,497 | 0.19431 | 0.00492 | 4.24% | 99.42% | 0.02784 | 22.73% (5/22) | 2.80s | 200.39 | 28.43 |
| **Model C (CNN + GNN + GRU)** | Spatial Ablation | 91,858 | 0.98970 | 0.80681 | 83.83% | 99.82% | 0.68025 | 95.45% (21/22) | 6.05s | 62.66 | 7.38 |
| **Spatial + Causal GRU** | Temporal Comparison | 91,858 | 0.98340 | 0.75450 | 76.77% | 99.88% | 0.69907 | 95.45% (21/22) | 7.48s | 42.87 | 4.55 |
| **Spatial + Causal LSTM** | Temporal Comparison | 104,274 | 0.98641 | 0.74638 | 82.10% | 99.73% | 0.59398 | 95.45% (21/22) | 5.52s | 94.38 | 9.42 |
| **Spatial + Causal TCN** | Temporal Comparison | 112,914 | 0.98067 | 0.71582 | 87.60% | 99.64% | 0.56222 | 95.45% (21/22) | 4.74s | 124.07 | 11.78 |
| **EEGNet** | EEG-Specific Deep Learning | 2,113 | 0.82506 | 0.04037 | 59.50% | 81.42% | 0.01815 | 95.45% (21/22) | 3.02s | 6399.24 | 153.28 |
| **ShallowConvNet** | EEG-Specific Deep Learning | 41,041 | 0.84762 | 0.03687 | 54.32% | 87.59% | 0.02454 | 95.45% (21/22) | 4.50s | 4274.28 | 275.77 |
| **DeepConvNet** | EEG-Specific Deep Learning | 178,776 | 0.85335 | 0.07248 | 47.72% | 89.91% | 0.02635 | 63.64% (14/22) | 4.75s | 3476.02 | 258.65 |
| **Lightweight 1D CNN** | EEG-Specific Deep Learning | 173,601 | 0.87227 | 0.08721 | 45.05% | 93.77% | 0.03933 | 68.18% (15/22) | 6.37s | 2147.11 | 224.73 |
| **Random Forest (57 Features)** | Classical Machine Learning | Tabular | 0.62465 | 0.00407 | 0.00% | 100.00% | 0.00000 | 0.00% (0/22) | N/A | 0.00 | 0.00 |
| **XGBoost (57 Features)** | Classical Machine Learning | Tabular | 0.71778 | 0.00980 | 38.93% | 90.48% | 0.02280 | 68.18% (15/22) | 6.03s | 3277.04 | 333.09 |
| **LightGBM (57 Features)** | Classical Machine Learning | Tabular | 0.69973 | 0.00692 | 2.67% | 99.52% | 0.01983 | 9.09% (2/22) | 21.75s | 166.62 | 14.29 |
| **Linear SVM (57 Features)** | Classical Machine Learning | Tabular | 0.65417 | 0.01189 | 68.45% | 50.65% | 0.00798 | 81.82% (18/22) | 1.31s | 16993.46 | 316.13 |
| **BENDR Biosignal Transformer** | Pretrained Foundation Model | 2,467,521 | 0.58518 | 0.00460 | 100.00% | 0.00% | 0.00578 | 100.00% (22/22) | 0.00s | 34435.43 | Continuous Alert |