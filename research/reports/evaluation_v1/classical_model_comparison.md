# NeuroAegis Protocol V1.0 — Classical Machine Learning Baselines

| Model | Features | AUROC | AUPRC | Window Sens | Window Spec | Event Sens | Mean Delay | Raw FP / 24h | Clinical FA / 24h |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (57 Features)** | 57 Engineered | 0.62465 | 0.00407 | 0.00% | 100.00% | 0.00% (0/22) | nans | 0.00 | 0.00 |
| **XGBoost (57 Features)** | 57 Engineered | 0.71778 | 0.00980 | 38.93% | 90.48% | 68.18% (15/22) | 6.03s | 3277.04 | 333.09 |
| **LightGBM (57 Features)** | 57 Engineered | 0.69973 | 0.00692 | 2.67% | 99.52% | 9.09% (2/22) | 21.75s | 166.62 | 14.29 |
| **Linear SVM (57 Features)** | 57 Engineered | 0.65417 | 0.01189 | 68.45% | 50.65% | 81.82% (18/22) | 1.31s | 16993.46 | 316.13 |