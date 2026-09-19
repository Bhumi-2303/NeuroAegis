# NeuroAegis Protocol V1.0 — Spatial Representation Ablation

| Stage | Model | Parameters | AUROC | AUPRC | Window Sens | Window Spec | Event Sens | Mean Delay | Raw FP / 24h | Clinical FA / 24h |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN-only (1D CNN)** | CNN-only (1D CNN) | 173,601 | 0.36389 | 0.04148 | 16.01% | 94.35% | 40.91% (9/22) | 3.78s | 1946.56 | 228.03 |
| **CNN + Spatial GNN (θ=0.30)** | CNN + Spatial GNN (θ=0.30) | 52,497 | 0.19431 | 0.00492 | 4.24% | 99.42% | 22.73% (5/22) | 2.80s | 200.39 | 28.43 |
| **Model C (CNN + GNN + GRU)** | Model C (CNN + GNN + GRU) | 91,858 | 0.98970 | 0.80681 | 83.83% | 99.82% | 95.45% (21/22) | 6.05s | 62.66 | 7.38 |