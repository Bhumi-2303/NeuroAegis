# NeuroAegis Protocol V1.0 — EEG-Specific Deep Learning Architectures

| Model | Parameters | AUROC | AUPRC | Window Sens | Window Spec | Event Sens | Mean Delay | Raw FP / 24h | Clinical FA / 24h |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EEGNet** | 2,113 | 0.82506 | 0.04037 | 59.50% | 81.42% | 95.45% (21/22) | 3.02s | 6399.24 | 153.28 |
| **ShallowConvNet** | 41,041 | 0.84762 | 0.03687 | 54.32% | 87.59% | 95.45% (21/22) | 4.50s | 4274.28 | 275.77 |
| **DeepConvNet** | 178,776 | 0.85335 | 0.07248 | 47.72% | 89.91% | 63.64% (14/22) | 4.75s | 3476.02 | 258.65 |
| **Lightweight 1D CNN** | 173,601 | 0.87227 | 0.08721 | 45.05% | 93.77% | 68.18% (15/22) | 6.37s | 2147.11 | 224.73 |