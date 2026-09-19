# NeuroAegis Protocol V1.0 — Temporal Sequence Model Comparison

| Architecture | Parameters | AUROC | AUPRC | Window Sens | Window Spec | Window F1 | Event Sens | Mean Delay | Raw FP / 24h | Clinical FA / 24h |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spatial + Causal GRU** | 91,858 | 0.98340 | 0.75450 | 76.77% | 99.88% | 0.69907 | 95.45% (21/22) | 7.48s | 42.87 | 4.55 |
| **Spatial + Causal LSTM** | 104,274 | 0.98641 | 0.74638 | 82.10% | 99.73% | 0.59398 | 95.45% (21/22) | 5.52s | 94.38 | 9.42 |
| **Spatial + Causal TCN** | 112,914 | 0.98067 | 0.71582 | 87.60% | 99.64% | 0.56222 | 95.45% (21/22) | 4.74s | 124.07 | 11.78 |