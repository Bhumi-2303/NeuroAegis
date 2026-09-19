# NeuroAegis Protocol V1.0 — Cross-Domain Siena Generalization

| Transfer Cohort | Patients | Duration | AUROC | AUPRC | Window Sens | Window Spec | Event Sens | Mean Delay | Raw FP / 24h | Clinical FA / 24h |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Siena Zero-Shot (All 6 EDFs)** | PN00 + PN12 | 2.67h | 0.89934 | 0.70284 | 52.42% | 99.87% | 50.00% (3/6) | 14.33s | 44.94 | 8.99 |
| **Siena Calibration Split (Exp 6B)** | PN00 | 2.12h | 0.91259 | 0.77098 | 67.06% | 99.83% | 40.00% (2/5) | 17.50s | 56.60 | 11.32 |
| **Siena Held-Out Test Split (Exp 6B)** | PN12 | 0.55h | 0.90980 | 0.69320 | 20.51% | 100.00% | 100.00% (1/1) | 8.00s | 0.00 | 0.00 |