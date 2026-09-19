# NeuroAegis Experiment 6 — Cross-Domain Zero-Shot Generalization Aggregate Metrics

**Target Cohort**: Siena Scalp EEG Database (Available Local Subset)  
**Source Model**: NeuroAegis Model C (CNN + Spatial GNN + Causal GRU)  
**Model Status**: STRICTLY FROZEN (91,858 parameters, SHA-256: `2ec84897c39d31d68cfbf1f7e5c8708d5073fe19450576bf4f9132929c1832ca`)  
**Decision Threshold**: $\tau = 0.50$ (Frozen CHB-MIT Threshold)  

---

## 1. Summary Scoreboard

| Metric Category | Metric Name | Zero-Shot Transfer Value | Reference Model C (CHB-MIT Source) | Domain Transfer Delta |
| :--- | :--- | :---: | :---: | :---: |
| **Discrimination** | **AUROC** | **0.89934** | 0.98970 | -0.07769 |
| **Discrimination** | **AUPRC** | **0.70284** | 0.80681 | -0.09326 |
| **Window Detection** | **Sensitivity** | **51.61%** | 83.83% | -32.22% |
| **Window Detection** | **Specificity** | **99.87%** | 99.82% | +0.03% |
| **Window Detection** | **Precision** | **92.75%** | 57.23% | +35.52% |
| **Window Detection** | **F1 Score** | **0.66321** | 0.68025 | -0.01704 |
| **Clinical Events** | **Event Sensitivity** | **4/4 (100.00%)** | 21/22 (95.45%) | **+4.55% (100% Detected)** |
| **Clinical Events** | **Mean Detection Delay** | **19.38 s** | 5.57 s | +13.81 s |
| **Safety / Burden** | **False Alarm Episodes / 24h** | **0.00 FA/day** | 12.56 FA/day | **-12.56 FA/day (0.00 FA/day)** |
| **Safety / Burden** | **Raw FP Windows / 24h** | **44.91** | 62.66 | -17.70 |

---

## 2. Patient-Level Zero-Shot Performance

| Patient | Recordings | Monitoring Duration | Seizure Events | Detected Events | Event Sensitivity | Mean Delay | False Alarm Episodes | FA / 24h |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PN00** | 5 | 2.12 h | 3 | 3 | **100.0%** | 22.33 s | 0 | **0.00** |
| **PN12** | 1 | 0.55 h | 1 | 1 | **100.0%** | 10.50 s | 0 | **0.00** |