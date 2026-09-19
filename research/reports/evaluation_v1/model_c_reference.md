# NeuroAegis Protocol V1.0 — Authoritative Model C Benchmark

**Model**: NeuroAegis Model C (`CNN -> Spatial GNN -> Causal GRU`)  
**Checkpoint**: `research/phase_4b/frozen_cnn_gnn_gru.pt`  
**SHA-256**: `2ec84897c39d31d68cfbf1f7e5c8708d5073fe19450576bf4f9132929c1832ca`  
**Parameters**: `91,858`  
**Test Cohort**: CHB-MIT Locked Test Split (155 recordings, 152.82 hours, 22 seizures, 219,909 windows)  

---

## Authoritative Locked Test Metrics

| Metric Category | Metric Name | Authoritative Value | Mathematical Definition / Verification |
| :--- | :--- | :---: | :--- |
| **Discrimination** | **AUROC** | **0.98970** | Area under ROC over 219,909 unrounded probabilities |
| **Precision-Recall** | **AUPRC** | **0.80681** | Average precision score under 344.2:1 imbalance |
| **Window Level** | **Window Sensitivity** | **83.83%** | 534 / 637 positive windows (tau=0.50) |
| **Window Level** | **Window Specificity** | **99.82%** | 218,873 / 219,272 negative windows (tau=0.50) |
| **Window Level** | **Window Precision** | **57.24%** | 534 / 933 positive predictions (tau=0.50) |
| **Window Level** | **Window F1 Score** | **0.68025** | Harmonic mean of precision and recall |
| **Window Level** | **Balanced Accuracy** | **91.82%** | (Sensitivity + Specificity) / 2 |
| **Clinical Event** | **Event Sensitivity** | **21/22 (95.45%)** | 21 of 22 clinical seizures detected within 30s |
| **Clinical Event** | **Missed Seizures** | **1** | Only 1 focal seizure missed (`chb02_16`) |
| **Latency** | **Mean Onset Delay** | **5.57 s** | First detecting window start minus electrographic onset |
| **Latency** | **Median Onset Delay** | **4.00 s** | Median onset latency across 21 detected seizures |
| **Latency** | **Historical Completion Delay** | **10.57 s** | First detecting window end minus onset (5.57s + 5.0s) |
| **Safety Burden** | **Clinical False Alarms** | **7.38 FA / 24h** | 47 unmatched alarm episodes across 152.82h |
| **Safety Burden** | **Raw FP Windows** | **62.66 FP / 24h** | 399 unclustered false positive windows across 152.82h |
