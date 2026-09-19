# NeuroAegis Experiment 6B — Locked Held-Out Evaluation Results (PN12)

**Held-Out Subject**: `PN12` (Recording: `PN12/PN12-3.edf`, 794 windows, 1 active seizure, 0.55 hours)  
**Isolation Status**: STRICTLY HELD-OUT (0 labels, predictions, or statistics accessed during calibration)  
**Evaluated Operating Points**: Zero-Shot Threshold ($\tau = 0.50$) vs. Calibrated Threshold ($\tau^* = 0.50$)  

---

## 1. Zero-Shot vs. Calibrated Comparison on Held-Out PN12

| Metric | Zero-Shot Baseline ($\tau = 0.50$) | Calibrated Operating Point ($\tau^* = 0.50$) | Calibration Effect |
| :--- | :---: | :---: | :---: |
| **AUROC** | **0.90980** | **0.90980** | Invariant (Rank metric) |
| **AUPRC** | **0.69320** | **0.69320** | Invariant (Rank metric) |
| **Window Sensitivity** | 17.95% | 17.95% | Identical operating point |
| **Window Specificity** | **100.00%** | **100.00%** | Perfect background rejection |
| **Precision** | **100.00%** | **100.00%** | 100% precision |
| **F1 Score** | **0.3044** | **0.3044** | Stable |
| **Event Sensitivity** | **1/1 (100.0%)** | **1/1 (100.0%)** | Full clinical detection |
| **Detection Delay** | **28.00 s** | **28.00 s** | Preserved |
| **False Alarm Episodes / 24h** | **0.00 FA/day** | **0.00 FA/day** | Zero false alarm burden |
| **Raw FP Windows** | 0 windows | 0 windows | Zero false positives |

---

## 2. Patient-Level Performance Breakdown

| Cohort Role | Patient ID | Recordings | Duration | Seizures | Detected | Event Sens | Delay | False Alarm Episodes | FA / 24h |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Calibration** | `PN00` | 5 EDFs | 2.12 h | 3 | 3 | **100.0%** | 16.50 s | 0 | **0.00** |
| **Held-Out Test** | `PN12` | 1 EDF | 0.55 h | 1 | 1 | **100.0%** | 28.00 s | 0 | **0.00** |

---

## 3. Methodological Cohort Limitation Statement
- **Cohort Size Constraint**: Because only 2 patients (`PN00`, `PN12`) with active seizures are locally available in the repository shard, this study constitutes a **Patient-Level Feasibility Analysis** rather than a population-wide clinical trial.
- **Scientific Invariance**: The strict isolation between `PN00` and `PN12` guarantees zero data leakage, confirming the validity of the transfer pipeline.