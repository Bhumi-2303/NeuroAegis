# NeuroAegis — Final Model C Event-Level Forensic Audit
**Pre-Protocol-V1 Freeze Validation & Authoritative 22-Seizure Event Table**

**Date**: September 19, 2026  
**Auditor**: Lead Clinical Event Audit Engineer  
**Status**: **PROTOCOL V1.0 FULLY FROZEN**  
**Target Hardware**: Apple Silicon M4 (16 GB Unified RAM, macOS, PyTorch MPS)  

---

## 1. Executive Summary

This forensic audit resolves the final remaining numerical question prior to declaring Evaluation Protocol V1.0 permanently frozen:
1. **Mathematical Resolution of 5.57s vs. 6.05s Delay**:
   - **$5.57\text{ seconds}$** is the **Raw Unfiltered Window Onset Delay** ($\text{first raw window with } p \ge 0.50 - \text{seizure onset}$).
   - **$6.05\text{ seconds}$** is the **Authoritative Protocol V1 Post-Processed Alarm Episode Onset Delay** ($\text{first sustained alarm episode start} - \text{seizure onset}$).
   - The $0.48\text{s}$ difference is traced to exactly **one event** (Event 4, `chb01_16`), where a single isolated positive window at $2.5\text{s}$ was correctly filtered out as transient chatter by the 3-window majority filter, causing the sustained clinical alarm episode to begin $10.0\text{s}$ later ($12.5\text{s}$ delay). For all other 20 detected seizures, raw window onset and alarm episode onset are **100% identical**.
2. **Authoritative 22-Seizure Event Table**: Generated [`model_c_event_audit.csv`](file:///Volumes/BLACK-BOX/NeuroAegis/research/audits/evaluation_protocol/model_c_event_audit.csv) containing all 22 clinical seizures with complete window-level probabilities, alarm episode IDs, onset delays, and duration metrics.
3. **Definitive Missed Seizure Identification**: Confirmed that the single missed seizure occurred in recording **`chb01_15`** (patient `chb01`, $1,732\text{s} - 1,772\text{s}$, peak $p = 0.4813$), **not `chb02_16`**.
4. **Final Freeze Declaration**: All 15 models, 12 synthetic unit tests, and cross-domain benchmarks have been validated without a single unresolved issue. **Evaluation Protocol V1.0 is hereby declared FULLY FROZEN**.

---

## 2. Forensic Investigation: 5.57s vs. 6.05s Delay

A step-by-step trace of all 22 clinical seizures was performed across the locked CHB-MIT test cohort:

| Event Index | Patient | Recording ID | Seizure Start | Seizure End | Max Prob | Raw First Pos Win Start | Raw Onset Delay | Protocol V1 Alarm Start | Protocol V1 Onset Delay | Delay Discrepancy ($\Delta$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `chb01` | `chb01_03` | 2,996.0s | 3,036.0s | 0.6752 | 3,000.0s | 4.0s | 3,000.0s | 4.0s | 0.0s |
| 2 | `chb01` | `chb01_04` | 1,467.0s | 1,494.0s | 0.6716 | 1,470.0s | 3.0s | 1,470.0s | 3.0s | 0.0s |
| 3 | `chb01` | `chb01_15` | 1,732.0s | 1,772.0s | 0.4813 | None | None | None | None | *(Missed)* |
| **4** | **`chb01`** | **`chb01_16`** | **1,015.0s** | **1,066.0s** | **0.6427** | **1,017.5s** | **2.5s** | **1,027.5s** | **12.5s** | **+10.0s** |
| 5 | `chb01` | `chb01_18` | 1,720.0s | 1,810.0s | 0.7077 | 1,725.0s | 5.0s | 1,725.0s | 5.0s | 0.0s |
| 6 | `chb01` | `chb01_21` | 327.0s | 420.0s | 0.7067 | 330.0s | 3.0s | 330.0s | 3.0s | 0.0s |
| 7 | `chb01` | `chb01_26` | 1,862.0s | 1,963.0s | 0.7325 | 1,870.0s | 8.0s | 1,870.0s | 8.0s | 0.0s |
| 8 | `chb02` | `chb02_16` | 130.0s | 212.0s | 0.7538 | 137.5s | 7.5s | 137.5s | 7.5s | 0.0s |
| 9 | `chb02` | `chb02_16+` | 2,972.0s | 3,053.0s | 0.7559 | 2,977.5s | 5.5s | 2,977.5s | 5.5s | 0.0s |
| 10 | `chb02` | `chb02_19` | 3,369.0s | 3,378.0s | 0.6066 | 3,370.0s | 1.0s | 3,370.0s | 1.0s | 0.0s |
| 11 | `chb03` | `chb03_01` | 362.0s | 414.0s | 0.6427 | 365.0s | 3.0s | 365.0s | 3.0s | 0.0s |
| 12 | `chb03` | `chb03_02` | 731.0s | 796.0s | 0.6685 | 735.0s | 4.0s | 735.0s | 4.0s | 0.0s |
| 13 | `chb03` | `chb03_03` | 432.0s | 501.0s | 0.6304 | 432.5s | 0.5s | 432.5s | 0.5s | 0.0s |
| 14 | `chb03` | `chb03_04` | 2,162.0s | 2,214.0s | 0.6518 | 2,165.0s | 3.0s | 2,165.0s | 3.0s | 0.0s |
| 15 | `chb03` | `chb03_34` | 1,982.0s | 2,029.0s | 0.6165 | 1,997.5s | 15.5s | 1,997.5s | 15.5s | 0.0s |
| 16 | `chb03` | `chb03_35` | 2,592.0s | 2,656.0s | 0.6569 | 2,595.0s | 3.0s | 2,595.0s | 3.0s | 0.0s |
| 17 | `chb03` | `chb03_36` | 1,725.0s | 1,778.0s | 0.6766 | 1,730.0s | 5.0s | 1,730.0s | 5.0s | 0.0s |
| 18 | `chb05` | `chb05_06` | 417.0s | 532.0s | 0.8093 | 422.5s | 5.5s | 422.5s | 5.5s | 0.0s |
| 19 | `chb05` | `chb05_13` | 1,086.0s | 1,196.0s | 0.7337 | 1,092.5s | 6.5s | 1,092.5s | 6.5s | 0.0s |
| 20 | `chb05` | `chb05_16` | 2,317.0s | 2,413.0s | 0.7503 | 2,320.0s | 3.0s | 2,320.0s | 3.0s | 0.0s |
| 21 | `chb05` | `chb05_17` | 2,451.0s | 2,571.0s | 0.7749 | 2,455.0s | 4.0s | 2,455.0s | 4.0s | 0.0s |
| 22 | `chb05` | `chb05_22` | 2,348.0s | 2,465.0s | 0.7825 | 2,372.5s | 24.5s | 2,372.5s | 24.5s | 0.0s |

---

## 3. Mathematical Proof of Root Cause

1. **Raw Unfiltered Window Delay Sum**:
   $$\sum_{j=1}^{21} \delta_j^{\text{raw}} = 117.0\text{ seconds} \implies \bar{\delta}^{\text{raw}} = \frac{117.0}{21} = \mathbf{5.5714\text{ seconds}} \approx \mathbf{5.57\text{ s}}$$
2. **Protocol V1 Alarm Episode Delay Sum**:
   $$\sum_{j=1}^{21} \delta_j^{\text{alarm}} = 117.0 + 10.0 = 127.0\text{ seconds} \implies \bar{\delta}^{\text{alarm}} = \frac{127.0}{21} = \mathbf{6.0476\text{ seconds}} \approx \mathbf{6.05\text{ s}}$$

### Standardization Decision
- **Authoritative Primary Metric**: **$6.05\text{ seconds}$ (Protocol V1 Post-Processed Alarm Episode Onset Delay)**. This metric measures the exact latency at which a real-world clinical alarm is sounded following 3-window majority filtering.
- **Secondary Unfiltered Reference**: **$5.57\text{ seconds}$ (Raw Window Onset Delay)**.
- **Historical Completion Reference**: **$10.57\text{ seconds}$ (Buffer Completion Delay, $5.57\text{s} + 5.0\text{s}$)**.

---

## 4. Authoritative 22-Seizure Event Table Summary

The table [`model_c_event_audit.csv`](file:///Volumes/BLACK-BOX/NeuroAegis/research/audits/evaluation_protocol/model_c_event_audit.csv) provides complete per-event metadata:
- **Total Seizures Evaluated**: 22 events across 155 EDF files ($152.8231\text{ hours}$).
- **Detected Events**: 21 events (**95.45% event sensitivity**).
- **Missed Events**: 1 event (`chb01_15`).
- **Mean Post-Processed Onset Delay**: **$6.05\text{ seconds}$** (Median: $4.00\text{s}$, Min: $0.50\text{s}$, Max: $24.50\text{s}$, Std: $5.37\text{s}$).

---

## 5. Missed Seizure Analysis (`chb01_15`)

- **Recording**: `chb01/chb01_15.edf` (Seizure onset $1,732.0\text{s}$, offset $1,772.0\text{s}$, duration $40.0\text{s}$).
- **Maximum Probability**: **$0.481285$** (peaking in window $[1,737.5\text{s}, 1,742.5\text{s}]$).
- **Cause**: Electrographic manifestation peaked $0.0187$ below the locked decision boundary $\tau = 0.50$.
- **Detailed Forensic Report**: Saved in [`model_c_missed_event_analysis.md`](file:///Volumes/BLACK-BOX/NeuroAegis/research/audits/evaluation_protocol/model_c_missed_event_analysis.md).

---

## 6. Alarm Audit Summary (`model_c_alarm_audit.csv`)

Generated in [`model_c_alarm_audit.csv`](file:///Volumes/BLACK-BOX/NeuroAegis/research/audits/evaluation_protocol/model_c_alarm_audit.csv):
- **Total Alarm Episodes Formed**: 68 episodes across 155 EDF files.
- **True Seizure Alarms**: 21 episodes (matching all 21 detected seizures).
- **Clinical False Alarm Episodes**: **47 episodes** (unmatched alarms outside seizure intervals).
- **Clinical False-Alarm Rate**:
  $$R_{\text{clinical\_FA}} = \frac{47}{152.8231\text{ hours}} \times 24.0 = \mathbf{7.38\text{ Clinical FA episodes / 24h}}$$

---

## 7. Raw False Positives vs. Clinical False Alarms

- **Raw False-Positive Windows**: **399 windows** ($99.82\%$ background specificity).
- **Raw False-Positive Rate**:
  $$R_{\text{raw\_FP}} = \frac{399}{152.8231\text{ hours}} \times 24.0 = \mathbf{62.66\text{ Raw FP windows / 24h}}$$
- **False-Alarm Reduction**: Protocol V1 post-processing suppresses isolated window chatter, cutting the actionable clinical alarm burden by **88.2%** (from $62.66$ raw window alarms down to $7.38$ clinical episodes per day).

---

## 8. Cross-Suite Delay Consistency Check

All 15 models were verified to ensure identical calculation:
- **Delay Timestamp Standard**: `delay_timestamp_definition == "alarm_onset"` verified across all 15 models.
- **All Models Re-Evaluated**: Verified in [`master_model_comparison.csv`](file:///Volumes/BLACK-BOX/NeuroAegis/research/reports/evaluation_v1/master_model_comparison.csv).

---

## 9. Final Authoritative Model C Benchmark Scoreboard

```
========================================================================================
                      NEUROAEGIS MODEL C AUTHORITATIVE BENCHMARK
========================================================================================
Architecture:         CNN -> Spatial GNN (theta=0.30) -> Causal GRU (L=8, hidden=64)
Parameters:           91,858
Test Cohort:          CHB-MIT Locked Split (155 EDFs, 152.8231h, 219,909 windows, 22 sz)
Decision Threshold:   tau = 0.50 (Frozen)
----------------------------------------------------------------------------------------
Discrimination:       AUROC = 0.98970 | AUPRC = 0.80681
Window Level:         Sensitivity = 83.83% | Specificity = 99.82% | F1 = 0.68025
Clinical Events:      Event Sensitivity = 21/22 (95.45%) | Missed Seizures = 1 (chb01_15)
Detection Latency:    Mean Onset Delay = 6.05 s | Median Onset Delay = 4.00 s
                      (Raw Window Onset Delay = 5.57 s | Completion Delay = 10.57 s)
Safety Burden:        Clinical FA Episodes = 7.38 FA / 24h (47 episodes)
                      Raw FP Windows = 62.66 FP / 24h (399 windows)
Hardware Speed:       Inference Latency = 0.010 ms / window (PyTorch MPS, Apple M4)
========================================================================================
```

---

## 10. Official Protocol V1.0 Freeze Declaration

All forensic audits, unit tests, and cross-model reconciliations have completed with 100% mathematical consistency.

$$\mathbf{EVALUATION\ PROTOCOL\ V1.0\ = \ FULLY\ FROZEN}$$
