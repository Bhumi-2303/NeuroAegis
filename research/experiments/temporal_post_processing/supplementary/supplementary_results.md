# NeuroAegis: Supplementary Results & Experimental Records

This document provides extended experimental data, per-channel attributions, complete event concordance listings, and statistical distributions supporting the primary manuscript: **"Patient-Independent Electroencephalographic Seizure Detection Using Spatial Graph Convolutions and Causal Recurrent Neural Networks: Sensitivity, False Alarm Control, and Cross-Domain Evaluation"**.

---

## 1. Per-Patient Test Cohort Performance Breakdown

The held-out CHB-MIT test cohort comprises 4 pediatric patients evaluated over 155 continuous unsegmented EDF recordings totaling 152.82 hours (219,909 evaluation windows of 5.0 s with 50% temporal overlap).

| Patient ID | Record Count | Monitoring Hours | Seizures Total | Seizures Detected | Event Sensitivity | Window Sensitivity | Window Specificity | Precision | F1 Score | AUROC | AUPRC | False Alarms (FP Windows) | FA / 24h | Mean Delay (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **chb01** | 41 | 40.55 | 7 | 6 | 85.71% | 76.92% | 99.88% | 58.82% | 0.6667 | 0.9842 | 0.7712 | 70 | 41.43 | 8.83 |
| **chb02** | 35 | 35.28 | 3 | 3 | 100.0% | 88.89% | 99.79% | 51.61% | 0.6531 | 0.9915 | 0.8124 | 75 | 51.02 | 9.33 |
| **chb03** | 38 | 38.00 | 7 | 7 | 100.0% | 85.71% | 99.81% | 59.41% | 0.7018 | 0.9928 | 0.8340 | 82 | 51.79 | 10.14 |
| **chb05** | 41 | 38.99 | 5 | 5 | 100.0% | 83.82% | 99.69% | 58.76% | 0.6909 | 0.9903 | 0.8095 | 172 | 105.87 | 13.60 |
| **Overall Test Cohort** | **155** | **152.82** | **22** | **21** | **95.45%** | **83.83%** | **99.82%** | **57.24%** | **0.6803** | **0.9897** | **0.8068** | **399** | **62.66** | **10.57** |

*Note: In `chb01`, exactly one clinical seizure (`chb01_15`, duration 40s) was missed due to highly localized, low-amplitude rhythmic sharp-wave activity that failed to reach the sequential activation threshold of the causal GRU.*

---

## 2. Complete 22-Event Concordance & Detection Delay Comparison

Table S1 lists all 22 test seizure events and compares detection outcomes across Model A (1D-CNN baseline), Model B (CNN+GNN), and Model C (CNN+GNN+GRU).

| Event # | Patient | Recording | Seizure ID | Onset (s) | Offset (s) | Duration (s) | Model A Detected | Model A Delay (s) | Model B Detected | Model B Delay (s) | Model C Detected | Model C Delay (s) | Concordance Pattern |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | chb01 | chb01_03 | seizure_01 | 2996 | 3036 | 40 | False | - | False | - | **True** | 9.0 | A- B- C+ |
| 2 | chb01 | chb01_04 | seizure_01 | 1467 | 1494 | 27 | False | - | False | - | **True** | 8.0 | A- B- C+ |
| 3 | chb01 | chb01_15 | seizure_01 | 1732 | 1772 | 40 | False | - | False | - | **False** | - | A- B- C- (Missed) |
| 4 | chb01 | chb01_16 | seizure_01 | 1015 | 1066 | 51 | True | 12.0 | False | - | **True** | 10.0 | A+ B- C+ |
| 5 | chb01 | chb01_18 | seizure_01 | 1720 | 1810 | 90 | True | 15.0 | False | - | **True** | 8.0 | A+ B- C+ |
| 6 | chb01 | chb01_21 | seizure_01 | 327 | 420 | 93 | False | - | False | - | **True** | 9.0 | A- B- C+ |
| 7 | chb01 | chb01_26 | seizure_01 | 1862 | 1963 | 101 | True | 24.5 | False | - | **True** | 9.0 | A+ B- C+ |
| 8 | chb02 | chb02_16 | seizure_01 | 130 | 212 | 82 | True | 7.5 | True | 5.0 | **True** | 8.0 | A+ B+ C+ |
| 9 | chb02 | chb02_16+ | seizure_02 | 2972 | 3053 | 81 | False | - | False | - | **True** | 10.0 | A- B- C+ |
| 10 | chb02 | chb02_19 | seizure_01 | 3369 | 3447 | 78 | True | 10.0 | True | 7.5 | **True** | 10.0 | A+ B+ C+ |
| 11 | chb03 | chb03_01 | seizure_01 | 362 | 414 | 52 | False | - | False | - | **True** | 9.0 | A- B- C+ |
| 12 | chb03 | chb03_02 | seizure_01 | 731 | 796 | 65 | True | 8.0 | True | 8.0 | **True** | 9.0 | A+ B+ C+ |
| 13 | chb03 | chb03_03 | seizure_01 | 432 | 501 | 69 | True | 6.5 | True | 6.5 | **True** | 10.0 | A+ B+ C+ |
| 14 | chb03 | chb03_04 | seizure_01 | 2162 | 2214 | 52 | False | - | False | - | **True** | 9.0 | A- B- C+ |
| 15 | chb03 | chb03_34 | seizure_01 | 1982 | 2029 | 47 | False | - | False | - | **True** | 11.0 | A- B- C+ |
| 16 | chb03 | chb03_35 | seizure_01 | 259 | 323 | 64 | True | 8.0 | True | 7.5 | **True** | 9.0 | A+ B+ C+ |
| 17 | chb03 | chb03_36 | seizure_01 | 302 | 364 | 62 | True | 9.0 | True | 8.0 | **True** | 14.0 | A+ B+ C+ |
| 18 | chb05 | chb05_09 | seizure_01 | 2240 | 2355 | 115 | True | 8.5 | False | - | **True** | 12.0 | A+ B- C+ |
| 19 | chb05 | chb05_13 | seizure_01 | 1086 | 1196 | 110 | False | - | False | - | **True** | 14.0 | A- B- C+ |
| 20 | chb05 | chb05_16 | seizure_01 | 2342 | 2465 | 123 | True | 9.0 | False | - | **True** | 11.0 | A+ B- C+ |
| 21 | chb05 | chb05_17 | seizure_01 | 632 | 741 | 109 | True | 7.0 | False | - | **True** | 14.0 | A+ B- C+ |
| 22 | chb05 | chb05_22 | seizure_01 | 703 | 823 | 120 | False | - | False | - | **True** | 17.0 | A- B- C+ |

**Summary Statistics:**
- Model A Detected: 12 / 22 (54.55%), Mean Delay: 9.58 s, Median Delay: 8.5 s
- Model B Detected: 6 / 22 (27.27%), Mean Delay: 7.08 s, Median Delay: 7.5 s
- Model C Detected: **21 / 22 (95.45%)**, Mean Delay: **10.57 s**, Median Delay: **9.0 s**

---

## 3. Full 23-Channel Quantitative Attribution Ranking (Integrated Gradients)

Attributions were computed across all 637 true positive seizure windows using Integrated Gradients with 50 Riemann approximation steps against a zero baseline.

| Rank | Channel | Bipolar Pair | Mean Attribution | Std Dev | Normalized Score | Top-1 Freq (%) | Top-3 Freq (%) | Top-5 Freq (%) | Anatomic Region |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **T7-P7** | Temporal-Parietal (Left) | 1.6147 | 1.0340 | 0.0579 | 18.18% | 31.82% | 40.91% | Left Middle/Posterior Temporal |
| 2 | **P3-O1** | Parietal-Occipital (Left) | 1.6469 | 1.4023 | 0.0590 | 0.00% | 18.18% | 27.27% | Left Parietal-Occipital Junction |
| 3 | **FP1-F7** | Frontopolar-Temporal (Left) | 1.4921 | 0.8845 | 0.0535 | 9.09% | 22.73% | 36.36% | Left Anterior Temporal / Frontal |
| 4 | **F7-T7** | Frontal-Temporal (Left) | 1.4812 | 0.9123 | 0.0531 | 9.09% | 18.18% | 31.82% | Left Anterior-to-Mid Temporal |
| 5 | **T8-P8** | Temporal-Parietal (Right) | 1.4250 | 0.8654 | 0.0511 | 9.09% | 18.18% | 27.27% | Right Middle/Posterior Temporal |
| 6 | **FP2-F8** | Frontopolar-Temporal (Right) | 1.3980 | 0.8420 | 0.0501 | 4.55% | 13.64% | 22.73% | Right Anterior Temporal |
| 7 | **F8-T8** | Frontal-Temporal (Right) | 1.3854 | 0.8120 | 0.0496 | 4.55% | 13.64% | 22.73% | Right Anterior-to-Mid Temporal |
| 8 | **C3-P3** | Central-Parietal (Left) | 1.3210 | 0.7650 | 0.0473 | 4.55% | 9.09% | 18.18% | Left Centro-Parietal |
| 9 | **C4-P4** | Central-Parietal (Right) | 1.3050 | 0.7420 | 0.0468 | 4.55% | 9.09% | 18.18% | Right Centro-Parietal |
| 10 | **FZ-CZ** | Frontal-Central Midline | 1.2840 | 0.7100 | 0.0460 | 4.55% | 9.09% | 13.64% | Midline Frontal-Central |
| 11 | **CZ-PZ** | Central-Parietal Midline | 1.2610 | 0.6900 | 0.0452 | 4.55% | 4.55% | 13.64% | Midline Centro-Parietal |
| 12 | **P4-O2** | Parietal-Occipital (Right) | 1.2450 | 0.6850 | 0.0446 | 0.00% | 4.55% | 9.09% | Right Parietal-Occipital Junction |
| 13 | **P7-O1** | Parieto-Occipital (Left Lateral) | 1.2200 | 0.6700 | 0.0437 | 4.55% | 4.55% | 9.09% | Left Lateral Occipital |
| 14 | **P8-O2** | Parieto-Occipital (Right Lateral)| 1.2100 | 0.6650 | 0.0434 | 4.55% | 4.55% | 9.09% | Right Lateral Occipital |
| 15 | **F3-C3** | Frontal-Central (Left) | 1.1950 | 0.6400 | 0.0428 | 4.55% | 4.55% | 9.09% | Left Parasagittal Frontal |
| 16 | **F4-C4** | Frontal-Central (Right) | 1.1800 | 0.6350 | 0.0423 | 0.00% | 4.55% | 9.09% | Right Parasagittal Frontal |
| 17 | **FP1-F3** | Frontopolar-Frontal (Left) | 1.1650 | 0.6200 | 0.0417 | 4.55% | 4.55% | 4.55% | Left Frontopolar |
| 18 | **FP2-F4** | Frontopolar-Frontal (Right) | 1.1500 | 0.6100 | 0.0412 | 0.00% | 4.55% | 4.55% | Right Frontopolar |
| 19 | **P7-T7** | Alternate Montage Lead | 1.0800 | 0.5800 | 0.0387 | 0.00% | 0.00% | 4.55% | Left Temporal Re-reference |
| 20 | **T8-P8-alt**| Alternate Montage Lead | 1.0600 | 0.5700 | 0.0380 | 0.00% | 0.00% | 4.55% | Right Temporal Re-reference |
| 21 | **FT9-FT10**| Frontotemporal Transverse | 0.9500 | 0.5200 | 0.0340 | 0.00% | 0.00% | 0.00% | Anterior Transverse Lead |
| 22 | **T9-T10** | Mid-Temporal Transverse | 0.8900 | 0.4900 | 0.0319 | 0.00% | 0.00% | 0.00% | Middle Transverse Lead |
| 23 | **P9-P10** | Posterior Temporal Transverse | 0.8100 | 0.4500 | 0.0290 | 0.00% | 0.00% | 0.00% | Posterior Transverse Lead |

---

## 4. Siena Scalp EEG Target Domain Calibration Protocol

Target-domain post-hoc calibration was conducted on patient `PN00` (846 evaluation windows, 3 seizure events, 0.59h duration) to derive optimal temperature $T^*$ and decision threshold $\tau^*$ without altering any internal network weights.

```
Calibration Optimization:
    Minimize Negative Log-Likelihood (NLL) over PN00 logits:
    z_cal = z / T
    p_cal = sigmoid(z_cal)
    
    Optimal Parameters Derived on PN00:
    T* = 0.3495
    tau* = 0.3800
```

Evaluation on the unseen held-out test patient `PN12` (2,692 evaluation windows, 1 seizure event, 1.87h duration):
- **Zero-Shot Evaluation (Default: $T=1.0, \tau=0.50$):**
  - Event Sensitivity: 100% (1/1 detected)
  - Detection Delay: 19.0 s
  - Window Sensitivity: 17.95% (7 / 39)
  - Precision: 100.0% (7 / 7)
  - F1 Score: 0.3043
  - False Alarms / 24h: 0.0 (0 FP windows)
- **Adapted Evaluation (Calibrated: $T^*=0.3495, \tau^*=0.3800$):**
  - Event Sensitivity: 100% (1/1 detected)
  - Detection Delay: 19.0 s
  - Window Sensitivity: 23.08% (9 / 39)
  - Precision: 100.0% (9 / 9)
  - F1 Score: **0.3750** (+23.2% relative gain)
  - False Alarms / 24h: 0.0 (0 FP windows)
