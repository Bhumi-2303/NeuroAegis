# NeuroAegis Research — Phase 2 Labeling Strategy Audit Report

## Final Pre-Model Audit & Frozen Window Labeling Protocol

**Author**: NeuroAegis Research Team  
**Date**: September 6, 2026  
**Status**: LABELING PROTOCOL FROZEN  
**Audit Execution Time**: 10.04 seconds  
**Automated Test Suite**: 12/12 PASS  

---

## 1. Executive Summary

This report documents the final pre-model statistical audit of the two candidate seizure-window labeling strategies for the CHB-MIT Scalp EEG Database. The audit was conducted **before any CNN training, model validation, or test-set evaluation**. The primary labeling strategy has been selected based exclusively on preprocessing-level coverage analysis, boundary contamination assessment, short-seizure preservation, and label quality metrics.

**Primary Labeling Protocol (FROZEN)**: **Strategy B (`label_50pct_overlap`)**  
- Mathematical Definition: $y_i = \mathbb{I}(\text{overlap\_ratio}_i \ge 0.50)$  
- A window is positive iff at least 50% of its duration (≥ 2.5s of 5.0s) contains clinically annotated seizure activity.  

**Secondary Sensitivity Benchmark**: **Strategy A (`label_any_overlap`)**  
- Mathematical Definition: $y_i = \mathbb{I}(\text{overlap\_duration\_sec}_i > 0.0)$  
- Retained in the master index for controlled downstream sensitivity and boundary-transition ablation experiments.

**Class Imbalance Decision 2**: FROZEN UNCHANGED  
- Dynamic Negative Subsampling (10:1) + Binary Focal Loss ($\gamma = 2.0, \alpha = 0.25$)

---

## 2. Dataset Used

| Parameter | Value |
|---|---|
| **Dataset** | CHB-MIT Scalp EEG Database |
| **Patients** | 24 pediatric epilepsy patients (chb01–chb24) |
| **EDF Recordings** | 686 |
| **Continuous EEG Duration** | ~982.9 hours |
| **Seizure Events** | 198 clinically annotated |
| **Sampling Frequency** | 256.0 Hz (uniform) |
| **Canonical Montage** | 23 bipolar channels (International 10-20) |
| **Window Duration** | 5.0 seconds (1,280 samples) |
| **Window Stride** | 2.5 seconds (640 samples, 50% overlap) |
| **Total Windows** | 1,414,710 |
| **Master Index** | `research/data/manifests/chbmit_window_index.csv` |
| **Master Index SHA256** | `f76dddb19fa2c24b3af24b989168b12465512b650c5e6d84bb0125a6d666302c` |

---

## 3. Strategy Definitions

### Strategy A: `label_any_overlap` (Any Overlap > 0s)

$$y_i^A = \begin{cases} 1 & \text{if } \text{overlap\_duration\_sec}_i > 0.0 \\ 0 & \text{otherwise} \end{cases}$$

A window is labeled positive if **any** fraction of its 5.0-second span intersects with a clinically annotated seizure event, even if the overlap is as small as 0.5 seconds (10% of the window).

### Strategy B: `label_50pct_overlap` (Overlap ≥ 50%)

$$y_i^B = \begin{cases} 1 & \text{if } \text{overlap\_ratio}_i \ge 0.50 \\ 0 & \text{otherwise} \end{cases}$$

A window is labeled positive only if **at least 50%** of its duration (≥ 2.5s) contains seizure activity. This ensures that the majority of the window's signal content represents ictal dynamics.

### Mathematical Relationship

Strategy B is a strict subset of Strategy A:

$$\{i : y_i^B = 1\} \subseteq \{i : y_i^A = 1\}$$

This means $y_i^A = 0 \implies y_i^B = 0$ for all windows $i$. The converse ($y_i^A = 0, y_i^B = 1$) is mathematically impossible and has been verified to hold across all 1,414,710 windows.

---

## 4. Global Comparison

| Metric | Strategy A | Strategy B | Difference (A − B) | % Difference |
|---|---:|---:|---:|---:|
| **Total Windows** | 1,414,710 | 1,414,710 | 0 | 0.00% |
| **Positive Windows** | 4,999 | 4,684 | +315 | +6.73% |
| **Negative Windows** | 1,409,711 | 1,410,026 | −315 | −0.02% |
| **Positive Percentage** | 0.35336% | 0.33109% | +0.02227% | +6.73% |
| **Negative Percentage** | 99.64664% | 99.66891% | −0.02227% | −0.02% |
| **Imbalance Ratio (Neg:Pos)** | 282.00 : 1 | 301.03 : 1 | −19.03 | −6.32% |

---

## 5. Patient-Level Comparison

| Patient | Total Windows | A Pos | A Pos% | B Pos | B Pos% | A−B Diff |
|:---:|---:|---:|---:|---:|---:|---:|
| chb01 | 58,353 | 190 | 0.326% | 180 | 0.308% | 10 |
| chb02 | 50,747 | 75 | 0.148% | 70 | 0.138% | 5 |
| chb03 | 54,684 | 176 | 0.322% | 163 | 0.298% | 13 |
| chb04 | 224,686 | 160 | 0.071% | 152 | 0.068% | 8 |
| chb05 | 56,125 | 233 | 0.415% | 224 | 0.399% | 9 |
| chb06 | 96,079 | 80 | 0.083% | 64 | 0.067% | 16 |
| chb07 | 96,535 | 135 | 0.140% | 131 | 0.136% | 4 |
| chb08 | 28,789 | 376 | 1.306% | 367 | 1.275% | 9 |
| chb09 | 97,713 | 117 | 0.120% | 111 | 0.114% | 6 |
| chb10 | 72,007 | 191 | 0.265% | 177 | 0.246% | 14 |
| chb11 | 50,067 | 327 | 0.653% | 323 | 0.645% | 4 |
| chb12 | 34,093 | 659 | 1.933% | 596 | 1.748% | 63 |
| chb13 | 47,487 | 238 | 0.501% | 216 | 0.455% | 22 |
| chb14 | 37,414 | 82 | 0.219% | 68 | 0.182% | 14 |
| chb15 | 57,573 | 832 | 1.445% | 801 | 1.391% | 31 |
| chb16 | 27,341 | 52 | 0.190% | 39 | 0.143% | 13 |
| chb17 | 30,228 | 122 | 0.404% | 118 | 0.390% | 4 |
| chb18 | 51,277 | 138 | 0.269% | 127 | 0.248% | 11 |
| chb19 | 43,067 | 100 | 0.232% | 95 | 0.221% | 5 |
| chb20 | 39,714 | 131 | 0.330% | 119 | 0.300% | 12 |
| chb21 | 47,242 | 86 | 0.182% | 79 | 0.167% | 7 |
| chb22 | 44,613 | 87 | 0.195% | 83 | 0.186% | 4 |
| chb23 | 38,232 | 180 | 0.471% | 173 | 0.453% | 7 |
| chb24 | 30,644 | 232 | 0.757% | 208 | 0.679% | 24 |
| **TOTAL** | **1,414,710** | **4,999** | **0.353%** | **4,684** | **0.331%** | **315** |

The largest absolute difference is observed in **chb12** (+63 boundary windows), followed by **chb15** (+31) and **chb24** (+24). The difference distribution is approximately proportional to seizure count and seizure duration per patient.

---

## 6. Recording-Level Comparison

- **Total recordings**: 686
- **Recordings where A > 0 but B == 0**: **0** (No recording loses all positive windows under Strategy B)
- **Top recordings with largest A−B differences**: Concentrated in recordings with multiple short seizures (chb12, chb15, chb24)

---

## 7. 198-Seizure Event Coverage

| Metric | Strategy A | Strategy B |
|---|---:|---:|
| **Events Covered** | 198 / 198 (100.0%) | 198 / 198 (100.0%) |
| **Events Uncovered** | 0 | 0 |
| **Mean Windows per Event** | 25.25 | 23.66 |
| **Median Windows per Event** | 20.0 | 18.0 |
| **Min Windows per Event** | 4 | 3 |
| **Max Windows per Event** | 303 | 301 |
| **Max Overlap Ratio per Event (Mean)** | 1.0000 | 1.0000 |
| **Avg Overlap Ratio per Event (Mean)** | 0.8837 | 0.8837 |

**Critical finding**: Both strategies achieve **100% event coverage** across all 198 seizure events. No clinically meaningful seizure event becomes invisible under either labeling rule. Every event, including the shortest 6.0-second focal seizures, has at least 3 positive windows under Strategy B.

---

## 8. Short-Seizure Analysis

### chb16_17.edf — Seizure 2 (1694.0s – 1700.0s, Duration = 6.0s)

| Window ID | Start | End | Overlap | Ratio | Strategy A | Strategy B |
|---|---:|---:|---:|---:|:---:|:---:|
| chb16_17_w00676 | 1690.0s | 1695.0s | 1.0s | 20% | **1** | 0 |
| chb16_17_w00677 | 1692.5s | 1697.5s | 3.5s | 70% | **1** | **1** |
| chb16_17_w00678 | 1695.0s | 1700.0s | 5.0s | 100% | **1** | **1** |
| chb16_17_w00679 | 1697.5s | 1702.5s | 2.5s | 50% | **1** | **1** |

### chb16_16.edf — Seizure 1 (1214.0s – 1220.0s, Duration = 6.0s)

| Window ID | Start | End | Overlap | Ratio | Strategy A | Strategy B |
|---|---:|---:|---:|---:|:---:|:---:|
| chb16_16_w00484 | 1210.0s | 1215.0s | 1.0s | 20% | **1** | 0 |
| chb16_16_w00485 | 1212.5s | 1217.5s | 3.5s | 70% | **1** | **1** |
| chb16_16_w00486 | 1215.0s | 1220.0s | 5.0s | 100% | **1** | **1** |
| chb16_16_w00487 | 1217.5s | 1222.5s | 2.5s | 50% | **1** | **1** |

**Key finding**: Under Strategy B, the shortest 6.0-second seizures retain **3 positive windows** each (with overlap ratios of 50%, 70%, and 100%). **No short seizure loses all positive windows under Strategy B.**

The boundary window excluded by Strategy B in each case has only 20% seizure content (1.0 second of ictal activity in a 5.0-second window), meaning 80% of its signal is non-seizure background EEG.

---

## 9. Boundary-Window Analysis

| Overlap Ratio Category | Count | % of A-Positives | Strategy A | Strategy B |
|---|---:|---:|:---:|:---:|
| 0.00 < ratio < 0.10 | 0 | 0.00% | Positive | Negative |
| 0.10 ≤ ratio < 0.25 | 156 | 3.12% | Positive | Negative |
| 0.25 ≤ ratio < 0.50 | 159 | 3.18% | Positive | Negative |
| 0.50 ≤ ratio < 0.75 | 237 | 4.74% | Positive | Positive |
| 0.75 ≤ ratio < 1.00 | 159 | 3.18% | Positive | Positive |
| ratio == 1.00 | 4,288 | 85.78% | Positive | Positive |

**Key findings**:
- **85.78%** of Strategy A positive windows contain **100% pure seizure signal** (fully enclosed within a seizure event).
- **315 windows** (6.30% of A-positives) are boundary windows with < 50% seizure content. These are transition windows at seizure onset/offset boundaries.
- Of those 315 boundary windows: 0 have < 10% overlap, 156 have 10–25% overlap, and 159 have 25–50% overlap.
- Strategy B eliminates these 315 windows, increasing the proportion of pure ictal content from 85.78% to 91.55% among positive windows.

---

## 10. Label Disagreement

| Condition | Count | % of Total | % of A-Positives |
|---|---:|---:|---:|
| **A=0, B=0** (Background Agreement) | 1,409,711 | 99.64664% | — |
| **A=1, B=0** (Boundary Disagreement) | 315 | 0.02227% | 6.30% |
| **A=1, B=1** (Core Ictal Agreement) | 4,684 | 0.33109% | 93.70% |
| **A=0, B=1** (Impossible Inversion) | 0 | 0.00000% | 0.00% |

The subset invariant ($B \subseteq A$) holds perfectly: zero windows exist where Strategy A assigns negative and Strategy B assigns positive. The 315 disagreement windows represent transition boundaries that Strategy A includes but Strategy B excludes.

---

## 11. Class-Imbalance Impact

| Dimension | Strategy A | Strategy B |
|---|---:|---:|
| Raw Positive Pool | 4,999 | 4,684 |
| Raw Negative Pool | 1,409,711 | 1,410,026 |
| Imbalance Ratio | 282.00 : 1 | 301.03 : 1 |
| 10:1 Sampled Negatives per Epoch | 49,990 | 46,840 |
| 10:1 Total Training Windows per Epoch | 54,989 | 51,524 |
| Boundary Noise in Positive Pool | 6.30% (315 windows) | 0.00% (0 windows) |

The frozen 10:1 Dynamic Negative Subsampling protocol remains unchanged. Strategy B produces a slightly smaller but cleaner positive pool. With Strategy B, every positive gradient update during training is driven by a window containing ≥ 50% true seizure signal, eliminating misleading updates from noise-dominant boundary windows.

---

## 12. Methodological Trade-offs

### Strategy A Advantages
- **More positive training examples** (4,999 vs 4,684): +315 windows (+6.73%)
- **Greater sensitivity to seizure boundaries**: Captures transition zones at onset/offset
- **Marginally lower imbalance ratio** (282:1 vs 301:1)

### Strategy A Disadvantages
- **315 windows (6.30%) contain 60–90% non-seizure background**: These boundary windows may produce ambiguous or misleading gradient signals during training
- **Signal contamination risk**: A model trained on these windows may learn to associate mixed seizure/background patterns as positive, potentially increasing false alarms

### Strategy B Advantages
- **100% clean positive pool**: Every positive window has ≥ 2.5s of seizure content (≥ 50%)
- **91.55% of positive windows contain 100% pure ictal signal**
- **Eliminates boundary contamination**: 0 windows with < 50% seizure content in the positive class
- **100% event coverage maintained**: No seizure event loses all representation

### Strategy B Disadvantages
- **Fewer positive training examples**: 4,684 vs 4,999 (−6.73%)
- **Slightly higher imbalance ratio**: 301:1 vs 282:1
- **Reduced boundary transition representation**: Onset/offset windows excluded from positive class

### The actual data resolves the trade-off
- Strategy B does **not** lose any seizure event. All 198 events remain covered with ≥ 3 positive windows.
- The 315 excluded windows contain on average only 27% seizure signal — predominantly background EEG.
- The frozen 10:1 sampler already aggressively undersamples negatives, so the 6.7% reduction in positives has minimal effect on epoch size (54,989 → 51,524 windows).

---

## 13. Primary Strategy Recommendation

**PRIMARY LABEL (FROZEN)**: **Strategy B (`label_50pct_overlap`)**

Selected as the primary frozen labeling protocol based on preprocessing-level coverage and label-quality analysis:

1. **Priority 1 (Event Coverage)**: SATISFIED. Strategy B achieves 100% seizure event coverage (198/198). No clinically meaningful seizure event becomes invisible.

2. **Priority 2 (Boundary Contamination)**: SATISFIED. Strategy B eliminates all 315 contaminated boundary windows where 60–90% of the signal is non-seizure background.

3. **Priority 3 (Seizure-Window Representation)**: SATISFIED. Strategy B provides 4,684 positive windows with ≥ 3 windows per event, including the shortest 6.0-second seizures.

4. **Priority 4 (Class Imbalance)**: ACCEPTABLE. The 301:1 imbalance ratio is within acceptable bounds for the frozen 10:1 dynamic subsampler and Binary Focal Loss.

5. **Priority 5 (Reproducibility)**: SATISFIED. The definition is simple, deterministic, and reproducible: $y_i = \mathbb{I}(\text{overlap\_ratio}_i \ge 0.50)$.

**SECONDARY SENSITIVITY BENCHMARK**: **Strategy A (`label_any_overlap`)**

Retained in the master window index (`chbmit_window_index.csv`) for controlled downstream experiments. Strategy A can be used as a sensitivity analysis benchmark to evaluate whether including boundary-transition windows improves onset detection, at the cost of boundary contamination.

This selection does **not** claim that Strategy B is optimal. It is selected as the primary protocol based on preprocessing-level coverage and label-quality analysis, **prior to any CNN training or performance evaluation**.

---

## 14. Frozen Protocol

The frozen labeling protocol is recorded in `research/phase_2/labeling_protocol.json`:

```json
{
  "primary_label_strategy": "Strategy B (label_50pct_overlap)",
  "definition": "Positive (1) iff overlap_ratio >= 0.50; Negative (0) otherwise.",
  "overlap_threshold": 0.50,
  "window_duration_sec": 5.0,
  "window_stride_sec": 2.5,
  "sampling_frequency_hz": 256.0,
  "window_samples": 1280,
  "stride_samples": 640,
  "dataset": "CHB-MIT",
  "decision_date": "2026-09-06",
  "git_commit": "17943cdaccfa1d6857f787b91e53b223dbbb8616",
  "master_index_sha256": "f76dddb19fa2c24b3af24b989168b12465512b650c5e6d84bb0125a6d666302c"
}
```

---

## 15. Reproducibility Information

| Parameter | Value |
|---|---|
| **Experiment ID** | Phase 2 Labeling Strategy Audit |
| **Execution Timestamp** | 2026-09-06T06:57Z (UTC) |
| **Operating System** | macOS 26.6.2 (arm64) |
| **Python Version** | 3.11.15 |
| **NumPy Version** | 1.26.4 |
| **Pandas Version** | 2.2.1 |
| **OpenPyXL Version** | 3.1.5 |
| **Matplotlib Version** | 3.11.1 |
| **Git Commit** | 17943cdaccfa1d6857f787b91e53b223dbbb8616 |
| **Master Index SHA256** | f76dddb19fa2c24b3af24b989168b12465512b650c5e6d84bb0125a6d666302c |
| **Random Seed** | N/A (deterministic audit, no random sampling) |
| **Configuration Hash** | Derived from preprocessing_config.json |

---

## 16. Automated Test Results

| Test # | Test Description | Result |
|:---:|---|:---:|
| 1 | Strategy A definition is correct | **PASS** |
| 2 | Strategy B definition is correct | **PASS** |
| 3 | Strategy B positive set is a subset of Strategy A | **PASS** |
| 4 | No A=0/B=1 windows exist | **PASS** |
| 5 | All 198 seizure events are included | **PASS** |
| 6 | Short seizure events correctly mapped and covered | **PASS** |
| 7 | Overlap ratios in [0, 1] | **PASS** |
| 8 | No duplicate window IDs | **PASS** |
| 9 | Master index unchanged (SHA256) | **PASS** |
| 10 | Primary strategy from frozen protocol file | **PASS** |
| 11 | No model training executed | **PASS** |
| 12 | No model performance in decision | **PASS** |

**Total: 12/12 PASS**

---

## 17. Limitations

1. **No clinical validation**: The labeling strategy was selected based on statistical properties, not clinical outcome. Clinical validation requires downstream CNN evaluation and expert review.

2. **Boundary semantics are debatable**: Whether a window with 20% seizure content should be labeled positive is a research question that cannot be fully resolved without evaluating detection performance. Strategy A's boundary windows may contain valuable seizure-onset signatures.

3. **Short-seizure edge case**: Seizures shorter than 5.0 seconds (the window duration) would always produce boundary windows. The minimum observed seizure duration in CHB-MIT is 6.0 seconds, which is adequately covered. Datasets with shorter seizures may require different labeling thresholds.

4. **Annotation granularity**: CHB-MIT seizure annotations are provided at 1-second resolution. Sub-second onset timing variations could affect boundary window overlap calculations.

5. **Strategy sensitivity**: The choice between Strategy A and Strategy B may affect downstream model sensitivity vs. specificity trade-offs. This is acknowledged and documented through the dual-retention design.

---

*Phase 2 Labeling Strategy Audit is formally complete. The primary labeling protocol is frozen as Strategy B (`label_50pct_overlap`) effective 2026-09-06. Proceeding to CNN baseline training requires user instruction.*
