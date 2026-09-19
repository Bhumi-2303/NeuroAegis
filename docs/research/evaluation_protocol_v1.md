# NeuroAegis — Evaluation Protocol V1.0 Specification

**Version**: 1.0 (Frozen)  
**Effective Date**: September 19, 2026  
**Status**: Authoritative Reference Standard for NeuroAegis  

---

## 1. Scope & Purpose

This document codifies the unified scientific evaluation protocol for the NeuroAegis epileptic seizure detection system. It defines the exact mathematical, clinical, and data-handling rules for assessing window-level classification, post-processed clinical alarm episodes, detection latency, and domain generalization.

---

## 2. Core Protocol Parameters

### A. Windowing & Ground Truth
- **Window Length ($W$)**: $5.0\text{ seconds}$ ($1,280\text{ samples}$ at $256\text{ Hz}$).
- **Window Stride ($S$)**: $2.5\text{ seconds}$ ($640\text{ samples}$ at $256\text{ Hz}$).
- **Overlap**: $50\%$ temporal overlap.
- **Labeling Rule (`label_50pct_overlap`)**: A window is labeled positive ($y=1$) if and only if $\ge 50\%$ of its duration ($2.5\text{s}$) overlaps an annotated seizure event.

### B. Decision Threshold
- **Default Fixed Threshold**: $\tau = 0.50$.
- **Rule**: Thresholds must never be selected using test cohort performance. In cross-domain transfer, threshold optimization must be restricted to a designated calibration split (e.g., `PN00` in Exp 6B), leaving held-out splits (`PN12`) strictly locked.

### C. Alarm Post-Processing Pipeline
Raw window probabilities $p_i$ are converted to discrete clinical alarm episodes via:
1. **Binarization**: $\hat{y}_i = \mathbb{I}(p_i \ge \tau)$.
2. **3-Window Temporal Majority Filtering**: Suppresses single-window chatter.
3. **15.0s Alarm Merging**: Alarms separated by $\le 15.0\text{s}$ are merged.
4. **5.0s Minimum Duration**: Alarms $< 5.0\text{s}$ are rejected.
5. **Per-Recording Isolation**: Post-processing is executed per EDF file.

### D. Metrics Separation
1. **Raw FP Window Rate**:
   $$\text{Raw FP Windows / 24h} = \frac{FP}{H_{\text{EDF}}} \times 24.0$$
2. **Clinical False Alarm Episode Rate**:
   $$\text{Clinical FA Episodes / 24h} = \frac{N_{\text{unmatched\_alarm\_episodes}}}{H_{\text{EDF}}} \times 24.0$$
3. **Detection Latency**:
   $$\text{Mean Onset Delay} = \frac{1}{N_{\text{det}}} \sum_{j \in \text{Detected}} \max(0.0, A_j^{\text{start}} - S_j^{\text{start}})$$
