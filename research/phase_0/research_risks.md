# NeuroAegis Research — Phase 0 Research Risks & Methodological Audit

**Date**: 2026-09-05  
**Audit Scope**: Codebase, Datasets, Preprocessing Pipelines, Evaluation Schemes, and Inference Services

---

## 1. RESEARCH RISK 1: Potential Patient Data Leakage (Random Window Splitting vs LOPO-CV)

### Description
In early research scripts and exploratory notebooks (`notebooks/neuroaegis-v1.ipynb`, `scripts/extracted_notebook.py`), EEG window segmentation was followed by standard random stratified train/test splitting (`train_test_split(..., shuffle=True)`). 

### Severity: **CRITICAL / HIGH**

### Clinical & Scientific Impact
Continuous EEG recordings from a single patient possess strong patient-specific baseline physiological signatures (skull conductance, electrode impedances, background alpha rhythm frequency, and individual epileptogenic morphology). When windows from the same patient or recording session are randomly partitioned into both training and test sets:
- The model memorizes patient-specific background features rather than learning generalized pathophysiological seizure dynamics.
- Reported test accuracy and AUROC appear artificially inflated (>98–99%).
- When deployed on an unseen patient in real clinical testing, model performance collapses drastically (recall dropping from 98% to <20%).

### Resolution & Mandatory Protocol
1. **Strict Leave-One-Patient-Out Cross-Validation (LOPO-CV)** must be enforced across all 24 CHB-MIT patients and external datasets.
2. No sliding window from a test patient may ever appear in the training or validation fold during hyperparameter selection or model training.

---

## 2. RESEARCH RISK 2: Extreme Class Imbalance & The "Accuracy Paradox"

### Description
Across continuous multi-channel monitoring in CHB-MIT (and real-world clinical epilepsy monitoring units), seizures are rare paroxysmal events. In `chbmit_subset.parquet`, seizure windows constitute only **1.82% of all data (53.97:1 non-seizure to seizure ratio)**.

### Severity: **CRITICAL / HIGH**

### Clinical & Scientific Impact
- A trivial dummy classifier that predicts "Non-Seizure (0)" for 100% of samples achieves **98.18% raw accuracy**, despite having a clinical utility of **0% (Recall = 0.0, F1 = 0.0)**.
- Standard cross-entropy loss gradients are dominated by background non-seizure noise, causing standard neural networks to collapse to the majority class without class re-weighting or focal loss.

### Resolution & Mandatory Protocol
1. **AUPRC (Area Under Precision-Recall Curve)** and **Balanced Accuracy** must replace raw accuracy as primary evaluation criteria.
2. Training loss functions must incorporate **Focal Loss** ($\gamma=2.0$), **Cost-Sensitive Class Weights** ($w_{pos} pprox 50.0$), or **Hard Negative Mining**.
3. Evaluate threshold-independent curves and fixed false-alarm operating points (Sensitivity @ 0.25 FP/hr and 1.0 FP/hr).

---

## 3. RESEARCH RISK 3: Single-Channel Extraction Discarding Spatial EEG Topology

### Description
In `scripts/extract_chbmit_subset.py` (line 148), feature extraction was performed strictly on `window_data[0:1, :]` (Channel 0: `FP1-F7`), completely discarding the remaining 22 bipolar channels.

### Severity: **HIGH**

### Clinical & Scientific Impact
- Focal seizures in patients `chb02` (central/frontal), `chb03` (parietal/occipital), and `chb04` (temporal bilateral) originate in brain regions distant from channel `FP1-F7`.
- Single-channel monitoring cannot capture seizure propagation, hemispheric synchronization, or spatial phase lead-lag relationships.

### Resolution & Mandatory Protocol
- The locked target research architecture (**CNN + GNN + GRU + Attention**) processes all 23 channels simultaneously:
  - **1D CNN backbone** extracts multi-scale temporal-spectral features per channel.
  - **Graph Neural Network (GNN)** models anatomical and functional connectivity over the 10-20 electrode layout.
  - **GRU / BiGRU** captures long-range temporal recurrence.
  - **Spatial-Temporal Attention** dynamically highlights the epileptogenic onset zone.

---

## 4. RESEARCH RISK 4: Physical Unit Scaling Mismatch Across Datasets

### Description
Bonn University raw data is dimensionless/arbitrary units (standardized to Z-scores in `bonn_features.csv`), whereas CHB-MIT scalp EEG is in microvolts ($\mu	ext{V}$, typical amplitude $10 - 150 \mu	ext{V}$).

### Severity: **MEDIUM / HIGH**

### Clinical & Scientific Impact
- Direct cross-dataset zero-shot evaluation without independent standardization causes massive covariate shift, resulting in extreme false positive rates or near-zero sensitivity.

### Resolution & Mandatory Protocol
- Independent per-dataset z-score normalization (`(X - mean) / std`) or robust scaling (`RobustScaler`) must be fit strictly on training patient sets and applied to evaluation sets.

---

## 5. RESEARCH RISK 5: Missing External Validation Dataset (Siena Scalp EEG)

### Description
The Siena Scalp EEG Database (PhysioNet) is the locked primary external dataset for zero-shot cross-domain testing, but is not yet downloaded into the workspace.

### Severity: **MEDIUM**

### Resolution & Mandatory Protocol
- Maintain dataset isolation. In Phase 1/2, verify download utilities for Siena database (512 Hz, 14 patients, ~128 hours of continuous EEG).

---

## 6. RESEARCH RISK 6: Short Seizures vs Windowing Resolution

### Description
The minimum verified clinical seizure duration in CHB-MIT is **6 seconds** (`chb16_17.edf`) and 9 seconds (`chb02_16.edf`). If window durations are set too large (e.g. 23.6s or 60s), short focal seizures are diluted by surrounding non-seizure background EEG within the window.

### Severity: **MEDIUM**

### Resolution & Mandatory Protocol
- Evaluate fine-grained windowing (e.g. **4–5 second sliding windows** with 50% overlap) in Phase 2 data preparation, with formal event-level overlap criteria.
