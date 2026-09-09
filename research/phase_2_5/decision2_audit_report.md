# NeuroAegis Research: Decision 2 Class-Imbalance Audit Report

**Audit Target**: Decision 2 Class-Imbalance Handling Implementation  
**Dataset**: CHB-MIT Scalp EEG Database (24 Patients, 1,414,710 Windows)  
**Auditor**: Antigravity Research Agent  
**Date**: September 2026  
**Status**: Comprehensive Verification Complete  

---

## A. FINAL VERDICT

**PASS WITH WARNINGS**

> The Decision 2 implementation (Dynamic Negative Subsampling + Binary Focal Loss) is scientifically sound, leak-free, mathematically verified, and strictly isolated at the patient, recording, and window boundaries. The warnings documented below relate to minor hardcoded figure labels that were identified during audit and dynamically refactored, as well as the need to ensure the forthcoming CNN model never appends a final sigmoid layer.

---

## B. FOCAL LOSS

- **Implementation File**: `research/imbalance/focal_loss.py`
- **Class Name**: `BinaryFocalLossWithLogits(nn.Module)`
- **Mathematical Formulation**:
  $$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$
  $$p_t = y \cdot p + (1 - y)(1 - p)$$
  $$\alpha_t = y \cdot \alpha + (1 - y)(1 - \alpha)$$
  $$\text{FL}(z, y) = \alpha_t \cdot (1 - p_t)^\gamma \cdot \text{BCEWithLogits}(z, y)$$
  where $\text{BCEWithLogits}(z, y) = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})$.
- **Configurable Parameters**:
  - $\gamma$ (`gamma`): Primary default = $2.0$ (modulates hard vs easy examples).
  - $\alpha$ (`alpha`): Primary default = $0.25$ (weights rare positive class).
  - Both parameters are explicit instance attributes, fully configurable, and never hidden constants.
- **Logits Handling**: Directly accepts unnormalized real-valued logits $z \in \mathbb{R}$.
- **Sigmoid Location**:
  - Training: `torch.sigmoid()` is evaluated strictly inside `BinaryFocalLossWithLogits` for calculating $p_t$ in the focal weight; loss reduction utilizes `F.binary_cross_entropy_with_logits` for log-sum-exp numerical stabilization.
  - Metrics / Inference: Evaluated exactly once in `logits_to_probabilities(logits)` for thresholding.
  - Model: The future CNN baseline must terminate with linear logits (no final `nn.Sigmoid()`).
- **Numerical Stability**:
  - Probability $p_t$ is clamped to $[\epsilon, 1 - \epsilon]$ with $\epsilon = 10^{-8}$.
  - Verified across extreme logit bounds $[-100.0, +100.0]$: produces finite scalar loss ($0.008664$), zero NaN, and zero Inf.
- **Unit-Test Results (Mathematical Verification)**:
  - **TEST A ($\gamma=0$ Equivalence)**: Identical to $\alpha$-weighted BCE ($\text{max\_diff} = 0.0000 \times 10^0 < 10^{-6}$) — **PASS**
  - **TEST B (Perfect Positive $z=+15, y=1$)**: Loss = $0.00000000 < 10^{-4}$ — **PASS**
  - **TEST C (Confident Wrong Positive $z=-15, y=1$)**: Loss = $3.7500$ ($3.75 \times 10^{12}\times$ larger than perfect) — **PASS**
  - **TEST D (Confident Correct Negative $z=-15, y=0$)**: Loss = $0.00000000 < 10^{-4}$ — **PASS**
  - **TEST E (Focal Modulation)**: Easy example suppressed by $0.03\times$ relative to hard example compared to standard BCE — **PASS**
  - **TEST F (Random Logits)**: Finite loss ($0.1706$), zero NaN, zero Inf — **PASS**
  - **TEST G (Gradient Check)**: Valid backward pass, finite non-zero gradients — **PASS**
  - **TEST H (Extreme Logits $[-100, +100]$)**: Finite loss ($0.008664$), zero NaN/Inf — **PASS**

---

## C. PATIENT SPLITS

- **Partitioning Module**: `research/imbalance/patient_splitter.py`
- **TRAIN Patient IDs (16 Patients)**:
  `['chb04', 'chb09', 'chb11', 'chb12', 'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19', 'chb20', 'chb21', 'chb22', 'chb23', 'chb24']`
- **VALIDATION Patient IDs (4 Patients)**:
  `['chb06', 'chb07', 'chb08', 'chb10']`
- **TEST Patient IDs (4 Patients)**:
  `['chb01', 'chb02', 'chb03', 'chb05']`
- **Set Intersections**:
  - $\text{Train} \cap \text{Validation} = \emptyset$ (Cardinality: 0) — **PASS**
  - $\text{Train} \cap \text{Test} = \emptyset$ (Cardinality: 0) — **PASS**
  - $\text{Validation} \cap \text{Test} = \emptyset$ (Cardinality: 0) — **PASS**
- **Isolation Result**: 100% mutually disjoint. Zero patient leakage.

---

## D. RECORDING SPLITS

- **Train Recording Count**: 449 recordings
- **Validation Recording Count**: 82 recordings
- **Test Recording Count**: 155 recordings
- **Total Recordings Accounted For**: $449 + 82 + 155 = 686$ recordings (100.0% of CHB-MIT)
- **Overlap Results**:
  - $\text{Train Recordings} \cap \text{Val Recordings} = \emptyset$ (0 overlap) — **PASS**
  - $\text{Train Recordings} \cap \text{Test Recordings} = \emptyset$ (0 overlap) — **PASS**
  - $\text{Val Recordings} \cap \text{Test Recordings} = \emptyset$ (0 overlap) — **PASS**
- **Isolation Result**: No recording is split across sets.

---

## E. WINDOW SPLITS

- **Train Window Count**: 901,391 windows
- **Validation Window Count**: 293,410 windows
- **Test Window Count**: 219,909 windows
- **Total Windows Accounted For**: $901,391 + 293,410 + 219,909 = 1,414,710$ windows (100.0% of Master Index)
- **Overlap Results**:
  - $\text{Train Windows} \cap \text{Val Windows} = \emptyset$ (0 overlap) — **PASS**
  - $\text{Train Windows} \cap \text{Test Windows} = \emptyset$ (0 overlap) — **PASS**
  - $\text{Val Windows} \cap \text{Test Windows} = \emptyset$ (0 overlap) — **PASS**
- **Duplicates Within Splits**:
  - Train unique IDs: 901,391 / 901,391 (0 duplicates) — **PASS**
  - Validation unique IDs: 293,410 / 293,410 (0 duplicates) — **PASS**
  - Test unique IDs: 219,909 / 219,909 (0 duplicates) — **PASS**
- **Master Index Integrity**:
  - SHA256 Checksum: `f76dddb19fa2c24b3af24b989168b12465512b650c5e6d84bb0125a6d666302c`
  - Match Status: **STRICTLY IDENTICAL & UNCHANGED** — **PASS**

---

## F. SAMPLING

- **Implementation File**: `research/imbalance/dynamic_sampler.py`
- **Class**: `DynamicNegativeSampler(Sampler[int])`
- **Requested Primary Ratio**: 10.0 : 1
- **Actual Achieved Ratio**: 10.000 : 1
- **Available Train Positives**: 3,543 windows
- **Available Train Negatives**: 897,848 windows
- **Sampled Positives per Epoch**: 3,543 windows (100% retained, 0 discarded, 0 duplicated)
- **Sampled Negatives per Epoch**: 35,430 windows
- **Total Windows Presented per Epoch**: 38,973 windows
- **Epoch-to-Epoch Dynamics**:
  - Epoch 1 (Seed 42): 35,430 negatives sampled
  - Epoch 2 (Seed 43): 35,430 negatives sampled
  - Epoch 3 (Seed 44): 35,430 negatives sampled
  - Epoch 4 (Seed 45): 35,430 negatives sampled
  - Epoch 5 (Seed 46): 35,430 negatives sampled
  - Unique negatives sampled across 5 epochs: **163,716 windows** (18.23% coverage of entire pool)
  - Inter-epoch Jaccard overlap: **0.0203** (97.97% novel negative sampling across epochs)
  - Reproducibility check on identical seed: **100% bit-exact identical selection** — **PASS**

---

## G. VALIDATION

- **Validation Patients**: 4 (`chb06`, `chb07`, `chb08`, `chb10`)
- **Positive Windows**: 782
- **Negative Windows**: 292,628
- **Total Windows**: 293,410
- **Natural Class Imbalance Ratio**: **374.20 : 1** (0.267% positive)
- **Seizure Events**: 25 events (1,844 seconds ictal duration)
- **Sampler Bypass Confirmation**: Verified. The validation dataset completely bypasses `DynamicNegativeSampler`. It is evaluated under its natural, untouched clinical distribution. — **PASS**

---

## H. TEST

- **Test Patients**: 4 (`chb01`, `chb02`, `chb03`, `chb05`)
- **Positive Windows**: 674
- **Negative Windows**: 219,235
- **Total Windows**: 219,909
- **Natural Class Imbalance Ratio**: **325.27 : 1** (0.306% positive)
- **Seizure Events**: 22 events (1,574 seconds ictal duration)
- **Sampler Bypass Confirmation**: Verified. Zero negative subsampling, zero positive oversampling, zero SMOTE, zero synthetic windows, zero test-based sampling. The test set represents natural clinical monitoring, ensuring False Alarms per 24 Hours (FA/24h) is scientifically valid. — **PASS**

---

## I. LEAKAGE

- **Patient Leakage**: None ($\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$).
- **Recording Leakage**: None. Recordings are disjoint across splits.
- **Window Leakage**: None. 0 window overlap across sets.
- **Normalization Leakage**: None. Normalization is strictly recording-local (`zscore_recording_local` in `CHBMITNormalizer`), computing mean and std strictly within each isolated recording file. No cross-patient global statistics are fitted.
- **Threshold Leakage**: None. Default decision threshold is 0.5. No test labels or test predictions have been used for threshold tuning.
- **Model Selection / Early Stopping Leakage**: None. Zero model training has occurred.

---

## J. REPRODUCIBILITY

- **Master Seed**: 42
- **Epoch Seed Formula**: $\text{seed} = \text{base\_seed} + \text{epoch}$
- **Git Commit**: `17943cdaccfa1d6857f787b91e53b223dbbb8616`
- **Execution Timestamp**: `2026-09-05T17:56:42Z`
- **Environment Logged**:
  - Python: `3.11.15`
  - PyTorch: `2.13.0` (Apple MPS Backend: `True`)
  - MNE: `1.7.0`
  - NumPy: `1.26.4`
  - SciPy: `1.12.0`
  - Pandas: `2.2.1`
  - OpenPyXL: `3.1.5`
  - Matplotlib: `3.11.1`
  - OS: `macOS-26.6.2-arm64-arm-64bit`
  - Hardware: `Apple Silicon M4 (16 GB Unified Memory)`
- **Configuration Files**:
  - `research/config/class_imbalance.yaml`
  - `research/imbalance/class_imbalance_config.json`
  - `research/imbalance/environment.json`

---

## K. AUTOMATED TESTS

- **Test Suite**: `research/imbalance/test_class_imbalance.py`
- **Total Tests**: 14
- **Passed**: 14 / 14 (100%)
- **Failed**: 0
- **Skipped**: 0
- **Warnings**: 0
- **Runtime**: 2.24 seconds

All 14 tests execute live computations (hashing files, performing set intersections, simulating dynamic sampler epochs, checking gradients, verifying mathematical reductions) rather than testing superficial mock flags.

---

## L. HARD-CODING AUDIT

- **Findings**:
  - [WARNING - RESOLVED]: In the initial visualization generator `run_imbalance_pipeline.py`, four string and scalar counts (e.g. `sampled_counts = [35430, 3543]`, `neg_ratios = [282.0, ...]`, `pct_pool = unq / 897848 * 100`, and distribution string templates) were written as literal constants instead of dynamically derived attributes.
  - [CORRECTION APPLIED]: All occurrences were refactored to dynamically extract counts from `split_summary` and `DynamicNegativeSampler` attributes. Re-execution confirmed bit-exact consistency.
  - In `test_class_imbalance.py`: Test assertions 07 and 08 assert against ground-truth partition counts (782 and 674 positives). These are regression invariants verifying that partition logic remains strictly deterministic.

---

## M. GENERATED FILES

1. `research/config/class_imbalance.yaml` (Declarative hyperparameters)
2. `research/imbalance/focal_loss.py` (Focal Loss & probability extraction)
3. `research/imbalance/patient_splitter.py` (Leak-safe patient partitioner)
4. `research/imbalance/dynamic_sampler.py` (PyTorch Sampler for dynamic negative subsampling)
5. `research/imbalance/metrics.py` (Clinical seizure metrics suite & FA/24h)
6. `research/imbalance/run_imbalance_pipeline.py` (Master ablation & logging pipeline)
7. `research/imbalance/test_class_imbalance.py` (14-point automated test suite)
8. `research/imbalance/class_imbalance_config.json` (Machine-readable config)
9. `research/imbalance/environment.json` (Environment metadata log)
10. `research/imbalance/Class_Imbalance_Audit.xlsx` (3-sheet styled Excel workbook)
11. `research/imbalance/class_imbalance_report.md` (Comprehensive audit report)
12. `research/imbalance/figures/figure1_full_training_pool_distribution.png` (300 DPI)
13. `research/imbalance/figures/figure2_sampled_training_epoch_distribution.png` (300 DPI)
14. `research/imbalance/figures/figure3_distribution_comparison_splits.png` (300 DPI)
15. `research/imbalance/figures/figure4_sampling_ratio_comparison_5_10_20.png` (300 DPI)
16. `research/phase_2_5/decision2_audit_report.md` (This document)
17. `research/phase_2_5/decision2_audit.json` (Machine-readable audit summary)

---

## N. FINAL RECOMMENDATION

**READY FOR CNN BASELINE WITH DOCUMENTED WARNINGS**

### Documented Warnings for Phase 3 (CNN Baseline):
1. **Linear Logits Output**: The CNN model architecture must terminate with an unnormalized linear output layer (`nn.Linear(..., 1)`). Do NOT place `nn.Sigmoid()` at the end of the CNN, as `BinaryFocalLossWithLogits` expects raw logits.
2. **Evaluation Protocol**: When evaluating on validation or test sets, pass unnormalized logits to `logits_to_probabilities()` to obtain single-sigmoid probabilities before calculating Sensitivity, AUPRC, and FA/24h.
3. **Training Protocol**: Wrap only the `train_df` with `DynamicNegativeSampler`. Do not pass validation or test data to the negative sampler.
