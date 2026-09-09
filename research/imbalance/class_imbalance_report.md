# NeuroAegis Research: Class Imbalance Handling Audit Report

**Author**: NeuroAegis Research Team  
**Date**: September 2026  
**Status**: Complete & Verified (14/14 Automated Tests PASS)  
**Execution Runtime**: 3.28 seconds (Pipeline) | 2.25 seconds (Automated Test Suite)  

---

## 1. Executive Summary & Core Research Interpretation

In this task, we implemented and rigorously audited the **patient-independent class imbalance handling strategy** for the NeuroAegis epileptic seizure detection system on the CHB-MIT Scalp EEG dataset (24 patients, 1,414,710 windows, ~982.9 continuous hours).

> [!IMPORTANT]
> **Authoritative Research Interpretation**:  
> *"The underlying clinical dataset remains severely imbalanced (~282:1). The class imbalance was not 'solved' or removed. Rather, training exposure to the majority class was controlled using dynamic negative subsampling, while focal loss was used to emphasize difficult examples. The natural validation and test distributions remain completely untouched."*

### Key Accomplishments
1. **Master Index Immutability**: The master window index (`research/data/manifests/chbmit_window_index.csv`, 1,414,710 rows) remains **strictly unchanged** (verified via SHA256 checksum `f76dddb19fa2c24b3af24b989168b12465512b650c5e6d84bb0125a6d666302c`).
2. **Strict Patient-Independent Splitting**: Data partitioning is strictly enforced at the patient identity boundary **before** any sampling occurs.
   - **Training (16 patients)**: 901,391 windows (3,543 positives, 897,848 negatives, natural ratio 253.4:1, 151 seizure events).
   - **Validation (4 patients)**: 293,410 windows (782 positives, 292,628 negatives, natural ratio 374.2:1, 25 seizure events).
   - **Test (4 patients)**: 219,909 windows (674 positives, 219,235 negatives, natural ratio 325.3:1, 22 seizure events).
3. **Dynamic Negative Subsampling**: Implemented `DynamicNegativeSampler` as a PyTorch `Sampler`.
   - **Primary Baseline Ratio**: 10.0:1 (35,430 sampled negatives vs 3,543 positives per epoch = 38,973 total windows per epoch).
   - **100% Positive Retention**: Zero positive training windows are discarded, and zero positive windows are artificially duplicated.
   - **Dynamic Diversity**: Each epoch samples a different subset of negatives using deterministic seed $\text{seed} = \text{base\_seed} + \text{epoch}$. Over 5 simulated epochs, the model encounters 163,716 unique negative windows (18.2% of the entire negative pool).
4. **Numerically Stable Binary Focal Loss**: Implemented `BinaryFocalLossWithLogits`.
   - Accepts unnormalized raw logits directly to avoid double-sigmoid and numerical instability.
   - Initial experimental hyperparameters: $\gamma = 2.0$, $\alpha = 0.25$.
   - Mathematically and numerically verified: when $\gamma = 0$, identically reduces to $\alpha$-weighted BCE ($\Delta = 0.00 \times 10^0$).
5. **Multi-Ratio Ablation Framework**: Implemented and executed logging for 5:1, 10:1 (baseline), and 20:1 sampling ratios, as well as BCE vs Focal Loss.
6. **Automated Verification**: **14 out of 14 mandatory validation tests passed** in `test_class_imbalance.py`.
7. **Zero Model Training**: Zero neural network training was performed in this task.

---

## 2. Patient Partitioning & Natural Distribution Preservation

Class balancing is strictly applied **only** to the training fold. The validation and test sets preserve their exact natural clinical distributions to ensure metrics like **False Alarms per 24 Hours (FA/24h)** remain clinically valid.

| Partition | Patient Count | Patient Identifiers | Total Windows | Positive Windows | Negative Windows | Positive % | Natural Imbalance Ratio | Seizure Events | Seizure Duration |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Training** | 16 | `chb04, chb09, chb11, chb12, chb13, chb14, chb15, chb16, chb17, chb18, chb19, chb20, chb21, chb22, chb23, chb24` | 901,391 | 3,543 | 897,848 | 0.393% | **253.4 : 1** | 151 | 8,193 s |
| **Validation** | 4 | `chb06, chb07, chb08, chb10` | 293,410 | 782 | 292,628 | 0.267% | **374.2 : 1** | 25 | 1,844 s |
| **Test** | 4 | `chb01, chb02, chb03, chb05` | 219,909 | 674 | 219,235 | 0.306% | **325.3 : 1** | 22 | 1,574 s |
| **Master Index** | **24** | **All 24 Patients** | **1,414,710** | **4,999** | **1,409,711** | **0.353%** | **282.0 : 1** | **198** | **11,611 s** |

### Patient Isolation Assertions
- $\text{Train} \cap \text{Validation} = \emptyset$ (0% overlap)
- $\text{Train} \cap \text{Test} = \emptyset$ (0% overlap)
- $\text{Validation} \cap \text{Test} = \emptyset$ (0% overlap)
- Every sampled window ID in training strictly satisfies: $\text{patient\_id} \in \text{Train Patients}$.

---

## 3. Dynamic Negative Subsampling Design

### Sampling Algorithm
In each epoch $e \in \{0, 1, \dots\}$:
1. **Epoch Seed**: $s_e = \text{base\_seed} + e = 42 + e$.
2. **Positive Pool**: All $N_{\text{pos}} = 3,543$ positive training windows are included.
3. **Negative Sampling**: $N_{\text{neg\_sample}} = \min(\text{round}(N_{\text{pos}} \times \text{ratio}), N_{\text{neg\_available}})$.
   - For ratio $10.0$: $N_{\text{neg\_sample}} = 35,430$.
   - Sampled without replacement from the $897,848$ negative training windows using NumPy `default_rng(s_e)`.
4. **Epoch Batching**: Positive and negative indices are concatenated and shuffled for batch iteration.

### Multi-Ratio Ablation Comparison

| Configuration | Configured Ratio | Sampled Positives | Available Negatives | Sampled Negatives | Actual Ratio | Total Epoch Windows | Unique Negatives Seen (5 Epochs) | Pool Coverage (5 Epochs) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **EXP-C-FOCAL-5TO1** | 5.0 : 1 | 3,543 | 897,848 | 17,715 | 5.000 : 1 | 21,258 | 85,141 | 9.5% |
| **EXP-B-FOCAL-10TO1 (Baseline)** | **10.0 : 1** | **3,543** | **897,848** | **35,430** | **10.000 : 1** | **38,973** | **163,716** | **18.2%** |
| **EXP-C-FOCAL-20TO1** | 20.0 : 1 | 3,543 | 897,848 | 70,860 | 20.000 : 1 | 74,403 | 302,797 | 33.7% |

---

## 4. Binary Focal Loss Formulation & Numerical Stability

### Mathematical Definition
Given model logits $z \in \mathbb{R}$ and ground truth $y \in \{0, 1\}$:
$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$
$$p_t = y \cdot p + (1 - y)(1 - p)$$
$$\alpha_t = y \cdot \alpha + (1 - y)(1 - \alpha)$$
$$\text{FL}(z, y) = \alpha_t \cdot (1 - p_t)^\gamma \cdot \text{BCEWithLogits}(z, y)$$

Where $\text{BCEWithLogits}(z, y) = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})$.

### Properties Verified
1. **Numerical Clamping**: $p_t$ is clamped to $[\epsilon, 1 - \epsilon]$ with $\epsilon = 10^{-8}$, preventing $\log(0)$ or division by zero.
2. **Extreme Logit Stability**: Verified finite loss ($0.006189$) and valid gradients across extreme bounds $[-100.0, +100.0]$. Zero NaN or Inf.
3. **Single Sigmoid Rule**: Model outputs raw unnormalized logits. Sigmoid is applied **only once** inside `logits_to_probabilities()` for metric evaluation and thresholding.
4. **Weighted BCE Equivalence**: When $\gamma = 0$, $(1 - p_t)^0 = 1.0$, reducing identically to $\alpha_t \cdot \text{BCEWithLogits}(z, y)$. Verified numerical difference: $\mathbf{0.00 \times 10^0}$.

---

## 5. Automated Validation Results (14/14 PASS)

The test suite `research/imbalance/test_class_imbalance.py` was executed:

| Test ID | Test Description | Target Requirement | Status | Execution Detail |
|:---:|---|---|:---:|---|
| **Test 01** | Master index immutability | Section 22.1 | **PASS** | SHA256 matches `f76dddb1...`, 1,414,710 rows intact |
| **Test 02** | No validation patient in training sampler | Section 22.2 | **PASS** | 0 validation patients across all epochs |
| **Test 03** | No test patient in training sampler | Section 22.3 | **PASS** | 0 test patients across all epochs |
| **Test 04** | All sampled window IDs exist in training set | Section 22.4 | **PASS** | 38,973 sampled IDs strictly in training manifest |
| **Test 05** | Zero synthetic EEG windows created | Section 22.5 | **PASS** | 100% genuine indexed CHB-MIT windows |
| **Test 06** | 100% positive window retention | Section 22.6 | **PASS** | Exactly 3,543 positives retained; zero duplicated |
| **Test 07** | Validation distribution unchanged | Section 22.7 | **PASS** | Exactly 782 pos / 292,628 neg (374.2:1) preserved |
| **Test 08** | Test distribution unchanged | Section 22.8 | **PASS** | Exactly 674 pos / 219,235 neg (325.3:1) preserved |
| **Test 09** | Dynamic sampling diversity & reproducibility | Section 22.9 | **PASS** | Jaccard overlap 0.0203 between epochs; 100% reproducible |
| **Test 10** | Configured sampling ratio accuracy | Section 22.10 | **PASS** | 5:1, 10:1, 20:1 verified within $\pm 0.05$ |
| **Test 11** | Focal Loss accepts raw logits | Section 22.11 | **PASS** | Accepts unnormalized logits directly; scalar loss 0.0122 |
| **Test 12** | Single-sigmoid probability conversion | Section 22.12 | **PASS** | Output strictly in $[0, 1]$; sigmoid called once |
| **Test 13** | No NaN/Inf across extreme logits | Section 22.13 | **PASS** | Stable across $[-100.0, +100.0]$; finite loss |
| **Test 14** | $\gamma=0$ equivalence to weighted BCE | Section 22.14 | **PASS** | Maximum difference $0.00 \times 10^0 < 10^{-6}$ |

---

## 6. Deliverables Inventory

All deliverables for this phase have been generated, saved, and verified:

### Core Code Modules
- `research/imbalance/focal_loss.py`: Binary Focal Loss with logits + single-sigmoid probability extractor.
- `research/imbalance/patient_splitter.py`: Strict patient partitioner (16/4/4 holdout + 24 LOPO-CV folds).
- `research/imbalance/dynamic_sampler.py`: PyTorch `Sampler` with deterministic per-epoch seeding.
- `research/imbalance/metrics.py`: Clinical metrics suite (Sensitivity, Specificity, Precision, F1, AUROC, AUPRC, FA/24h).
- `research/imbalance/run_imbalance_pipeline.py`: Master pipeline executing all ablations, figure generation, and Excel creation.
- `research/imbalance/test_class_imbalance.py`: 14 automated unit and integration tests.

### Configuration & Logs
- `research/config/class_imbalance.yaml`: Declarative hyperparameter specification.
- `research/imbalance/class_imbalance_config.json`: Machine-readable execution configuration.
- `research/imbalance/environment.json`: Hardware/software execution log.

### Publication-Grade Figures (300 DPI)
- `research/imbalance/figures/figure1_full_training_pool_distribution.png`: Original training pool class distribution.
- `research/imbalance/figures/figure2_sampled_training_epoch_distribution.png`: Sampled training distribution (10:1 ratio).
- `research/imbalance/figures/figure3_distribution_comparison_splits.png`: Ratio comparison across Master, Train, Val, and Test.
- `research/imbalance/figures/figure4_sampling_ratio_comparison_5_10_20.png`: Multi-ratio ablation comparison (5:1 vs 10:1 vs 20:1).

### Styled Excel Audit Workbook
- `research/imbalance/Class_Imbalance_Audit.xlsx`:
  - Sheet 1: `Class Imbalance Strategy` (Summary of all 4 ablation experiments)
  - Sheet 2: `Sampling Epoch History` (Epoch-by-epoch sample counts, seeds, and actual ratios)
  - Sheet 3: `Split Partitions` (Patient lists, window counts, and seizure counts)

---

## 7. Next Steps & Phase 3 Handoff

The class imbalance handling framework is frozen and verified:
1. When neural network training commences, `DynamicNegativeSampler` will wrap `train_df`, and `BinaryFocalLossWithLogits` will compute training gradients from raw model logits.
2. Validation and test evaluation will run on natural distributions through `SeizureEvaluationMetrics`.
3. The baseline model architecture (CNN) can now be implemented on top of this standardized data pipeline.
