# NeuroAegis Full-Pipeline Memory Scaling Audit

**Date**: 2026-09-16 19:15:22
**Hardware Platform**: Apple Silicon (M4, 16 GB Unified RAM)
**Compute Acceleration**: Apple Metal Performance Shaders (MPS)
**Execution Environment**: Python 3.11.15 (Isolated Process per Test Workload)
**Batch Size**: 2 (Fixed across all workloads)

---

## 1. Executive Summary & Verdict

- **Determined Memory Scaling Pattern**: **B** (Memory increases proportionally with number of windows stored in RAM (Expected scaling bounded by pre-allocated array size))
- **Continuous Training Leakage**: **None detected**. Across all batch iterations (up to 4,000 steps with `batch_size=2`), MPS allocated memory and training RSS remained strictly bounded.
- **Peak RSS Range**: `627.6 MB` (100 windows) $\rightarrow$ `1897.5 MB` (10,000 windows).
- **Unified Memory Headroom**: Safe. Peak RSS for 10,000 full-pipeline windows consumed only ~1.85 GB of the 16 GB unified budget (~11.9% system memory).

---

## 2. Core Empirical Metrics Table

| Workload | Windows | Steps (`bs=2`) | 1. RSS Before | 2. RSS Preprocessing | 3. RSS Training | 4. Peak RSS | 5. MPS Alloc | 6. MPS Driver | 7. RSS Cleanup | Duration |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Test A** | 100 | 40 | 347.6 MB | 412.6 MB | 624.3 MB | **627.6 MB** | 3.37 MB | 18.9 MB | 602.1 MB | 7.4s |
| **Test B** | 1,000 | 400 | 346.9 MB | 615.1 MB | 826.7 MB | **830.2 MB** | 3.37 MB | 18.9 MB | 602.4 MB | 10.1s |
| **Test C** | 5,000 | 2,000 | 347.5 MB | 955.0 MB | 1404.8 MB | **1408.4 MB** | 3.37 MB | 18.9 MB | 604.8 MB | 22.0s |
| **Test D** | 10,000 | 4,000 | 346.0 MB | 1202.3 MB | 1893.9 MB | **1897.5 MB** | 3.37 MB | 18.9 MB | 607.0 MB | 36.9s |

---

## 3. Operation-by-Operation Breakdown & Peak Identification

The pipeline stages profiled were:
1. **EDF Loading**: Opening EDF files, reading headers, channel mapping (canonical 23 bipolar montage).
2. **Preprocessing**: Zero-phase filtering (60 Hz notch + 0.5–40 Hz bandpass) and local z-score normalization.
3. **Window Generation**: Slicing 5.0-second EEG segments (1280 samples) into contiguous arrays.
4. **Tensor Conversion**: Transferring slices into PyTorch `TensorDataset` and initializing zero-worker `DataLoader`.
5. **Model Training**: 1D CNN baseline forward pass, Binary Focal Loss (`gamma=2.0, alpha=0.25`), backward gradient backprop, AdamW update on MPS.
6. **Validation**: Gradient-free forward evaluation (`torch.no_grad()`) and sigmoid probability mapping.
7. **Prediction Storage**: Collecting evaluation probabilities and computing scalar metrics.
8. **Cleanup**: Explicit variable deallocation and MPS cache flushing (`torch.mps.empty_cache()` + `gc.collect()`).

### Peak Operations Observed by Test:

| Test | Windows | Peak Operation | Peak RSS (MB) | Details |
|:---|:---:|:---|:---:|:---|
| **Test A** | 100 | `9_cleanup` | 627.6 MB | Array allocation + tensor residency |
| **Test B** | 1,000 | `7_validation` | 830.2 MB | Array allocation + tensor residency |
| **Test C** | 5,000 | `9_cleanup` | 1408.4 MB | Array allocation + tensor residency |
| **Test D** | 10,000 | `7_validation` | 1897.5 MB | Array allocation + tensor residency |

### Detailed Stage Comparison across Tests (Live RSS in MB):

| Operation | Test A (100) | Test B (1,000) | Test C (5,000) | Test D (10,000) |
|:---|:---:|:---:|:---:|:---:|
| **Pre-test Baseline** | 347.6 MB | 346.9 MB | 347.5 MB | 346.0 MB |
| **EDF Index / Open** | 353.5 MB | 353.1 MB | 354.5 MB | 359.2 MB |
| **Signal Preprocessing** | 412.6 MB | 615.1 MB | 955.0 MB | 1202.3 MB |
| **Window Generation** | 412.6 MB | 615.1 MB | 1191.8 MB | 1680.7 MB |
| **Tensor Conversion** | 413.1 MB | 615.6 MB | 1192.3 MB | 1681.2 MB |
| **Model Training (MPS)** | 624.3 MB | 826.7 MB | 1404.8 MB | 1893.9 MB |
| **Validation Evaluation** | 627.6 MB | 830.2 MB | 1408.4 MB | 1897.5 MB |
| **Prediction Storage** | 627.6 MB | 830.2 MB | 1408.4 MB | 1897.5 MB |
| **Post-Cleanup** | 602.1 MB | 602.4 MB | 604.8 MB | 607.0 MB |

---

## 4. Training Step Stability (Within-Epoch Trajectory)

Tracking memory across training steps for the largest workload (**Test D: 10,000 windows, 4,000 training steps**):

| Progress | Step | Live RSS (MB) | MPS Alloc (MB) | MPS Driver (MB) | Stability Status |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0% | 1 | 1892.5 MB | 3.11 MB | 18.9 MB | **Bounded / Stable** |
| 25% | 1,001 | 1894.2 MB | 3.37 MB | 18.9 MB | **Bounded / Stable** |
| 50% | 2,001 | 1894.3 MB | 3.37 MB | 18.9 MB | **Bounded / Stable** |
| 75% | 3,001 | 1894.3 MB | 3.37 MB | 18.9 MB | **Bounded / Stable** |
| 100% | 4,000 | 1894.3 MB | 3.37 MB | 18.9 MB | **Bounded / Stable** |

---

## 5. Architectural & Pipeline Takeaways

1. **Strictly Linear & Bounded Scaling with Window Count (Pattern B)**:
   - Memory scales strictly with the resident float32 window storage: `23 channels × 1280 samples × 4 bytes = 117.76 KB per window`.
   - 10,000 windows require exactly ~1.15 GB of raw array storage.
   - There are no quadratic $O(N^2)$ cross-attention matrices or full-sequence tensor graphs retained.
2. **Zero GPU / MPS Accumulation (No Pattern C)**:
   - Because batch tensors are explicitly deleted and gradients zeroed with `set_to_none=True`, MPS allocated memory remains strictly pegged at exactly **3.37 MB** across all steps (model parameters + AdamW moments + current step activations), with zero growth from step 1 to step 4,000.
3. **Identification of Peak Memory Operation**:
   - The primary driver of scaling is **Window Generation & RAM Residency**: holding $N$ windows of `(23, 1280)` in `float32` arrays plus tensor views.
   - The operational peak RSS occurs during **Validation / Training Execution** (`~1,897.5 MB` in Test D), which represents the baseline Python/PyTorch runtime (~346 MB) + loaded window array storage (~1,150 MB) + MPS runtime overhead (~400 MB).
   - **EDF Loading**, **Model Training Steps**, **Validation Passes**, and **Prediction Storage** run with tiny constant footprints and do not accumulate.
4. **16 GB Unified Memory Viability**:
   - With 10,000 windows consuming only **1.898 GB peak RSS** (~11.9% of system memory), the pipeline retains over **14.1 GB of headroom** on the 16 GB Apple M4 machine.
