# NeuroAegis Phase 4A-C Audit Report
## Comprehensive Pre-Selection Audit of Spatial Graph Construction, Topologies, and Validation Experiments

**Audit Date:** 2026-09-06  
**Git Commit:** `17943cdaccfa1d6857f787b91e53b223dbbb8616`  
**Master Window Index SHA256:** `f76dddb19fa2c24b3af24b989168b12465512b650c5e6d84bb0125a6d666302c`  
**Scope:** Training & Validation Only (Test Set Untouched)

---

## 1. Executive Summary & Audit Mandate

This audit report formally establishes the provenance, algorithmic implementation, mathematical topology, and validation outcomes of the spatial graph experiments in Phase 4A and Phase 4A-C.
The primary objective of this phase is to:
1. Conduct an exhaustive audit of all source code, models, and predictions that generated Phase 4A and Phase 4A-C results.
2. Resolve and correct the topological reporting inconsistency regarding connected component counts across candidate thresholds $\theta \in \{0.25, 0.30, 0.35\}$.
3. Perform mathematical graph selection using **Validation Performance Only**.
4. Freeze the selected graph configuration before performing exactly **one** untouched final test evaluation.

---

## 2. Experimental Framework & Artifact Provenance

| Dimension | Specification | Source Artifact |
| :--- | :--- | :--- |
| **Dataset** | CHB-MIT Scalp EEG (PhysioNet) | `data/manifests/chbmit_manifest.csv` |
| **Bipolar Montage** | Canonical 23 Channels (International 10-20) | `research/data/config/chbmit_channel_order.json` |
| **Window Duration** | 5.0 seconds (1280 samples @ 256 Hz) | `data/manifests/chbmit_window_index.csv` |
| **Window Stride** | 2.5 seconds (640 samples @ 256 Hz) | `data/manifests/chbmit_window_index.csv` |
| **Primary Label** | $\ge 50\%$ Seizure Overlap (`label_50pct_overlap`) | `chbmit_window_index.csv` |
| **Training Patients (16)** | chb04, 09, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 | `research/imbalance/class_imbalance_config.json` |
| **Validation Patients (4)** | chb06, chb07, chb08, chb10 (293,410 windows, 25 seizures, 203.76h) | `research/imbalance/class_imbalance_config.json` |
| **Test Patients (4)** | chb01, chb02, chb03, chb05 (219,909 windows, 22 seizures, 152.82h) | `research/imbalance/class_imbalance_config.json` |
| **Graph Construction Engine** | Static cross-channel Pearson correlation on unlabelled training data | `research/phase_4a/graph_builder.py` |
| **Graph Construction Scope** | **Strictly Training Patients (Zero Val / Zero Test)** | Verified in Section 5 |
| **Model Architecture** | `Baseline1DCNN_GNN` (1D CNN Backbone + 2-Layer Spatial GCN + Dual Readout) | `research/phase_4a/cnn_gnn_model.py` |
| **Optimizer** | AdamW (learning rate = $10^{-3}$, weight decay = $10^{-4}$) | `train_candidate_models.py` |
| **Scheduler** | CosineAnnealingLR ($T_{\max} = 3, \eta_{\min} = 10^{-5}$) | `train_candidate_models.py` |
| **Loss Function** | BinaryFocalLossWithLogits ($\gamma = 2.0, \alpha = 0.25$) | `research/imbalance/focal_loss.py` |
| **Data Sampler** | Dynamic Negative Subsampling ($10:1$ ratio, $\text{seed} = 42 + \text{epoch}$) | `research/imbalance/dynamic_sampler.py` |
| **Batch Size** | 128 (Training), 256 (Validation Inference) | `train_candidate_models.py` |
| **Training Epochs** | 3 Epochs | `train_candidate_models.py` |
| **Model Checkpoint Policy**| Peak Validation AUPRC | `train_candidate_models.py` |
| **Decision Threshold** | $0.50$ (Uncalibrated sigmoid probability) | `train_candidate_models.py` |

---

## 3. Critical Topology Consistency Check

### 3.1 Explanation of Prior Discrepancy
Earlier exploratory diagnostic text referenced a hypothesis that lowering the adjacency threshold $\theta$ from $0.35$ to $0.25$ or $0.30$ would connect the graph into a single 23-node component.
However, re-computation directly from the actual serialized adjacency matrices (`graph_adjacency.csv`) reveals that **all three candidate graphs have exactly 2 connected components**:
- **Why this occurs:** The 4 occipital bipolar channels (`P7-O1`, `P3-O1`, `P4-O2`, `P8-O2`) exhibit high mutual correlations ($r > 0.35$ among each other), but their maximum cross-regional Pearson correlation with ANY non-occipital channel in the training set is $r = 0.2381$ (between `P8-O2` and `C4-P4`).
- Because $\theta = 0.25$ and $\theta = 0.30$ are both greater than $0.2381$, neither threshold bridges the gap between the occipital lobe and the parietal lobe.
- Both $\theta = 0.25$ and $\theta = 0.30$ merely add intra-regional edges within the 19-node anterior component and within the 4-node posterior component.

### 3.2 Exact Recomputed Topological Metrics

All values below were computed directly from the saved CSV matrices:

| Metric | Graph A ($\theta = 0.25$) | Graph B ($\theta = 0.30$) | Graph C ($\theta = 0.35$, Ref) |
| :--- | :---: | :---: | :---: |
| **Adjacency Source File** | `theta_025/graph_adjacency.csv` | `theta_030/graph_adjacency.csv` | `exp_01/graph_adjacency.csv` |
| **Adjacency SHA256** | `900c27c192b03ef2...` | `aece8c437f3eb267...` | `63a2e34d0ca965e3...` |
| **Number of Nodes** | 23 | 23 | 23 |
| **Undirected Edges** | **60** | **40** | **32** |
| **Graph Density** | **23.72%** | **15.81%** | **12.65%** |
| **Connected Components** | **2** | **2** | **2** |
| **Largest Component Size** | 19 nodes | 19 nodes | 19 nodes |
| **Smallest Component Size** | 4 nodes | 4 nodes | 4 nodes |
| **Occipital Component** | 4 nodes (`P7-O1`, `P3-O1`, `P4-O2`, `P8-O2`) | 4 nodes (`P7-O1`, `P3-O1`, `P4-O2`, `P8-O2`) | 4 nodes (`P7-O1`, `P3-O1`, `P4-O2`, `P8-O2`) |
| **Isolated (Degree 0) Nodes** | 0 | 0 | 0 |
| **Minimum Node Degree** | 1 | 1 | 1 |
| **Maximum Node Degree** | 10 | 8 | 6 |
| **Mean Node Degree** | 5.22 | 3.48 | 2.78 |
| **Median Node Degree** | 5.0 | 3.0 | 2.0 |
| **Degree Standard Deviation**| 2.38 | 1.93 | 1.32 |
| **Matrix Symmetry** | PASS ($|A - A^T| < 10^{-15}$) | PASS ($|A - A^T| < 10^{-15}$) | PASS ($|A - A^T| < 10^{-15}$) |
| **Self-Loops (Diagonal > 0)** | YES (All 23 nodes) | YES (All 23 nodes) | YES (All 23 nodes) |
| **Normalized Weight Range** | [0.0648, 0.3396] | [0.0771, 0.4228] | [0.1050, 0.4228] |

---

## 4. Recomputed Validation Performance Comparison

*Evaluated on full validation set: 293,410 windows across 82 recordings, 25 seizure events, 203.76 hours.*

| Metric | Graph A ($\theta = 0.25$) | Graph B ($\theta = 0.30$) | Graph C ($\theta = 0.35$, Ref) | Selection Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Validation AUPRC (Primary)** | $0.00152$ | **$0.00159$** | **$0.00159$** | **Tied: B & C** |
| **Validation AUROC** | $0.23005$ | **$0.25887$** | $0.25654$ | **Graph B (+0.0023)** |
| **Window Sensitivity** | $3.52\%$ | **$3.65\%$** | $2.71\%$ | **Graph B (+0.94%)** |
| **Window Specificity** | $90.03\%$ | $90.88\%$ | **$91.80\%$** | Graph C (+0.92%) |
| **Window Precision** | $0.00089$ | **$0.00101$** | $0.00083$ | **Graph B** |
| **Window F1 Score** | $0.00174$ | **$0.00197$** | $0.00162$ | **Graph B** |
| **Balanced Accuracy** | $0.46772$ | **$0.47268$** | $0.47253$ | **Graph B** |
| **Validation Event Sensitivity** | **$44.00\%$ (11/25)** | $36.00\%$ (9/25) | $36.00\%$ (9/25) | Graph A (+8.0%) |
| **Mean Detection Delay** | **$33.91$s** | $40.06$s | $39.89$s | Graph A (-5.98s) |
| **False Alarms / 24 Hours** | $3,438.10$ FA/24h | $3,143.39$ FA/24h | **$2,827.13$ FA/24h** | Graph C (-316 FA) |
| **Best Checkpoint Epoch** | Epoch 3 | Epoch 3 | Epoch 3 | All selected at Epoch 3 |
| **Checkpoint Path** | `theta_025/best_cnn_gnn.pt` | `theta_030/best_cnn_gnn.pt` | `exp_01/best_cnn_gnn.pt` | Verified on disk |

---

## 5. Patient-Independent Leakage Audit

Automated assertions executed against the dataset partitions:
1. **Patient Leakage:**
   - Train patients: `chb04, 09, 11-24` (16)
   - Val patients: `chb06, 07, 08, 10` (4)
   - Test patients: `chb01, 02, 03, 05` (4)
   - Intersection(Train, Val) = $\emptyset$ (**PASS**)
   - Intersection(Train, Test) = $\emptyset$ (**PASS**)
   - Intersection(Val, Test) = $\emptyset$ (**PASS**)
2. **Recording Leakage:** Zero recordings shared across splits (**PASS**).
3. **Window Leakage:** Zero window IDs shared across splits (**PASS**).
4. **Graph Estimation Scope:** Pearson correlation computed exclusively using training patients' EEG files (`training_correlation_matrix.npy`). Zero validation or test data accessed (**PASS**).
5. **Normalization Scope:** Local per-recording z-score normalization computed solely on each recording's own data. Zero test statistics leaked into train/val (**PASS**).

---

## 6. Graph Selection Verdict (Validation Data Only)

Based on the hierarchy in Section 6:
- **Primary Metric:** $\theta = 0.30$ and $\theta = 0.35$ tie at **AUPRC = $0.00159$**, outperforming $\theta = 0.25$ ($0.00152$).
- **Secondary Metric:** $\theta = 0.30$ achieves the highest **AUROC = $0.25887$** (vs $0.25654$ for $\theta = 0.35$), highest window sensitivity ($3.65\%$ vs $2.71\%$), and highest F1 score ($0.00197$ vs $0.00162$).
- **Structural Sparsity:** $\theta = 0.30$ (40 edges, density $15.81\%$) provides a regularized topology that avoids the excessive false alarm penalty of $\theta = 0.25$ (60 edges, $3,438.1$ FA/24h).

**Conclusion:** **$\theta = 0.30$ is formally confirmed as the selected spatial graph configuration.**
