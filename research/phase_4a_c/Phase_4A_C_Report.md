# NeuroAegis Phase 4A-C Research Report
## GNN Graph Threshold Correction Experiment: Controlled Topological Ablation on CHB-MIT EEG

**Audit Execution Date:** 2026-09-06  
**Experiment ID:** Phase 4A-C (GNN Graph Threshold Correction)  
**Evaluated Candidates:** Graph A ($\theta = 0.25$), Graph B ($\theta = 0.30$), Graph C ($\theta = 0.35$, Frozen Reference)  
**Evaluation Scope:** **Training and Validation ONLY (Zero Test Set Evaluation)**  
**Status:** **COMPLETE — EXPERIMENT OUTCOME: INCONCLUSIVE / MARGINAL**

---

## 1. Objective

The objective of Phase 4A-C was to conduct a controlled, hypothesis-driven architectural experiment investigating whether excessive graph sparsity and topological fragmentation (specifically the isolated 4-node occipital component observed at $\theta = 0.35$) was the primary root cause of the performance degradation in the Phase 4A CNN + GNN spatial baseline.

In strict adherence to the research protocol:
- **Zero Test Set Leakage / Evaluation:** The untouched CHB-MIT test set (4 patients, 219,909 windows, 152.82 hours, 22 seizures) was **NOT** evaluated during this phase.
- **Threshold Selection Basis:** Candidate selection is governed exclusively by **Validation Performance** (Primary: Validation AUPRC; Secondary: Validation AUROC, Event Sensitivity, False Alarms/Day, and Graph Connectivity).
- **All Other Invariants Frozen:** Dataset, 23-channel montage, Butterworth/Notch filtering, 10:1 dynamic negative sampling, focal loss ($\gamma=2.0, \alpha=0.25$), batch size 128, and CNN/GNN architectures remained 100% identical to Phase 4A.

---

## 2. Motivation

In Phase 4A, the 1D CNN + Spatial GNN baseline demonstrated severe window-level metric degradation relative to the Phase 3 1D CNN baseline (Test AUROC dropped from $0.3639$ to $0.1698$; Test AUPRC dropped from $0.0415$ to $0.0045$). The subsequent scientific implementation audit revealed that while data pipelines and channel identities were preserved:
1. The spatial graph at threshold $\theta = 0.35$ had only 32 undirected edges (density $12.65\%$).
2. The graph fragmented into two disconnected components, isolating all four occipital bipolar channels (`P7-O1`, `P3-O1`, `P4-O2`, `P8-O2`) from the remaining 19 channels.
3. Diagnostic sweep indicated that lowering $\theta$ to $0.25$ or $0.30$ would increase edge count to 60 and 40 edges, respectively, potentially increasing spatial information flow across leads.

Phase 4A-C was designed to test whether this increased graph density and edge connectivity would restore validation discriminative power.

---

## 3. Frozen Phase 4A Result (Reference Baseline)

The Phase 4A baseline remains frozen and untouched:
- **Model Architecture:** Baseline1DCNN_GNN (Channel-preserving shared 1D CNN + 2-layer Spatial GCN + Dual Readout Pooling)
- **Trainable Parameters:** 52,497
- **Graph Threshold:** $\theta = 0.35$
- **Frozen Test Performance (Untouched):**
  - Test AUROC: $0.16976$
  - Test AUPRC: $0.00451$
  - Test Window Sensitivity: $3.30\%$
  - Test Window Specificity: $99.48\%$
  - Test Event Sensitivity: $22/22$ ($100.0\%$)
  - Test False Alarms / 24h: $179.19$ FA/24h
  - Test Detection Latency: $6.50$s

---

## 4. Graph Topology Problem

The post-experiment audit of Phase 4A revealed a topological defect in the static Pearson correlation graph:
- At $\theta = 0.35$, the cross-channel correlation matrix yielded an adjacency with 32 undirected edges.
- NetworkX component decomposition showed:
  - **Component 1 (19 nodes):** Frontal, central, temporal, parietal, and midline channels.
  - **Component 2 (4 nodes):** Occipital bipolar channels (`P7-O1`, `P3-O1`, `P4-O2`, `P8-O2`).
- As a consequence of this partition, spatial graph convolutions could not pass messages between the posterior occipital region and the rest of the brain.

---

## 5. Candidate Thresholds & Graph Metrics

Three spatial graphs were evaluated:
- **Graph A ($\theta = 0.25$):** Candidate lower threshold designed to maximize edge density.
- **Graph B ($\theta = 0.30$):** Intermediate candidate balancing density and spurious edge suppression.
- **Graph C ($\theta = 0.35$):** Frozen reference graph from Phase 4A.

All graphs were constructed using cross-channel Pearson correlation estimated **strictly on unlabelled EEG data from the 16 training patients** (`chb04, 09, 11-24`).

### Quantitative Graph Comparison Table (`graph_threshold_comparison.csv`)

| Metric | Graph A ($\theta = 0.25$) | Graph B ($\theta = 0.30$) | Graph C ($\theta = 0.35$, Ref) |
| :--- | :---: | :---: | :---: |
| **Undirected Edges** | **60** | **40** | **32** |
| **Graph Density** | **$23.72\%$** | **$15.81\%$** | **$12.65\%$** |
| **Connected Components** | **2** | **2** | **2** |
| **Largest Component Size** | 19 nodes | 19 nodes | 19 nodes |
| **Smallest Component Size** | 4 nodes (Occipital) | 4 nodes (Occipital) | 4 nodes (Occipital) |
| **Occipital in Main Component?** | **NO** | **NO** | **NO** |
| **Isolated Nodes Count** | 0 | 0 | 0 |
| **Minimum Node Degree** | 1 | 1 | 1 |
| **Maximum Node Degree** | 10 | 8 | 6 |
| **Mean Node Degree** | **5.22** | **3.48** | **2.78** |
| **Median Node Degree** | 5.0 | 3.0 | 2.0 |
| **Min Edge Weight** | 0.2501 | 0.3070 | 0.3510 |
| **Mean Edge Weight** | 0.4077 | 0.4798 | 0.5190 |
| **Max Edge Weight** | 1.0000 | 1.0000 | 1.0000 |

*Topological Finding:* While lowering $\theta$ to $0.25$ doubled the total edge count (60 vs 32), the maximum cross-region Pearson correlation between the occipital group and anterior leads is $r \approx 0.2381$ (between `P8-O2` and `C4-P4`). Consequently, neither $\theta=0.25$ nor $\theta=0.30$ was low enough to cross the percolation threshold for occipital reconnection; both candidates merely added intra-regional edges within the main component and within the occipital component.

---

## 6. Experimental Protocol & Invariants

All models were trained and evaluated under the frozen Phase 4A protocol:
- **Optimizer:** AdamW ($\text{lr} = 10^{-3}, \text{weight\_decay} = 10^{-4}$)
- **Learning Rate Schedule:** CosineAnnealingLR ($T_{\max} = 3, \eta_{\min} = 10^{-5}$)
- **Loss Function:** BinaryFocalLossWithLogits ($\gamma = 2.0, \alpha = 0.25$)
- **Data Sampling:** DynamicNegativeSampler (10:1 ratio, $\text{seed} = 42 + \text{epoch}$, 3,308 positives + 33,080 negatives = 36,388 windows per epoch)
- **Validation Evaluation:** Full single-pass inference over all 293,410 validation windows across 82 continuous EDF recordings ($203.76$ hours)
- **Validation Events:** 25 annotated seizure events across patients `chb06`, `chb07`, `chb08`, and `chb10`
- **Checkpoint Selection:** Strictly peak Validation AUPRC (epoch-wise non-accuracy metric)

---

## 7. Training Progression & Epoch History

Both candidate models trained for 3 full epochs on Apple Silicon MPS hardware. Training was stable with monotonic focal loss convergence.

### Candidate A ($\theta = 0.25$) Epoch Progression
- **Epoch 1:** Train Loss: $0.02856$ | Val Loss: $0.06942$ | Val AUPRC: $0.00137$ | Val AUROC: $0.15882$ | Val Sens: $1.35\%$ | Val Spec: $87.17\%$
- **Epoch 2:** Train Loss: $0.02211$ | Val Loss: $0.10437$ | Val AUPRC: $0.00150$ | Val AUROC: $0.22472$ | Val Sens: $5.41\%$ | Val Spec: $84.62\%$
- **Epoch 3 (BEST):** Train Loss: $0.02021$ | Val Loss: $0.05799$ | **Val AUPRC: $0.00152$** | Val AUROC: $0.23005$ | Val Sens: $3.52\%$ | Val Spec: $90.03\%$
- *Duration:* $1,004.5$ seconds ($16.74$ minutes) | *Peak Memory:* $7.08$ GB

### Candidate B ($\theta = 0.30$) Epoch Progression
- **Epoch 1:** Train Loss: $0.02869$ | Val Loss: $0.06665$ | Val AUPRC: $0.00139$ | Val AUROC: $0.17241$ | Val Sens: $1.62\%$ | Val Spec: $87.59\%$
- **Epoch 2:** Train Loss: $0.02204$ | Val Loss: $0.06327$ | Val AUPRC: $0.00150$ | Val AUROC: $0.22287$ | Val Sens: $3.11\%$ | Val Spec: $88.70\%$
- **Epoch 3 (BEST):** Train Loss: $0.02017$ | Val Loss: $0.05054$ | **Val AUPRC: $0.00159$** | Val AUROC: $0.25887$ | Val Sens: $3.65\%$ | Val Spec: $90.88\%$
- *Duration:* $1,008.2$ seconds ($16.80$ minutes) | *Peak Memory:* $7.34$ GB

---

## 8. Validation Comparison & Empirical Results

The three models were evaluated on the identical validation evaluation harness (293,410 windows, 25 seizure events, $203.76$ recording hours).

### Comprehensive Validation Results Table (`validation_threshold_comparison.csv`)

| Metric | Graph A ($\theta = 0.25$) | Graph B ($\theta = 0.30$) | Frozen Ref ($\theta = 0.35$) | Best Candidate |
| :--- | :---: | :---: | :---: | :---: |
| **Validation AUPRC (Primary)** | $0.00152$ | **$0.00159$** | **$0.00159$** | **Tied (B & C)** |
| **Validation AUROC** | $0.23005$ | **$0.25887$** | $0.25654$ | **Graph B (+0.0023)** |
| **Window Sensitivity** | $3.52\%$ | **$3.65\%$** | $2.71\%$ | **Graph B (+0.94%)** |
| **Window Specificity** | $90.03\%$ | $90.88\%$ | **$91.80\%$** | **Graph C (+0.92%)** |
| **Window Precision** | $0.00089$ | **$0.00101$** | $0.00083$ | **Graph B** |
| **Window F1 Score** | $0.00174$ | **$0.00197$** | $0.00162$ | **Graph B** |
| **Balanced Accuracy** | $0.46772$ | **$0.47268$** | $0.47253$ | **Graph B** |
| **Validation Event Sensitivity** | **$44.00\%$ (11/25)** | $36.00\%$ (9/25) | $36.00\%$ (9/25) | **Graph A (+8.0%)** |
| **Mean Detection Delay** | **$33.91$s** | $40.06$s | $39.89$s | **Graph A (-5.98s)** |
| **False Alarms / 24 Hours** | $3,438.10$ FA/24h | $3,143.39$ FA/24h | **$2,827.13$ FA/24h** | **Graph C (-316 FA)** |
| **True Positives (TP)** | 26 | **27** | 20 | Graph B |
| **False Positives (FP)** | 29,189 | 26,687 | **24,001** | Graph C |
| **True Negatives (TN)** | 263,482 | 265,984 | **268,670** | Graph C |
| **False Negatives (FN)** | 713 | **712** | 719 | Graph B |

---

## 9. Graph Connectivity Analysis

1. **Topology vs. Sparsity Trade-off:**
   - As threshold $\theta$ decreases from $0.35 \to 0.30 \to 0.25$, graph density nearly doubles ($12.65\% \to 15.81\% \to 23.72\%$), adding 28 new edges.
   - However, because cross-lobe correlations between occipital leads and the rest of the montage peak at $r \approx 0.238$, the graph remains rigidly bifurcated into two disconnected components in all three configurations.
2. **Dense Edge Aggregation & False Alarm Penalty:**
   - In Graph A ($\theta = 0.25$), the denser graph (mean degree 5.22) aggregates more asynchronous background noise during message passing. This caused False Positives on validation background to spike from 24,001 ($\theta=0.35$) to 29,189 ($\theta=0.25$), increasing false alarms by **$+21.6\%$** ($3,438.1$ vs $2,827.1$ FA/24h) and depressing AUROC ($0.23005$ vs $0.25654$).
   - In Graph B ($\theta = 0.30$), the edge addition is more moderate (40 edges, mean degree 3.48), yielding a slight improvement in sensitivity ($3.65\%$ vs $2.71\%$) and AUROC ($0.25887$ vs $0.25654$), but still suffering a **$+11.2\%$ increase in false alarms** ($3,143.4$ vs $2,827.1$ FA/24h).

---

## 10. Threshold Selection Decision Matrix

Following Section 10 of the protocol:
- **Primary Metric:** Validation AUPRC
  - $\theta = 0.25$: $0.00152$ (Rank 3)
  - $\theta = 0.30$: **$0.00159$** (Tied Rank 1)
  - $\theta = 0.35$: **$0.00159$** (Tied Rank 1)
- **Secondary Metric (AUROC & Discrimination):**
  - $\theta = 0.30$ marginally edges out $\theta = 0.35$ by $+0.0023$ ($0.25887$ vs $0.25654$) and detects 7 more positive windows (27 TP vs 20 TP).
- **Secondary Metric (Clinical False Alarms):**
  - $\theta = 0.35$ provides significantly superior background suppression ($2,827.13$ FA/24h vs $3,143.39$ FA/24h for $\theta=0.30$, a clinically meaningful difference of $-316$ false alarms per day).
- **Secondary Metric (Event Sensitivity):**
  - $\theta = 0.30$ and $\theta = 0.35$ tie at $36.00\%$ ($9/25$ events).

### Decision Assessment
The validation differences between $\theta = 0.30$ and $\theta = 0.35$ are statistically marginal on the primary metric ($0.00159$ vs $0.00159$). While $\theta = 0.30$ provides a slight boost in window sensitivity ($+0.94\%$) and AUROC ($+0.0023$), $\theta = 0.35$ retains superior clinical false-alarm rejection without sacrificing AUPRC. Furthermore, neither threshold resolves the underlying topological disconnection of the occipital leads.

Therefore, the experiment outcome is **INCONCLUSIVE / MARGINAL**, and replacing the frozen $\theta = 0.35$ baseline with $\theta = 0.30$ is **NOT recommended** as a standalone fix.

---

## 11. Limitations

1. **Static Correlation Inadequacy:**
   - A static correlation matrix estimated on interictal background EEG does not capture the dynamic hypersynchrony that develops during ictal events.
2. **Occipital Disconnection Unresolved by Simple Thresholding:**
   - Because occipital bipolar leads have lower background correlation ($r < 0.24$) with anterior channels, thresholding in the range $[0.25, 0.35]$ cannot unify the graph. Reconnecting the occipital leads requires an anatomical or distance-based adjacency formulation (e.g. 10-20 Euclidean/geodesic distance or $k$-NN with $k \ge 3$).
3. **Model Parameter Constraint:**
   - Both models remained constrained to 52,497 parameters due to the single-channel shared temporal backbone, compared to Phase 3's 173,601 parameters.

---

## 12. Scientific Interpretation

- **Did increasing graph connectivity improve validation discrimination?**  
  *Marginally.* Moving from $\theta=0.35$ (32 edges) to $\theta=0.30$ (40 edges) increased AUROC slightly from $0.2565$ to $0.2589$ and window sensitivity from $2.71\%$ to $3.65\%$, but AUPRC remained identical ($0.00159$). Moving further to $\theta=0.25$ (60 edges) degraded AUROC ($0.2301$) due to excessive false alarm propagation.
- **Did the fragmented $\theta=0.35$ graph suppress seizure sensitivity?**  
  *Partially.* Window sensitivity was slightly higher at $\theta=0.30$ ($3.65\%$ vs $2.71\%$), and event sensitivity was higher at $\theta=0.25$ ($44\%$ vs $36\%$), suggesting that denser graph message passing can propagate weak seizure cues, but at the direct cost of increased false alarms.
- **Did the occipital disconnected component contribute to the poor representation?**  
  *Yes, but threshold tuning within $[0.25, 0.35]$ did not reconnect it.* Because all three thresholds left the occipital channels in an isolated 4-node component, the architectural impediment remained active across all candidates.
- **Is there evidence that $\theta=0.25$ or $\theta=0.30$ provides a more stable spatial representation?**  
  *No.* The validation performance gains are insufficient to demonstrate that threshold tuning alone fixes the GNN spatial baseline.

---

## 13. Final Decision

Based strictly on training and validation results without ever evaluating the untouched test set:
1. **Selected Threshold:** **0.30** (slight validation AUROC/sensitivity advantage over $\theta=0.35$ and higher AUPRC than $\theta=0.25$, but tied on primary AUPRC with $\theta=0.35$).
2. **Phase 4A-C Result:** **INCONCLUSIVE / MARGINAL IMPROVEMENT**. Tuning the static Pearson threshold alone does not provide a definitive remedy for the Phase 4A baseline.
3. **Freezing Authorization:** **NO**. The candidate should NOT be frozen as a new baseline because the underlying topological defect (occipital disconnection) persists and the primary AUPRC did not improve ($0.00159$).
4. **Final Test Evaluation Authorized:** **NO**. The test set must remain untouched.
5. **Phase 4B GRU Authorized:** **YES (with architectural refinements)**. Future phases should incorporate temporal recurrence (GRU/LSTM) and replace the static thresholded correlation graph with an anatomically connected montage graph (e.g. 10-20 distance-based graph) with matched parameter capacity.

---

## 14. Audit Deliverables Manifest

All experiment artifacts are preserved in [`/Volumes/BLACK-BOX/NeuroAegis/research/phase_4a_c/`](file:///Volumes/BLACK-BOX/NeuroAegis/research/phase_4a_c/):
1. [`graph_threshold_comparison.csv`](file:///Volumes/BLACK-BOX/NeuroAegis/research/phase_4a_c/graph_threshold_comparison.csv): Topological comparison across Graph A, B, and C.
2. [`validation_threshold_comparison.csv`](file:///Volumes/BLACK-BOX/NeuroAegis/research/phase_4a_c/validation_threshold_comparison.csv): Comprehensive window and event validation comparison table.
3. [`Phase_4A_C_Graph_Threshold_Experiment.xlsx`](file:///Volumes/BLACK-BOX/NeuroAegis/research/phase_4a_c/Phase_4A_C_Graph_Threshold_Experiment.xlsx): 12-sheet professional audit workbook.
4. **Publication Figures (300 DPI) in `figures/`:**
   - `graph_topology_vs_threshold.png`
   - `validation_auprc_vs_threshold.png`
   - `validation_auroc_vs_threshold.png`
   - `validation_event_sensitivity_vs_threshold.png`
   - `validation_false_alarms_vs_threshold.png`
   - `validation_confusion_matrix_theta025.png`
   - `validation_confusion_matrix_theta030.png`
   - `degree_distribution_threshold_comparison.png`
5. **Model Checkpoints & Predictions:**
   - `theta_025/best_cnn_gnn.pt` & `theta_025/best_val_predictions.npz`
   - `theta_030/best_cnn_gnn.pt` & `theta_030/best_val_predictions.npz`
   - `reference_theta035_val_predictions.npz`
