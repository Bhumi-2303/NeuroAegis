# NeuroAegis: Final Authoritative Research Results Summary

## 1. Primary Scientific Findings
NeuroAegis establishes an end-to-end, patient-independent epileptic seizure detection framework combining channel-preserving 1D Convolutional Neural Networks, Spatial Graph Neural Networks, and Causal Gated Recurrent Units (**CNN + Spatial GNN + Causal GRU**).

Evaluated on the held-out CHB-MIT test cohort comprising **4 unseen patients, 155 continuous EDF recordings, 22 clinical seizure events, 152.82 continuous monitoring hours, and 219,909 evaluation windows (5.0s windows with 50% temporal overlap)**, the frozen model ($	au = 0.50$, 91,858 parameters) achieved:

- **Event Sensitivity**: **95.45%** (21 of 22 seizure events detected; 1 missed: `chb01_15`).
- **Window Sensitivity**: **83.83%** (534 true positive windows).
- **Window Specificity**: **99.82%** (218,873 true negative windows).
- **Precision (PPV)**: **57.24%** (534 true positive windows out of 933 total alarms against natural 344:1 class imbalance).
- **F1 Score**: **0.68025**.
- **Balanced Accuracy**: **91.82%**.
- **AUROC**: **0.98970**.
- **AUPRC**: **0.80681**.
- **False Alarm Rate**: **62.66 alarms / 24 hours** (a **96.8% reduction** compared to the 1D-CNN baseline of 1,946.56/24h).
- **Mean Detection Delay**: **10.57 seconds** (Median: **9.0 seconds**, 85.7% detected within 15 seconds).
- **Inference Latency**: **1.42 ms / window** (1,760x faster than real-time on consumer Apple Silicon MPS).

---

## 2. Component Ablation Evidence
1. **Temporal 1D-CNN Baseline (Model A)**: 173,601 parameters. While detecting events under nominal overlap, under strict continuous alerting it exhibited catastrophic false alarm contamination (**1,946.56 FA/24h**, 12,395 false positives), yielding an AUPRC of 0.0415 and F1 of 0.0155.
2. **Spatial Graph Convolution Addition (Model B)**: 52,497 parameters ($	heta = 0.30$, 40 undirected edges). Spatial filtering dramatically suppressed false alarms by **89.7%** (down to 200.39 FA/24h); however, lacking temporal dynamics, event sensitivity collapsed to **27.27%** (6/22 events detected).
3. **Causal Temporal GRU Integration (Model C)**: 91,858 parameters ($L=8$, 22.5s context). Causal sequential recurrence restored event sensitivity to **95.45%** (21/22), suppressed false alarms further to **62.66 FA/24h** (-96.8% vs Model A), and surged AUPRC to **0.80681**.

---

## 3. External Cross-Domain Evaluation (Siena Scalp EEG)
On an external benchmark subset from the University of Siena Hospital (**2 patients, 4 recordings, 4 events, 3,538 windows, 2.46 hours**):
- **Zero-Shot Transfer**: Detected **4 of 4 events (100%)**, achieving window specificity of **99.85%**, AUROC of **0.9120**, and AUPRC of **0.7140** with 0 false alarms.
- **Post-Hoc Adaptation**: Calibrating temperature ($T^* = 0.3495$) and threshold ($	au^* = 0.3800$) on PN00 improved held-out PN12 test F1 from **0.3043 to 0.3750** (+23.2% relative gain) with 100% precision.
