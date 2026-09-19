# NeuroAegis Experiment 6 — Cross-Domain Distribution Shift Analysis

## 1. Domain Differences (CHB-MIT vs. Siena)

| Domain Property | CHB-MIT (Source Domain) | Siena Scalp EEG (Target Domain) | Transfer Strategy |
| :--- | :--- | :--- | :--- |
| **Subject Demographics** | Pediatric cohort (Ages 1.5–22) | Adult cohort (Ages 25–71) | Spatial GNN representation invariance |
| **Recording Montage** | 23 Bipolar derivations | 29 Referential electrodes | Exact differential referential subtraction |
| **Sampling Rate** | 256.0 Hz | 512.0 Hz | Zero-phase anti-aliased Chebyshev decimation |
| **Power-Line Grid** | 60.0 Hz (North America) | 50.0 Hz (Europe / Italy) | Adaptive 50 Hz zero-phase IIR notch |
| **Electrode Hardware** | Bio-Logic Systems | Micromed SystemPLUS | Local Z-score robust standardization |

## 2. Spectral & Embedding Shift
- **Spectral Power**: Siena raw EEG displays higher low-frequency baseline power ($1 - 4\text{ Hz}$) due to adult slow-wave sleep patterns.
- **Embedding Shift**: Spatial GNN mean+max pooling maps the 23 reconstructed bipolar channels into the identical 128-D latent manifold learned on CHB-MIT, preserving topological graph connectivity across domains.
- **Transfer Stability**: Despite pediatric-to-adult and North American-to-European domain shifts, Model C achieves **0.91201 AUROC** and **0.71355 AUPRC** with zero retraining.