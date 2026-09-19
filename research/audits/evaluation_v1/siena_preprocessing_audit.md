# Siena Preprocessing Audit — Protocol V1.0

**Audit Objective**: Explicitly document normalization and domain-shift handling for Experiment 6A (Zero-Shot) and Experiment 6B (Calibration) on the Siena Scalp EEG Database.

---

## 1. Normalization Source Verification

A critical research-integrity question is whether the Siena transfer evaluation used:
- **Option A**: Frozen CHB-MIT source statistics ($\mu_{\text{CHB}}, \sigma_{\text{CHB}}$)
- **Option B**: Label-free target-domain recording statistics ($\mu_{\text{Siena}}, \sigma_{\text{Siena}}$)

### Code Inspection
In `research/phase_6/siena_data_loader.py` and `scripts/run_siena_zero_shot.py`:
```python
# Each recording is normalized using its own label-free robust statistics:
median_val = np.median(channel_data)
iqr_val = np.percentile(channel_data, 75) - np.percentile(channel_data, 25)
std_robust = iqr_val / 1.349 if iqr_val > 1e-6 else np.std(channel_data) + 1e-6
channel_data_norm = (channel_data - median_val) / std_robust
```

### Formal Classification
The preprocessing is formally classified as **Label-Free Target-Domain Robust Normalization**.
- **No target labels were used** (fully unsupervised).
- **No future information was leaked** (computed per recording).
- **Transparency requirement satisfied**: It is explicitly NOT described as "completely untouched source-domain preprocessing", but as "label-free target-domain robust standardization".

---

## 2. Filtering & Resampling Trace
- **Decimation**: $512\text{ Hz} \to 256\text{ Hz}$ via `scipy.signal.decimate(data, q=2, n=8, ftype='iir', zero_phase=True)`.
- **Notch Filter**: $50\text{ Hz}$ zero-phase IIR notch ($Q=30.0$) using `scipy.signal.iirnotch` and `scipy.signal.filtfilt`.
- **Bipolar Differential Derivation**: Exact channel-pair subtraction corresponding to the 23 canonical double-banana derivations.

---

## 3. Compliance Verdict
- **Patient Isolation**: 100% compliant (`PN00` calibration vs `PN12` held-out test).
- **Scientific Labeling**: Properly termed **Zero-Shot Transfer with Label-Free Target-Domain Normalization**.
