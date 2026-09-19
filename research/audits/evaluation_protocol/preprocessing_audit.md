# NeuroAegis — Preprocessing & Signal Harmonization Audit

**Standard**: Evaluation Protocol V1.0  
**Scope**: CHB-MIT (Source Domain) vs. Siena Scalp EEG (Target Domain)

---

## 1. CHB-MIT Source Domain Preprocessing

The CHB-MIT dataset was collected at Boston Children's Hospital and preprocessed under the following frozen pipeline:
- **Electrode Montage**: 23 canonical bipolar double-banana channels:
  - Longitudinal left temporal: `FP1-F7, F7-T7, T7-P7, P7-O1`
  - Longitudinal left parasagittal: `FP1-F3, F3-C3, C3-P3, P3-O1`
  - Longitudinal right parasagittal: `FP2-F4, F4-C4, C4-P4, P4-O2`
  - Longitudinal right temporal: `FP2-F8, F8-T8, T8-P8, P8-O2`
  - Transverse coronal / central: `FZ-CZ, CZ-PZ`
  - Auxiliary temporal / mastoid: `T7-FT9, FT9-TO1, T8-FT10, FT10-TO2, P7-T7`
- **Native Sampling Frequency**: $256.0\text{ Hz}$.
- **Bandpass Filtering**: $0.5\text{ Hz} - 70.0\text{ Hz}$ zero-phase Butterworth filter (4th order).
- **Notch Filtering**: $60.0\text{ Hz}$ zero-phase IIR notch filter ($Q=30.0$, North American power line).
- **Window Slicing**: $5.0\text{s}$ window length ($1,280\text{ samples}$), $2.5\text{s}$ stride ($640\text{ samples}$, $50\%$ overlap).
- **Normalization**: Per-channel robust z-score standardization computed on training-set baseline statistics:
  $$x_{\text{norm}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

---

## 2. Siena Target Domain Preprocessing & Harmonization

The Siena Scalp EEG dataset (University of Siena, Italy) was recorded from adult patients using a referential (monopolar) electrode layout:
- **Native Sampling Frequency**: $512.0\text{ Hz}$.
- **Native Channels**: 29 monopolar electrodes referenced to an earlobe/vertex common electrode.
- **Harmonization Steps**:
  1. **Montage Reconstruction**: Converted from referential to the exact 23 canonical bipolar double-banana channels via differential subtraction ($V_A - V_B$).
  2. **Anti-Aliased Decimation**: Downsampled $512.0\text{ Hz} \to 256.0\text{ Hz}$ using an 8th-order zero-phase Chebyshev Type I decimation filter.
  3. **European Power-Line Filtering**: Replaced $60\text{ Hz}$ notch with a $50.0\text{ Hz}$ zero-phase IIR notch filter ($Q=30.0$).
  4. **Target-Domain Normalization**: Label-free recording-level median/IQR standardization computed dynamically per recording:
     $$x_{\text{norm}} = \frac{x - \text{median}(x)}{\text{IQR}(x) / 1.349}$$

---

## 3. Preprocessing Audit Summary

| Parameter | CHB-MIT (Source Domain) | Siena (Target Domain) | Harmonization Method |
| :--- | :--- | :--- | :--- |
| **Native Sampling Rate** | $256.0\text{ Hz}$ | $512.0\text{ Hz}$ | Anti-aliased Chebyshev decimation ($512 \to 256\text{ Hz}$) |
| **Input Montage** | Bipolar Double-Banana (23 ch) | Monopolar Referential (29 ch) | Exact differential subtraction ($V_A - V_B$) |
| **Power-Line Notch** | $60.0\text{ Hz}$ (USA) | $50.0\text{ Hz}$ (Europe) | Zero-phase IIR notch filter ($Q=30.0$) |
| **Window Duration** | $5.0\text{ seconds}$ ($1,280\text{ samples}$) | $5.0\text{ seconds}$ ($1,280\text{ samples}$) | Identical |
| **Window Stride** | $2.5\text{ seconds}$ ($640\text{ samples}$) | $2.5\text{ seconds}$ ($640\text{ samples}$) | Identical ($50\%$ overlap) |
| **Normalization Scope** | Frozen Source-Domain $\mu, \sigma$ | Label-Free Target-Domain Robust Z-score | Documented as **Label-Free Target Normalization** |
