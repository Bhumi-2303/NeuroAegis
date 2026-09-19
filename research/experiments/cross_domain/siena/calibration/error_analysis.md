# NeuroAegis Experiment 6B — Calibration Error & Detection Diagnostics

## 1. Calibration Patient PN00 Seizure Events
1. **`PN00_sz01`** (Recording: `PN00/PN00-1.edf`, Duration: $70.0\text{s}$):
   - Detected: YES at $t=1157.5\text{s}$ (Delay: $14.50\text{s}$)
2. **`PN00_sz04`** (Recording: `PN00/PN00-4.edf`, Duration: $74.0\text{s}$):
   - Detected: YES at $t=1022.5\text{s}$ (Delay: $16.50\text{s}$)
3. **`PN00_sz05`** (Recording: `PN00/PN00-5.edf`, Duration: $67.0\text{s}$):
   - Detected: YES at $t=922.5\text{s}$ (Delay: $18.50\text{s}$)

## 2. Held-Out Patient PN12 Seizure Event
1. **`PN12_sz03`** (Recording: `PN12/PN12-3.edf`, Duration: $96.0\text{s}$):
   - Detected: YES at $t=800.0\text{s}$ (Delay: $28.00\text{s}$)
   - False Alarms: 0 episodes (0 FP windows across entire 0.55h session)

## 3. Threshold Robustness
- Lowering $\tau$ to $0.10$ increases window sensitivity to $46.15\%$ on PN12 with only 2 FP windows ($99.74\%$ specificity).
- The standard $\tau=0.50$ operating point provides optimal trade-off with $0.00$ false alarm episodes.