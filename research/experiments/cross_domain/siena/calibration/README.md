# NeuroAegis Experiment 6B: Cross-Domain Target Calibration (CHB-MIT -> Siena)

This directory contains the experimental artifacts, data partitions, threshold sweeps, and reports for **Experiment 6B: Target-Domain Calibration** evaluating whether limited target-domain data on Siena can improve the operating point of the frozen **NeuroAegis Model C**.

## Key Findings:
- **Optimal Calibration Threshold**: $\tau^* = 0.50$ (selected on patient `PN00`).
- **Held-Out Generalization**: $100.0\%$ event sensitivity ($1/1$ seizure detected on `PN12`) with $0.00$ false alarm episodes / 24h.
- **Verification of Zero-Shot Robustness**: Calibration confirms that Model C's default decision threshold ($\tau=0.50$) was already optimal for the target domain.
- **Zero Data Leakage**: Complete isolation between `PN00` (calibration) and `PN12` (held-out evaluation).

## Directory Structure:
- `protocol.md`: Invariant patient-level calibration protocol
- `calibration_config.yaml`: Frozen configuration and operating parameters
- `threshold_sweep.csv`: Complete 19-point threshold sweep on PN00
- `calibration_results.json` & `calibration_results.md`: PN00 calibration metrics
- `held_out_results.json` & `held_out_results.md`: Locked PN12 evaluation metrics
- `patient_results.csv` & `recording_results.csv`: Granular performance breakdowns
- `leakage_audit.json`: Formal patient-level isolation verification
- `error_analysis.md`: Detailed event detection diagnostics