# NeuroAegis Experiment 6: Cross-Domain Generalization (CHB-MIT -> Siena)

This directory contains the complete artifacts, configurations, and reports for **Experiment 6: Cross-Domain Zero-Shot Generalization** evaluating the frozen **NeuroAegis Model C** on the external **Siena Scalp EEG Database**.

## Key Findings:
- **Event Sensitivity**: **4/4 (100.0%)** clinical seizures detected zero-shot.
- **Clinical False Alarm Rate**: **0.00 FA/day** (0 false alarm episodes across 2.67 hours).
- **Discrimination**: **0.91201 AUROC** | **0.71355 AUPRC** on complete recording sessions.
- **Memory Safety**: Peak RSS < 2.5 GB, chunked streaming inference on Apple M4 MPS.

## Deliverables:
- `protocol.md`: Strict zero-shot transfer protocol
- `dataset_audit.md`: Dataset audit and availability breakdown
- `channel_mapping.csv` & `channel_mapping_report.md`: 23-channel topological reconstruction
- `zero_shot_config.yaml`: Frozen execution configuration
- `patient_results.csv`: Patient-level metrics
- `recording_results.csv`: Recording-level metrics
- `aggregate_metrics.json` & `aggregate_metrics.md`: Master metrics
- `error_analysis.md`: Detailed event and false alarm diagnostics
- `distribution_shift.md`: CHB-MIT vs Siena domain shift analysis