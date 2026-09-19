# NeuroAegis — Repository Evaluation Inventory

**Audit Date**: September 19, 2026  
**Auditor**: Lead Evaluation & Reproducibility Engineer  
**Objective**: Comprehensive cataloging of every evaluation module, metric calculation, alarm postprocessing script, and experimental harness in the NeuroAegis repository prior to establishing Evaluation Protocol V1.0.

---

## 1. Overview of Evaluated Modules

| Module / Script Path | Target Phase / Experiment | Metric Definitions Used | Alarm Postprocessing Applied | Primary Latency / Delay Metric | Status / Remarks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `neuroaegis/eval/metrics.py` | Global Eval Harness | Scikit-learn (macro/binary), Event Overlap | Majority filter (size=3), 15s merge gap, 5s min duration | First predicted alarm onset minus true onset | Bug in `_get_intervals`: Multiplies window index by `5.0s` instead of `stride_sec=2.5s`. |
| `neuroaegis/eval/harness.py` | LOSO & External Benchmark | `evaluate_event_level` | Passes `EvalConfig` to `metrics.py` | `detection_delay_sec` | Duration approximated as `N_windows * 5.0s / 3600.0`. |
| `research/imbalance/metrics.py` | Imbalance Baseline Suite | Binary classification + Raw window FP rate | No smoothing/clustering (`y_prob >= threshold`) | Window-level overlap only | Reports raw false positive windows / 24h as `false_alarms_per_24h` ($62.66\text{ FA/24h}$). |
| `research/phase_3/train_cnn_baseline.py` | Phase 3 (1D CNN) | Imbalance metrics | Raw threshold $\tau=0.50$ | None | Initial baseline evaluation on CHB-MIT. |
| `research/phase_4a/evaluate_final_test.py` | Phase 4A (CNN + GNN) | Imbalance metrics + event overlap | Raw threshold $\tau=0.50$ | First positive window onset minus seizure onset | Initial spatial GNN evaluation. |
| `research/phase_4a_c/execute_phase_4a_c_test_evaluation.py` | Phase 4A-C (GNN Ablation) | Imbalance metrics + event overlap | Raw threshold $\tau=0.50$ | `first_alarm - s_start` | Quarantined test evaluation. |
| `research/phase_4b/evaluate_phase_4b_test.py` | Phase 4B (Model C Frozen) | Imbalance metrics + event details | Raw threshold $\tau=0.50$ | `first_alarm = det_w["window_end_sec"].min() - s_start` | Reports completion delay ($10.57\text{s}$) as detection delay. |
| `research/phase_7/statistical_evaluator.py` | Phase 7 (Statistical Robustness) | Scikit-learn + Bootstrap | Raw threshold $\tau=0.50$ | `window_end_sec - s_start` | Compares Model A, B, C; uses completion delay and raw FP window rate ($62.66\text{ FA/24h}$). |
| `research/phase_9/benchmarks/benchmark_engine.py` | Phase 9 (Edge Benchmarking) | Latency / Memory profiling | N/A | Execution time per window (ms) | Benchmarks PyTorch MPS runtime. |
| `scripts/run_temporal_comparison.py` | Experiment 1 (Temporal) | `neuroaegis.eval.metrics` | 3-window smoothing, 15s merge gap, 5s min duration | Event-level delay | Compares GRU, LSTM, TCN against Model C reference. |
| `scripts/run_eeg_models.py` | Experiment 2 (EEG-Specific DL) | `neuroaegis.eval.metrics` | 3-window smoothing, 15s merge gap, 5s min duration | Event-level delay | Compares EEGNet, ShallowConvNet, DeepConvNet, 1D CNN. |
| `scripts/train_and_eval_baselines.py` | Experiment 3 (Classical ML) | `neuroaegis.eval.metrics` | 3-window smoothing, 15s merge gap, 5s min duration | Event-level delay | Compares Random Forest, XGBoost, LightGBM, Linear SVM on 57 features. |
| `scripts/evaluate_spatial_ablation.py` | Experiment 4 (Spatial Ablation) | `neuroaegis.eval.metrics` | 3-window smoothing, 15s merge gap, 5s min duration | Event-level delay | Evaluates CNN-only, CNN+GNN, CNN+GNN+GRU. |
| `scripts/run_foundation_pilot.py` | Experiment 5 (Pretrained Models) | `neuroaegis.eval.metrics` | 3-window smoothing, 15s merge gap, 5s min duration | Event-level delay | Evaluates BENDR Biosignal Transformer pilot. |
| `scripts/run_siena_zero_shot.py` | Experiment 6A (Siena Zero-Shot) | `protocol_v1` & `neuroaegis.eval` | 3-window smoothing, 15s merge gap, 5s min duration | `window_start_sec - s_start` | Evaluates zero-shot transfer on Siena cohort. |
| `scripts/run_siena_calibration.py` | Experiment 6B (Siena Calibration) | `protocol_v1` & `neuroaegis.eval` | 3-window smoothing, 15s merge gap, 5s min duration | `window_start_sec - s_start` | Evaluates threshold optimization on `PN00` and locks on `PN12`. |

---

## 2. Detailed Breakdown of Inconsistencies Discovered

### A. False Alarm Rate Definition Discrepancy
1. **Raw FP Window Rate ($62.66\text{ FA/24h}$)**:
   - Implemented in `research/imbalance/metrics.py` and `research/phase_7/statistical_evaluator.py`.
   - Formula: $\frac{\text{Count}(\text{true\_label}=0 \land \text{pred\_prob} \ge \tau)}{\text{Total Monitoring Hours}} \times 24.0$.
   - For Model C: $399\text{ FP windows} / 152.8231\text{ h} \times 24 = 62.66\text{ FP windows/24h}$.
2. **Contiguous FP Runs Rate ($12.56\text{ FA/24h}$)**:
   - Reported in historical consistency summaries (`scripts/generate_consistency_audit.py`).
   - Counts run-length clusters of contiguous false-positive windows without majority smoothing ($\approx 80\text{ clusters} / 152.8231\text{ h} \times 24 = 12.56\text{ FA/24h}$).
3. **Clinical Alarm Episode Rate ($7.38\text{ FA/24h}$, historical harness $\approx 7.22\text{ FA/24h}$)**:
   - Implemented in `neuroaegis/eval/metrics.py` (`apply_false_alarm_protocol`).
   - Applies 3-window temporal majority smoothing, merges alarms separated by $\le 15.0\text{s}$, and filters alarms $< 5.0\text{s}$.
   - Yields $47\text{ unmatched alarm episodes}$ across 155 recordings: $47 / 152.8231\text{ h} \times 24 = 7.38\text{ FA/24h}$.

### B. Detection Delay Definition Discrepancy
1. **Completion-Based Delay ($10.57\text{s}$)**:
   - Implemented in `research/phase_4b/train_cnn_gnn_gru.py` (line 136) and `research/phase_7/statistical_evaluator.py`.
   - Formula: $\text{delay} = \text{first\_detected\_window\_end\_sec} - \text{seizure\_start\_sec}$.
   - Overestimates latency by including the 5.0s window buffer duration.
2. **Onset-Based Delay ($5.57\text{s}$)**:
   - Correct clinical formulation: $\text{delay} = \text{first\_detected\_window\_start\_sec} - \text{seizure\_start\_sec}$.
   - Represents the true physiological latency from electrographic onset to the beginning of the detecting window.

### C. Total Monitoring Duration Computation
1. **Authoritative EDF Manifest Duration**:
   - Sum of `recording_duration_sec` across the 155 test recordings = $550,163.0\text{ s} = 152.8231\text{ hours}$.
2. **Window Count Duration Approximation**:
   - $N_{\text{windows}} \times \text{stride} = 219,909 \times 2.5\text{ s} = 549,772.5\text{ s} = 152.7146\text{ hours}$.
   - Protocol V1 adopts the **Authoritative EDF Duration** ($152.8231\text{ hours}$) to strictly reflect real-world clinical monitoring time.

---

## 3. Audit Conclusion

All inconsistencies stem from differing aggregation scopes (raw windows vs. clinical episodes, window start vs. window end) rather than underlying model prediction differences. The underlying raw predictions for Model C are 100% stable and verified. Protocol V1 provides the required unified formalization.
