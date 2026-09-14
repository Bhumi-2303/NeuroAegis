# NEUROAEGIS PHASE 9B: TEMPORAL POST-PROCESSING REPORT

## 1. Objective
Optimize temporal alarm post-processing for the FROZEN CNN + Spatial GNN + Causal GRU Model C using strictly the verified CHB-MIT validation predictions to reduce false alarms while maintaining optimal sensitivity.

## 2. Frozen Model C Definition
The neural model components (CNN backbone, Spatial GNN, Causal GRU) and threshold ($T=0.50$) were strictly frozen. No retraining or weight modifications occurred.

## 3. Validation Cohort
chb06, chb07, chb08, chb10.

## 4. Prediction Provenance
Verified against Phase 4B `val_predictions.npz`. 

## 5. Raw Probability Definition
$P \ge 0.50$ implies a positive raw window.

## 6. Temporal Post-processing Methodology
A deterministic post-processor implementing N-of-M persistence, segment merging, minimum duration filtering, and refractory periods.

## 7. Candidate Search Space
- **N-of-M:** 1-of-1 to 5-of-7
- **Merge Interval:** 0s to 60s
- **Minimum Duration:** 0s to 15s
- **Refractory Period:** 0s to 120s

## 8. Event Matching Protocol
Maintained existing authoritative protocol: an alarm is true-positive if its interval intersects a ground-truth seizure interval.

## 9. False Alarm Definition
A temporal alarm episode that does not intersect any true seizure. Counted relative to precise chronological monitoring duration.

## 10. Validation Results
- Baseline FA/day: 93.99
- Optimized FA/day: 16.84
- False-alarm reduction: 82.08%

## 11. Candidate Ranking
Configurations with minimum FA/day subject to preserving baseline sensitivity (since baseline was $<90\%$, it preserved the baseline max of $60\%$).

## 12. Selected Configuration
- N-of-M: 3-of-4
- Minimum Duration: 15.0 sec
- Merge Interval: 60.0 sec
- Refractory: 0.0 sec

## 13. Baseline vs Optimized Comparison
Achieved an 82% reduction in FA burden with zero decay in baseline sensitivity.

## 14. Per-patient results
Generated in auxiliary files.

## 15. Detection delay analysis
Median delay increased to 95.0s due to the 15s minimum duration constraint.

## 16. Limitations
Performance on validation data may not fully generalize. Final metrics must be verified on test data.

## 17. Leakage Audit
[PASS] Zero test patients or labels accessed.

## 18. Reproducibility Information
All configurations saved in `phase_9b_candidates_raw.csv`.

## 19. Final Frozen Parameters
Frozen temporal configuration saved to `phase_9b_selected_postprocessing.json`.
