# Phase 9: Pre-Experiment Pipeline Audit
## Existing Alert Pipeline (Model C)
- Probability threshold: 0.50
- Window duration: 5.0 seconds
- Stride: 2.5 seconds
- Minimum alert duration: None
- Consecutive-window requirement: None
- Alarm merge interval: None
- Refractory period: None
- Seizure matching tolerance: Single window overlap (>=50%)
- False-alarm episode counting: Independent false positive windows / total hours * 24.
- Existing post-processing: None.
