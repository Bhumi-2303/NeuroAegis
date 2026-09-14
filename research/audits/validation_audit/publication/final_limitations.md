# NeuroAegis: Authoritative Research Limitations & Boundaries

To preserve scientific rigor, all publications and reports derived from this research must acknowledge the following seven structural boundaries:

1. **CHB-MIT Test Cohort Sample Size ($N=4$)**:
   The held-out patient cohort comprises 4 subjects (`chb01`, `chb02`, `chb03`, `chb05`). While total monitoring duration (152.82 hours) and window volume (219,909) are substantial, patient-level non-parametric inferential statistics (Wilcoxon signed-rank) are mathematically underpowered for $p < 0.05$ (minimum achievable two-tailed $p = 0.125$). Large effect sizes (Cohen's $d_z > 2.0$) substantiate separation, but validation across larger patient cohorts is required.

2. **Single Missed Seizure Event (`chb01_15`)**:
   Model C successfully detected 21 of 22 test seizure events (95.45% sensitivity). Exactly one clinical event (`chb01_15`, duration 40.0s) was missed. Forensic analysis indicates focal epileptiform onset was localized in posterior occipital leads that form an isolated subgraph component in the $	heta=0.30$ topology.

3. **Siena External Benchmark Scale (2 Patients, 4 Events)**:
   The external hospital validation was conducted on an available benchmark subset of 2 patients (`PN00`, `PN12`) spanning 4 recordings, 4 seizure events, and 2.46 monitoring hours. Authors must never claim "universal generalization across all Siena patients" or "full external validation". Exactly 12 Siena patients were unavailable during experimentation.

4. **Single-Subject Domain Adaptation Calibration**:
   The post-hoc temperature and threshold calibration was fitted on a single subject (`PN00`) and evaluated on held-out subject (`PN12`). While test F1 improved from 0.3043 to 0.3750 with zero test leakage, window sensitivity was conservative (23.08%). Multicenter unsupervised domain adaptation on larger calibration cohorts remains future work.

5. **Absence of Neurologist Ground-Truth for Explainability**:
   Explainable AI attributions (Integrated Gradients, Saliency) were validated via rigorous computational perturbation experiments (monotonic insertion/deletion curves). However, formal concordance studies against independent board-certified clinical epileptologist channel annotations were not performed. Attributions reflect internal model mechanics rather than verified pathophysiological causality.

6. **False Alarm Metric Dependency on Alerting Definition**:
   The reported false alarm rate (62.66/24h) reflects 5.0-second sliding windows with single-window threshold crossings. Clinical alerting systems in practice employ persistence filtering, refractory lockout intervals, or multi-window voting, which would significantly alter nominal false alarm numbers.

7. **Retrospective Benchmark vs Prospective Clinical Deployment**:
   High retrospective benchmark performance on archival recordings does NOT establish prospective clinical deployment readiness. Real-time bedside deployment requires handling real-world artifacts (electrode displacement, muscle tremor, line noise bursts, patient movement), hardware integration, and prospective clinical trial clearance.
