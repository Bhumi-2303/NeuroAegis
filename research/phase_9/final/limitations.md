# Methodological Boundaries & Limitations

To ensure rigorous scientific integrity and prevent unwarranted clinical translation claims, this work explicitly outlines eight core methodological and empirical limitations:

1. **Small Patient-Level Test Cohort ($N=4$):**  
   While the CHB-MIT test cohort provides 152.82 continuous recording hours and 219,909 evaluation windows across 155 files, the number of distinct test subjects is limited to four pediatric patients (`chb01`, `chb02`, `chb03`, `chb05`). As a consequence, non-parametric patient-level significance tests (such as the Wilcoxon signed-rank test) have a minimum achievable p-value of $p = 0.125$ ($1/2^3$) and are underpowered to confirm asymptotic statistical significance across subjects, despite strong effect sizes ($d_z = 2.45$).

2. **Missed Seizure Analysis (`chb01_15`):**  
   The model failed to detect exactly one clinical seizure event: recording `chb01_15` (seizure 1, electrographic onset at 1,732 s, duration 40 s). Morphological inspection indicates this event was characterized by low-amplitude, highly focal beta/gamma rhythmic discharge without broad hemispheric synchronization, which did not sustain the sequential activation threshold within the 8-window causal GRU.

3. **External Benchmark Scope (Siena Benchmark Subset):**  
   Cross-domain external generalization was evaluated on a benchmark subset of the Siena Scalp EEG database consisting of 2 patients (`PN00`, `PN12`), 4 recordings, 4 seizures, and 2.46 hours. The remaining 12 patients in the repository were unavailable during benchmark acquisition. Consequently, these findings represent preliminary cross-domain feasibility rather than comprehensive multi-center external validation.

4. **Target Domain Adaptation Cohort Size:**  
   Post-hoc temperature scaling ($T^* = 0.3495$) and threshold adaptation ($\tau^* = 0.3800$) were derived using calibration data from a single subject (`PN00`, 3 seizures, 0.59h) and evaluated on a single held-out subject (`PN12`, 1 seizure, 1.87h). Validation across larger calibration cohorts is required to determine optimal population-level transfer parameters.

5. **Absence of Clinician-in-the-Loop Validation for XAI:**  
   Quantitative feature attribution was verified through mathematical axiomatic properties (Integrated Gradients) and empirical perturbation experiments (monotonic deletion curve drop from 0.8093 to 0.2011). However, formal clinical validation by certified board-certified epileptologists or EEG clinical technologists was NOT performed. Attributed regions represent model-saliency features rather than established neurophysiological biomarkers or guaranteed epileptogenic zones.

6. **Retrospective Benchmark Design:**  
   All evaluations were conducted on pre-recorded retrospective datasets. Retrospective evaluation cannot account for real-world bedside clinical conditions, such as patient movement, electrode dislodgement, telemetry dropouts, ongoing medication titration, or diverse metabolic encephalopathies.

7. **Dependency on Event-Postprocessing Definitions:**  
   Reported false alarm rates (62.66 FA/24h) and event sensitivities depend on the frozen operational post-processing parameters (5.0s window duration, 2.5s stride, contiguous positive window chaining threshold, and 50% seizure overlap criterion). Alternative clinical definitions of seizure occurrence or clustering would yield different numerical false alarm rates.

8. **Clinical Translation Boundary:**  
   NeuroAegis is a research prototype. It has NOT been approved as a medical device by any regulatory authority (e.g., FDA, CE-mark). The system must NOT be used for autonomous diagnostic decision-making, direct pharmacotherapy alteration, or unmonitored bedside seizure alarming without certified clinical supervision. Multi-center prospective clinical trials are necessary prior to any clinical translation.
