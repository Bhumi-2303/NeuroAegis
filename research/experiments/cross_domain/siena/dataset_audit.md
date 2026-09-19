# NeuroAegis Experiment 6 — Siena Dataset Audit & Availability Report

## 1. Full Siena Scalp EEG Cohort Overview (PhysioNet)
- **Institution**: Unit of Neurology and Neurophysiology, University of Siena, Italy
- **Subjects**: 14 epileptic patients (9 male, 5 female, ages 25–71)
- **Total EDF Recordings**: 42 long-term continuous EEG sessions
- **Total Annotated Seizure Events**: 47 clinician-verified clinical seizures
- **Total Monitoring Duration**: ~128.0 continuous hours
- **Native Sampling Frequency**: 512.0 Hz (16-bit A/D conversion)
- **Electrode System**: International 10-20 system (unipolar referential montage, 29 EEG channels + ECG/EOG)
- **Power-Line Grid**: 50.0 Hz (European standard electrical grid)

## 2. Local Dataset Shard & Availability Separation

```
FULL SIENA COHORT (14 Patients, 42 EDFs, 47 Seizures, ~128h)
      │
      ▼
AVAILABLE LOCAL RECORDINGS (2 Patients, 6 EDFs, 4 Seizures, 2.67h)
      ├── Complete Recording Sessions: 4 EDFs (PN00-1, PN00-4, PN00-5, PN12-3) -> 2.46h, 4 Seizures, 3,538 Windows
      └── Truncated Pre-Ictal Shards:  2 EDFs (PN00-2, PN00-3) -> 0.21h, 0 Seizures in 380s shard, 302 Windows
      │
      ▼
ACTUAL ZERO-SHOT EVALUATION COHORT: 2 Patients, 6 EDFs, 3,840 Windows (2.67h, 4 Active Seizures)
```

### Explicit Reasons for Excluded/Unavailable Recordings:
1. **Local Repository Constraints**: Only patients `PN00` and `PN12` were packaged in the local `data/siena_edf/` repository storage directory.
2. **Truncation in Shards PN00-2 and PN00-3**: `PN00-2.edf` and `PN00-3.edf` are truncated at 380 seconds (194,560 samples), whereas their annotated clinical seizures occurred at $t=1220\text{s}$ and $t=765\text{s}$. These 380s segments were evaluated as non-seizure background EEG.
3. **Scientific Reporting Label**: This experiment is formally titled and reported as **'Siena Zero-Shot Evaluation on Available Local Subset'**, never as 'full Siena evaluation'.