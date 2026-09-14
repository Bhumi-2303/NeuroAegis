# NeuroAegis Datasets

This directory manages datasets for the NeuroAegis project.

**IMPORTANT: NEVER commit raw EEG data (CHB-MIT, Siena, Bonn, etc.) to this repository.**

## Directory Structure
- `raw/`: Immutable, original downloaded dataset files. (Ignored by Git)
- `interim/`: Intermediate representations, parsed signals, etc. (Ignored by Git)
- `processed/`: Final datasets ready for model training. (Ignored by Git)
- `manifests/`: Lightweight CSV metadata, patient splits, and indexes. (Tracked by Git)
- `cache/`: Caches for features or embeddings. (Ignored by Git)

## CHB-MIT Dataset
- **Source**: PhysioNet (https://physionet.org/content/chbmit/1.0.0/)
- **Download**: `wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/`
- **Structure**: Place all patient folders (`chb01`, `chb02`, etc.) directly into `data/raw/chbmit_edf/`.
