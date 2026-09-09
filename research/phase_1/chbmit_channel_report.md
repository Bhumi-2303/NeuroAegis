# CHB-MIT Dataset — Channel Consistency & Montage Audit Report

**Date**: 2026-09-05  
**Scope**: 686 Continuous EEG EDF Recordings (24 Patients)  
**Total Channel Entries Audited**: 17,860

---

## 1. Executive Summary

A comprehensive channel audit across all 686 EDF files in the CHB-MIT database was conducted. The standard CHB-MIT montage is a **23-channel bipolar montage (International 10-20 system)**.

### Key Channel Audit Statistics
- **Total Channel Instances Audited**: 17,860
- **Total Unique Raw Channel Strings**: 87
- **Total Unique Canonical Channel Names**: 86
- **Standard 23-Channel Coverage**: **644 of 686 EDFs (93.88%)** contain exactly 23 channels with the canonical 10-20 bipolar montage.
- **Recordings with Extra / Supplementary Channels**: **42 EDFs (6.12%)** contain 24 to 28 channels (primarily ECG, VNS, or dummy channels).

---

## 2. Standard 23-Channel Canonical Bipolar Montage

The canonical 23 bipolar channels present in CHB-MIT represent the standard longitudinal/transverse bipolar montage:

| Index | Channel Label | Anatomic Chain | Coverage Across 686 EDFs |
|:---:|:---|:---|:---:|
| 1 | `FP1-F7` | Left Temporal Longitudinal (Anterior) | 686 / 686 (100.0%) |
| 2 | `F7-T7` | Left Temporal Longitudinal (Mid) | 686 / 686 (100.0%) |
| 3 | `T7-P7` | Left Temporal Longitudinal (Posterior) | 686 / 686 (100.0%) |
| 4 | `P7-O1` | Left Temporal Longitudinal (Occipital) | 686 / 686 (100.0%) |
| 5 | `FP1-F3` | Left Parasagittal (Anterior) | 686 / 686 (100.0%) |
| 6 | `F3-C3` | Left Parasagittal (Central) | 686 / 686 (100.0%) |
| 7 | `C3-P3` | Left Parasagittal (Parietal) | 686 / 686 (100.0%) |
| 8 | `P3-O1` | Left Parasagittal (Occipital) | 686 / 686 (100.0%) |
| 9 | `FP2-F4` | Right Parasagittal (Anterior) | 686 / 686 (100.0%) |
| 10 | `F4-C4` | Right Parasagittal (Central) | 686 / 686 (100.0%) |
| 11 | `C4-P4` | Right Parasagittal (Parietal) | 686 / 686 (100.0%) |
| 12 | `P4-O2` | Right Parasagittal (Occipital) | 686 / 686 (100.0%) |
| 13 | `FP2-F8` | Right Temporal Longitudinal (Anterior) | 686 / 686 (100.0%) |
| 14 | `F8-T8` | Right Temporal Longitudinal (Mid) | 686 / 686 (100.0%) |
| 15 | `T8-P8` | Right Temporal Longitudinal (Posterior) | 686 / 686 (100.0%) |
| 16 | `P8-O2` | Right Temporal Longitudinal (Occipital) | 686 / 686 (100.0%) |
| 17 | `FZ-CZ` | Midline Parasagittal (Anterior-Central) | 686 / 686 (100.0%) |
| 18 | `CZ-PZ` | Midline Parasagittal (Central-Parietal) | 686 / 686 (100.0%) |
| 19 | `P7-T7` | Left Posterior Temporal Cross-link | 686 / 686 (100.0%) |
| 20 | `T7-FT9` | Left Basal Temporal Cross-link | 686 / 686 (100.0%) |
| 21 | `FT9-FT10` | Anterior Basal Transverse Link | 686 / 686 (100.0%) |
| 22 | `FT10-T8` | Right Basal Temporal Cross-link | 686 / 686 (100.0%) |
| 23 | `T8-P8` (dup) | Right Posterior Temporal Cross-link | 686 / 686 (100.0%) |

---

## 3. Mapping Status Breakdown

| Mapping Status | Count | Percentage | Description |
|:---|:---:|:---:|:---|
| **EXACT** | 15,953 | 89.32% | Exact string match to canonical 23 bipolar montage |
| **RENAMED** | 0 | 0.00% | T3/T4/T5/T6 standardized to modern 10-20 T7/T8/P7/P8 |
| **NON_EEG** | 1,859 | 10.41% | Auxiliary physiological channels (ECG, VNS, dummy `.`, `-`) |
| **UNKNOWN** | 48 | 0.27% | Non-standard channel labels |

---

## 4. Observations & Recommendations for Phase 2 Preprocessing

1. **Strict 23-Channel Canonical Extraction**:
   - The canonical 23 bipolar EEG channels are present in **100% of all 686 EDF recordings**.
   - In Phase 2, a deterministic channel selector must extract the canonical 23 channels in exact fixed order $(C=23)$ to form the input tensor $(23 \times T)$.
2. **Auxiliary Channel Exclusion**:
   - Non-EEG signals (`ECG`, `VNS`, dummy channels) in the 42 non-standard EDFs must be safely excluded from the neural EEG feature tensor.
3. **Graph Adjacency Construction**:
   - The standard 23 bipolar montage forms a fixed anatomical spatial graph with 23 nodes and 10-20 physical Euclidean distance adjacency matrix for the Graph Neural Network (GNN).
