# NeuroAegis Experiment 6 — Channel Harmonization & Reconstruction Report

## 1. Topological Derivation Methodology
The CHB-MIT model expects the canonical 23-channel **Double-Banana Bipolar Montage**.
Siena Scalp EEG is recorded as a **Unipolar Referential Montage** against a common reference electrode ($V_{\text{ref}}$).
By Kirchhoff's voltage law, any bipolar differential derivation between electrode $A$ and electrode $B$ is reconstructed exactly by subtraction:

$$V_{A-B} = (V_A - V_{\text{ref}}) - (V_B - V_{\text{ref}}) = V_A - V_B$$

## 2. 10-20 Nomenclature Equivalence
The International 10-20 standard updated temporal electrode nomenclature in modern 10-10 extensions:
- $T_3 \equiv T_7$ (Left Mid-Temporal)
- $T_4 \equiv T_8$ (Right Mid-Temporal)
- $T_5 \equiv P_7$ (Left Posterior-Temporal)
- $T_6 \equiv P_8$ (Right Posterior-Temporal)
- $F_9 \equiv FT_9$ (Left Anterior-Inferior Temporal)
- $F_{10} \equiv FT_{10}$ (Right Anterior-Inferior Temporal)

## 3. Complete 23-Channel Mapping Matrix

| Index | Canonical CHB-MIT Derivation | Siena Differential Derivation | Siena Anode | Siena Cathode | Equivalence Rule |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 0 | **FP1-F7** | `EEG Fp1 - EEG F7` | FP1 | F7 | Exact 1:1 anatomical match |
| 1 | **F7-T7** | `EEG F7 - EEG T3` | F7 | T3 | 10-20 Standard Equivalent (T3 == T7) |
| 2 | **T7-P7** | `EEG T3 - EEG T5` | T3 | T5 | 10-20 Standard Equivalent (T3 == T7, T5 == P7) |
| 3 | **P7-O1** | `EEG T5 - EEG O1` | T5 | O1 | 10-20 Standard Equivalent (T5 == P7, O1 == O1) |
| 4 | **FP1-F3** | `EEG Fp1 - EEG F3` | FP1 | F3 | Exact 1:1 anatomical match |
| 5 | **F3-C3** | `EEG F3 - EEG C3` | F3 | C3 | Exact 1:1 anatomical match |
| 6 | **C3-P3** | `EEG C3 - EEG P3` | C3 | P3 | Exact 1:1 anatomical match |
| 7 | **P3-O1** | `EEG P3 - EEG O1` | P3 | O1 | Exact 1:1 anatomical match |
| 8 | **FP2-F4** | `EEG Fp2 - EEG F4` | FP2 | F4 | Exact 1:1 anatomical match |
| 9 | **F4-C4** | `EEG F4 - EEG C4` | F4 | C4 | Exact 1:1 anatomical match |
| 10 | **C4-P4** | `EEG C4 - EEG P4` | C4 | P4 | Exact 1:1 anatomical match |
| 11 | **P4-O2** | `EEG P4 - EEG O2` | P4 | O2 | Exact 1:1 anatomical match |
| 12 | **FP2-F8** | `EEG Fp2 - EEG F8` | FP2 | F8 | Exact 1:1 anatomical match |
| 13 | **F8-T8** | `EEG F8 - EEG T4` | F8 | T4 | 10-20 Standard Equivalent (T4 == T8) |
| 14 | **T8-P8** | `EEG T4 - EEG T6` | T4 | T6 | 10-20 Standard Equivalent (T4 == T8, T6 == P8) |
| 15 | **P8-O2** | `EEG T6 - EEG O2` | T6 | O2 | 10-20 Standard Equivalent (T6 == P8, O2 == O2) |
| 16 | **FZ-CZ** | `EEG Fz - EEG Cz` | FZ | CZ | Exact 1:1 anatomical match |
| 17 | **CZ-PZ** | `EEG Cz - EEG Pz` | CZ | PZ | Exact 1:1 anatomical match |
| 18 | **P7-T7** | `EEG T5 - EEG T3` | T5 | T3 | 10-20 Standard Equivalent (T5 == P7, T3 == T7) |
| 19 | **T7-FT9** | `EEG T3 - EEG F9` | T3 | F9 | 10-20 Standard Equivalent (T3 == T7, F9 == FT9) |
| 20 | **FT9-FT10** | `EEG F9 - EEG F10` | F9 | F10 | 10-20 Standard Equivalent (F9 == FT9, F10 == FT10) |
| 21 | **FT10-T8** | `EEG F10 - EEG T4` | F10 | T4 | 10-20 Standard Equivalent (F10 == FT10, T4 == T8) |
| 22 | **T8-P8** | `EEG T4 - EEG T6` | T4 | T6 | 10-20 Standard Equivalent (T4 == T8, T6 == P8) |