"""
NeuroAegis Phase 8: Publication Package Generator.
Generates:
- Tables 1 to 13 in Markdown (.md), CSV (.csv), and LaTeX (.tex) formats.
- Final publication summary documents:
    - final_results_summary.md
    - final_limitations.md
    - final_claims.md
    - reproducibility_instructions.md
- Copies and organizes metrics, supplementary files, and reproducibility assets.
"""

import os
import sys
import json
import shutil
import pandas as pd
import numpy as np

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
PUB_DIR = os.path.join(BASE_DIR, "research/audits/validation_audit/publication")
TABLES_DIR = os.path.join(PUB_DIR, "tables")
METRICS_DIR = os.path.join(PUB_DIR, "metrics")
SUPP_DIR = os.path.join(PUB_DIR, "supplementary")
REPRO_DIR = os.path.join(PUB_DIR, "reproducibility")
FINAL_RESULTS_DIR = os.path.join(BASE_DIR, "research/audits/validation_audit/final_results")

for d in [PUB_DIR, TABLES_DIR, METRICS_DIR, SUPP_DIR, REPRO_DIR]:
    os.makedirs(d, exist_ok=True)


def df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Converts a DataFrame into standard publication-quality booktabs LaTeX."""
    tex = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small", "\\begin{tabular}{" + "l" * len(df.columns) + "}", "\\toprule"]
    headers = " & ".join([str(c).replace("_", "\\_").replace("%", "\\%") for c in df.columns]) + " \\\\"
    tex.append(headers)
    tex.append("\\midrule")
    for _, row in df.iterrows():
        line = " & ".join([str(v).replace("_", "\\_").replace("%", "\\%") for v in row.values]) + " \\\\"
        tex.append(line)
    tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(tex)


def df_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v).replace("\n", " ") for v in row.values) + " |")
    return "\n".join(lines)


def save_table(table_num: int, table_name: str, df: pd.DataFrame, caption: str):
    slug = f"table_{table_num:02d}_{table_name}"
    # CSV
    csv_path = os.path.join(TABLES_DIR, f"{slug}.csv")
    df.to_csv(csv_path, index=False)
    # Markdown
    md_path = os.path.join(TABLES_DIR, f"{slug}.md")
    with open(md_path, "w") as f:
        f.write(f"### Table {table_num}: {caption}\n\n")
        f.write(df_to_markdown(df))
        f.write("\n")
    # LaTeX
    tex_path = os.path.join(TABLES_DIR, f"{slug}.tex")
    with open(tex_path, "w") as f:
        f.write(df_to_latex(df, caption, f"tab:{table_name}"))
        f.write("\n")
    print(f"  [SAVED] Table {table_num}: {slug} (CSV, MD, LaTeX)")


def generate_publication_tables():
    print("Generating Tables 1 to 13...")

    # Table 1: Dataset Characteristics
    t1_data = [
        {"Dataset": "CHB-MIT Scalp EEG", "Institution": "Boston Children's Hospital / MIT", "Subjects": 24, "Channels": "23 Bipolar", "Sampling Rate": "256 Hz", "Recordings": 983, "Seizures": 198, "Duration (h)": "969.8", "Montage": "Modified Bipolar 10-20"},
        {"Dataset": "Siena Scalp EEG (Full)", "Institution": "University of Siena Hospital, Italy", "Subjects": 14, "Channels": "29 Unipolar", "Sampling Rate": "512 Hz", "Recordings": 41, "Seizures": 47, "Duration (h)": "141.0", "Montage": "Harmonized 10-20 Bipolar"},
        {"Dataset": "Siena Benchmark Subset", "Institution": "University of Siena Hospital, Italy", "Subjects": 2, "Channels": "18 Bipolar Matched", "Sampling Rate": "256 Hz", "Recordings": 4, "Seizures": 4, "Duration (h)": "2.46", "Montage": "Harmonized 10-20 Bipolar"}
    ]
    save_table(1, "dataset_characteristics", pd.DataFrame(t1_data), "Summary of Pediatric and Adult Clinical Scalp EEG Databases")

    # Table 2: CHB-MIT Patient Partition
    t2_data = [
        {"Partition": "Training", "Subjects": 16, "Patient IDs": "chb04, chb09, chb11-chb24", "Recordings": 676, "Seizures": 137, "Windows": 981504, "Duration (h)": 681.60, "Class Imbalance": "327.4:1"},
        {"Partition": "Validation", "Subjects": 4, "Patient IDs": "chb06, chb07, chb08, chb10", "Recordings": 152, "Seizures": 39, "Windows": 213297, "Duration (h)": 148.40, "Class Imbalance": "335.2:1"},
        {"Partition": "Held-out Test", "Subjects": 4, "Patient IDs": "chb01, chb02, chb03, chb05", "Recordings": 155, "Seizures": 22, "Windows": 219909, "Duration (h)": 152.82, "Class Imbalance": "344.2:1"}
    ]
    save_table(2, "cohort_partition", pd.DataFrame(t2_data), "Patient-Stratified Disjoint Data Partitions for CHB-MIT")

    # Table 3: Model Architecture Comparison
    t3_data = [
        {"Model": "Model A (1D-CNN)", "Temporal Module": "Depthwise 1D-CNN (4 Conv Layers)", "Spatial Module": "Flatten / Dense Linear", "Sequential Context": "None (Single Window, 5.0s)", "Parameters": 173601, "Receptive Field": "5.0s"},
        {"Model": "Model B (CNN+GNN)", "Temporal Module": "Depthwise 1D-CNN (4 Conv Layers)", "Spatial Module": "2-Layer GCN (θ=0.30, 40 Edges)", "Sequential Context": "None (Single Window, 5.0s)", "Parameters": 52497, "Receptive Field": "5.0s"},
        {"Model": "Model C (CNN+GNN+GRU)", "Temporal Module": "Depthwise 1D-CNN (4 Conv Layers)", "Spatial Module": "2-Layer GCN (θ=0.30, 40 Edges)", "Sequential Context": "1-Layer Causal GRU (L=8, H=64)", "Parameters": 91858, "Receptive Field": "22.5s"}
    ]
    save_table(3, "model_architectures", pd.DataFrame(t3_data), "Architectural Dimensions and Hyperparameters of Evaluated Models")

    # Table 4: CHB-MIT Final Test Performance
    t4_df = pd.read_csv(os.path.join(FINAL_RESULTS_DIR, "final_model_comparison.csv"))
    cols_to_keep = ["model_id", "architecture_name", "parameters", "event_sensitivity_strict", "false_alarms_24h", "detection_delay_sec", "auroc", "auprc", "f1_score", "sensitivity", "specificity"]
    rename_map = {
        "model_id": "Model", "architecture_name": "Architecture", "parameters": "Params",
        "event_sensitivity_strict": "Event Sens", "false_alarms_24h": "FA/24h", "detection_delay_sec": "Delay (s)",
        "auroc": "AUROC", "auprc": "AUPRC", "f1_score": "F1", "sensitivity": "Win Sens", "specificity": "Win Spec"
    }
    t4_clean = t4_df[cols_to_keep].rename(columns=rename_map)
    save_table(4, "final_test_performance", t4_clean, "Authoritative Held-Out Test Evaluation on CHB-MIT Cohort (N=219,909 Windows, 152.82h)")

    # Table 5: Ablation Study
    t5_df = pd.read_csv(os.path.join(FINAL_RESULTS_DIR, "final_ablation.csv"))
    save_table(5, "ablation_study", t5_df, "Progressive Component Ablation and False Alarm Suppression")

    # Table 6: Patient-Level Performance
    t6_df = pd.read_csv(os.path.join(FINAL_RESULTS_DIR, "final_patient_results.csv"))
    save_table(6, "patient_level_performance", t6_df, "Patient-Specific Seizure Detection and False Alarm Breakdown")

    # Table 7: Event-Level Seizure Detection
    t7_df = pd.read_csv(os.path.join(FINAL_RESULTS_DIR, "final_event_results.csv"))
    save_table(7, "event_level_detection", t7_df, "Event-by-Event Detection Timing and Concordance Across Test Cohort (22 Seizures)")

    # Table 8: Cross-Domain Generalization
    t8_df = pd.read_csv(os.path.join(FINAL_RESULTS_DIR, "final_cross_domain.csv"))
    save_table(8, "cross_domain_generalization", t8_df, "Cross-Domain Transfer Benchmark: CHB-MIT Source vs Siena Target")

    # Table 9: Domain Adaptation
    t9_data = [
        {"Model State": "Zero-Shot PN12", "Temperature T*": "1.0000 (Default)", "Threshold τ*": "0.5000 (Default)", "Test Precision": "1.0000", "Test Window Sens": "0.1795", "Test F1": "0.3043", "Event Sens": "100% (1/1)", "FA/24h": "0.0"},
        {"Model State": "Adapted PN12", "Temperature T*": "0.3495 (Calibrated)", "Threshold τ*": "0.3800 (Calibrated)", "Test Precision": "1.0000", "Test Window Sens": "0.2308", "Test F1": "0.3750", "Event Sens": "100% (1/1)", "FA/24h": "0.0"}
    ]
    save_table(9, "domain_adaptation", pd.DataFrame(t9_data), "Post-Hoc Calibration Adaptation Results on Held-Out Siena Patient PN12")

    # Table 10: Explainability Results
    t10_df = pd.read_csv(os.path.join(FINAL_RESULTS_DIR, "final_xai_results.csv")).head(10)
    save_table(10, "explainability_attributions", t10_df, "Top-10 EEG Channels Identified by Integrated Gradients Attribution")

    # Table 11: Computational Complexity
    t11_data = [
        {"Architecture": "Model A (1D-CNN)", "Parameters": 173601, "Forward Latency (MPS)": "0.45 ms", "Forward Latency (CPU)": "1.12 ms", "FLOPs / Window": "34.2 MFLOPs", "Memory Footprint": "18.4 MB", "Real-Time Factor": "5,555x"},
        {"Architecture": "Model B (CNN+GNN)", "Parameters": 52497, "Forward Latency (MPS)": "0.82 ms", "Forward Latency (CPU)": "2.05 ms", "FLOPs / Window": "42.8 MFLOPs", "Memory Footprint": "22.6 MB", "Real-Time Factor": "3,048x"},
        {"Architecture": "Model C (CNN+GNN+GRU)", "Parameters": 91858, "Forward Latency (MPS)": "1.42 ms", "Forward Latency (CPU)": "3.68 ms", "FLOPs / Window": "51.4 MFLOPs", "Memory Footprint": "26.8 MB", "Real-Time Factor": "1,760x"}
    ]
    save_table(11, "computational_complexity", pd.DataFrame(t11_data), "Computational Runtime, Latency, and Memory Footprint Profile")

    # Table 12: Statistical Robustness
    t12_df = pd.read_csv(os.path.join(FINAL_RESULTS_DIR, "final_statistical_results.csv"))
    save_table(12, "statistical_robustness", t12_df, "Paired Statistical Significance, Effect Sizes, and Multiple Testing Adjustments")

    # Table 13: Limitations and Boundaries
    t13_data = [
        {"Limitation": "Small Holdout Cohort", "Boundary Description": "CHB-MIT test set has N=4 subjects", "Scientific Impact": "Underpowered for patient-level non-parametric inferential significance (min p=0.125)", "Future Research": "Validate on large multicenter cohorts (TUH, N > 1,000)"},
        {"Limitation": "Single Missed Seizure", "Boundary Description": "chb01_15 missed by Model C", "Scientific Impact": "Event sensitivity is 95.45% rather than 100%; focal onset in isolated occipital leads", "Future Research": "Dynamic temporal attention & multi-scale graph connectivity"},
        {"Limitation": "Siena Subset Benchmark", "Boundary Description": "Evaluated on 2 patients, 4 events, 2.46h", "Scientific Impact": "Does not prove universal external validity across all clinical populations", "Future Research": "Acquire and evaluate all 14 Siena patients"},
        {"Limitation": "Unvalidated XAI Biology", "Boundary Description": "Attributions verified computationally only", "Scientific Impact": "Salient channels reflect model mechanisms, not necessarily ground-truth ictal foci", "Future Research": "Clinician-in-the-loop validation with certified epileptologists"},
        {"Limitation": "Retrospective Evidence", "Boundary Description": "Evaluated on archived recordings", "Scientific Impact": "Cannot claim clinical deployment readiness without prospective trials", "Future Research": "Prospective bedside trials in ICU/EMU continuous monitoring"}
    ]
    save_table(13, "limitations_boundaries", pd.DataFrame(t13_data), "Methodological Boundaries, Limitations, and Necessary Clinical Qualifications")


def generate_publication_markdowns():
    print("Generating Publication Markdown Summaries...")

    # final_results_summary.md
    with open(os.path.join(PUB_DIR, "final_results_summary.md"), "w") as f:
        f.write("""# NeuroAegis: Final Authoritative Research Results Summary

## 1. Primary Scientific Findings
NeuroAegis establishes an end-to-end, patient-independent epileptic seizure detection framework combining channel-preserving 1D Convolutional Neural Networks, Spatial Graph Neural Networks, and Causal Gated Recurrent Units (**CNN + Spatial GNN + Causal GRU**).

Evaluated on the held-out CHB-MIT test cohort comprising **4 unseen patients, 155 continuous EDF recordings, 22 clinical seizure events, 152.82 continuous monitoring hours, and 219,909 evaluation windows (5.0s windows with 50% temporal overlap)**, the frozen model ($\tau = 0.50$, 91,858 parameters) achieved:

- **Event Sensitivity**: **95.45%** (21 of 22 seizure events detected; 1 missed: `chb01_15`).
- **Window Sensitivity**: **83.83%** (534 true positive windows).
- **Window Specificity**: **99.82%** (218,873 true negative windows).
- **Precision (PPV)**: **57.24%** (534 true positive windows out of 933 total alarms against natural 344:1 class imbalance).
- **F1 Score**: **0.68025**.
- **Balanced Accuracy**: **91.82%**.
- **AUROC**: **0.98970**.
- **AUPRC**: **0.80681**.
- **False Alarm Rate**: **62.66 alarms / 24 hours** (a **96.8% reduction** compared to the 1D-CNN baseline of 1,946.56/24h).
- **Mean Detection Delay**: **10.57 seconds** (Median: **9.0 seconds**, 85.7% detected within 15 seconds).
- **Inference Latency**: **1.42 ms / window** (1,760x faster than real-time on consumer Apple Silicon MPS).

---

## 2. Component Ablation Evidence
1. **Temporal 1D-CNN Baseline (Model A)**: 173,601 parameters. While detecting events under nominal overlap, under strict continuous alerting it exhibited catastrophic false alarm contamination (**1,946.56 FA/24h**, 12,395 false positives), yielding an AUPRC of 0.0415 and F1 of 0.0155.
2. **Spatial Graph Convolution Addition (Model B)**: 52,497 parameters ($\theta = 0.30$, 40 undirected edges). Spatial filtering dramatically suppressed false alarms by **89.7%** (down to 200.39 FA/24h); however, lacking temporal dynamics, event sensitivity collapsed to **27.27%** (6/22 events detected).
3. **Causal Temporal GRU Integration (Model C)**: 91,858 parameters ($L=8$, 22.5s context). Causal sequential recurrence restored event sensitivity to **95.45%** (21/22), suppressed false alarms further to **62.66 FA/24h** (-96.8% vs Model A), and surged AUPRC to **0.80681**.

---

## 3. External Cross-Domain Evaluation (Siena Scalp EEG)
On an external benchmark subset from the University of Siena Hospital (**2 patients, 4 recordings, 4 events, 3,538 windows, 2.46 hours**):
- **Zero-Shot Transfer**: Detected **4 of 4 events (100%)**, achieving window specificity of **99.85%**, AUROC of **0.9120**, and AUPRC of **0.7140** with 0 false alarms.
- **Post-Hoc Adaptation**: Calibrating temperature ($T^* = 0.3495$) and threshold ($\tau^* = 0.3800$) on PN00 improved held-out PN12 test F1 from **0.3043 to 0.3750** (+23.2% relative gain) with 100% precision.
""")

    # final_limitations.md
    with open(os.path.join(PUB_DIR, "final_limitations.md"), "w") as f:
        f.write("""# NeuroAegis: Authoritative Research Limitations & Boundaries

To preserve scientific rigor, all publications and reports derived from this research must acknowledge the following seven structural boundaries:

1. **CHB-MIT Test Cohort Sample Size ($N=4$)**:
   The held-out patient cohort comprises 4 subjects (`chb01`, `chb02`, `chb03`, `chb05`). While total monitoring duration (152.82 hours) and window volume (219,909) are substantial, patient-level non-parametric inferential statistics (Wilcoxon signed-rank) are mathematically underpowered for $p < 0.05$ (minimum achievable two-tailed $p = 0.125$). Large effect sizes (Cohen's $d_z > 2.0$) substantiate separation, but validation across larger patient cohorts is required.

2. **Single Missed Seizure Event (`chb01_15`)**:
   Model C successfully detected 21 of 22 test seizure events (95.45% sensitivity). Exactly one clinical event (`chb01_15`, duration 40.0s) was missed. Forensic analysis indicates focal epileptiform onset was localized in posterior occipital leads that form an isolated subgraph component in the $\theta=0.30$ topology.

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
""")

    # final_claims.md
    with open(os.path.join(PUB_DIR, "final_claims.md"), "w") as f:
        f.write("""# NeuroAegis: Scientific Claim Validation Matrix & Evidentiary Boundaries

| Claim ID | Scientific Claim | Empirical Evidence | Supporting Artifact | Validation Status | Mandatory Scientific Qualification |
|---|---|---|---|---|---|
| **CLM-01** | Spatial GNN suppresses false alarms | FA/24h reduced from 1,946.56 to 200.39 (-89.7%) | `false_alarm_results.csv` | **FULLY SUPPORTED** | Spatial filtering alone causes severe event sensitivity loss (27.27%) without temporal modeling. |
| **CLM-02** | GRU recurrence restores sensitivity | Event sensitivity surged from 27.27% to 95.45% (21/22) | `final_test_event_results.csv` | **FULLY SUPPORTED** | Evaluated on retrospective CHB-MIT test cohort; 1 event (`chb01_15`) remained undetected. |
| **CLM-03** | Composite architecture outperforms baselines | Model C AUPRC = 0.8068 vs 0.0415 (A) and 0.0049 (B). McNemar $p = 6 \\times 10^{-5}$ | `model_comparison.csv` | **FULLY SUPPORTED** | Superiority established under strict 10:1 dynamic negative sampling and patient holdouts. |
| **CLM-04** | Consistent patient generalization | Superior F1, AUROC, AUPRC on 4 of 4 test subjects (100% concordance) | `patient_level_results.csv` | **SUPPORTED WITH QUALIFICATION** | Patient count $N=4$ is statistically underpowered for paired tests ($p_{\\min} = 0.125$), though effect sizes ($d_z > 2.0$) are large. |
| **CLM-05** | Cross-domain external generalization | Detected 4/4 seizures (100%), AUROC 0.9120, AUPRC 0.7140 on Siena subset | `siena_zero_shot_summary.json` | **SUPPORTED WITH QUALIFICATION** | Must be designated 'preliminary benchmark subset' (2 patients, 2.46h). 12 patients were unavailable. |
| **CLM-06** | Post-hoc domain adaptation recovery | Test F1 on PN12 improved from 0.3043 to 0.3750 (+23.2%) with 100% precision | `siena_adapted_summary.json` | **SUPPORTED WITH QUALIFICATION** | Calibration performed on single patient (`PN00`); larger cohort adaptation required. |
| **CLM-07** | XAI attributions are computationally faithful | Monotonic degradation under feature deletion ($0.809 \\to 0.201$) | `insertion_deletion_results.csv` | **FULLY SUPPORTED** | Computationally faithful to model weights; clinician ground-truth validation was NOT performed. |
| **CLM-08** | Clinically deployable for real-time intervention | Inference latency = 1.42 ms / window (sub-second capability) | `real_time_inference_profile.png` | **NOT SUPPORTED** | Computational speed satisfies latency requirements, but retrospective benchmarks do NOT prove clinical deployment readiness. |
| **CLM-09** | Rapid event onset detection | Median detection delay = 9.0s; 85.7% of events detected within 15 seconds | `detection_delay_results.csv` | **FULLY SUPPORTED** | Detection delay measured relative to archival electrographic annotations. |
""")

    # reproducibility_instructions.md
    with open(os.path.join(PUB_DIR, "reproducibility_instructions.md"), "w") as f:
        f.write("""# NeuroAegis: Full Reproducibility Protocol & Instructions

## 1. Environment Setup
```bash
git clone https://github.com/Bhumi-2303/NeuroAegis.git
cd NeuroAegis
git checkout 7b9f06f4b01e27fe9dfe86e48374160f982a737a

python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas scipy scikit-learn openpyxl matplotlib
```

## 2. Decompress Master Window Index
```bash
gzip -dk data/manifests/chbmit_window_index.csv.gz
sha256sum data/manifests/chbmit_window_index.csv
# Expected SHA-256: f76dddb19fa2c24b3af24b989168b12465512b650c5e6d84bb0125a6d666302c
```

## 3. Cryptographic Checkpoint Verification
```bash
sha256sum artifacts/checkpoints/frozen_cnn_gnn_gru.pt
# Expected SHA-256: 2ec84897c39d31d68cfbf1f7e5c8708d5073fe19450576bf4f9132929c1832ca

sha256sum research/experiments/gnn/frozen_graph_adjacency.csv
# Expected SHA-256: 062c6aaffdc32ea1b9b5b97c82c224d40e0ab3b7e9b2afad5fd5b3e5f52db48e
```

## 4. Execute Full Audit & Verification Suite
```bash
python research/audits/validation_audit/run_phase8_audit.py
python research/audits/validation_audit/generate_phase8_figures.py
python research/audits/validation_audit/generate_phase8_workbook.py
python -m unittest research/audits/validation_audit/tests/test_final_research_audit.py -v
```
""")
    print("  [SAVED] All 4 publication markdown documents created.")


def copy_supporting_materials():
    print("Copying supporting metrics and supplementary artifacts...")
    # Copy JSONs to metrics
    for item in ["authoritative_final_metrics.json", "authoritative_final_metrics.csv", "reproducibility_manifest.json"]:
        src = os.path.join(FINAL_RESULTS_DIR, item)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(METRICS_DIR, item))
            shutil.copy(src, os.path.join(REPRO_DIR, item))

    # Copy supplementary CSVs
    for item in ["final_model_comparison.csv", "final_ablation.csv", "final_patient_results.csv", "final_event_results.csv", "final_cross_domain.csv", "final_statistical_results.csv", "final_claim_audit.csv", "cross_phase_consistency.csv"]:
        src = os.path.join(FINAL_RESULTS_DIR, item)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(SUPP_DIR, item))
    print("  [COPIED] Metrics, supplementary, and reproducibility directories populated.")


def main():
    print("============================================================")
    print("NEUROAEGIS PHASE 8: GENERATING PUBLICATION PACKAGE")
    print("============================================================")
    generate_publication_tables()
    generate_publication_markdowns()
    copy_supporting_materials()
    print("\n[SUCCESS] Complete Publication Package generated in research/audits/validation_audit/publication/.")


if __name__ == "__main__":
    main()
