# Methods and Experimental Design Scaffold

## What We Do Not Claim
Before presenting our methodology, we explicitly define the boundaries of our claims:
1. **No "Headline Accuracy":** We do not report raw classification accuracy as a primary headline metric. EEG seizure datasets are highly imbalanced; accuracy is fundamentally misleading and is provided only as a reference.
2. **Distinct Evaluation Boundaries:** We do not equate results on the Bonn dataset with results on the CHB-MIT/Siena datasets. Bonn is a curated, secondary robustness experiment (e.g., differentiating healthy vs. interictal vs. ictal snippets) and is explicitly not a continuous clinical benchmark.
3. **Attention is Not Explanation:** We do not claim that neural attention weights are inherently faithful or interpretable explanations of the model's physical decision-making process unless explicitly validated by our clinician-agreement study.

---

## 1. Hypotheses

**H1: Architectural Spatial-Temporal Superiority**
*   **Statement:** The proposed CNN → GNN → GRU → Attention architecture will achieve a statistically significant improvement in reducing False Alarms per 24 hours while maintaining event sensitivity, compared to spatial-only (CNN) or temporal-only baselines.
*   **Deciding Metric:** Event Sensitivity and False Alarms / 24h (FA/24h) on the Leave-One-Subject-Out (LOSO) cross-validation.
*   **Status:** [Placeholder: Supported / Not Supported / Partially Supported]

**H2: External Generalization**
*   **Statement:** The model will demonstrate strong zero-shot generalization capabilities on the unseen, external Siena dataset without requiring architecture modification.
*   **Deciding Metric:** Event Sensitivity, AUPRC, and F1 Score on the Siena dataset.
*   **Status:** [Placeholder: Supported / Not Supported / Partially Supported]

**H3: Domain Adaptation Recovery**
*   **Statement:** Target-domain calibration and feature harmonization applied to a strict subset of Siena calibration data will significantly recover performance drops in external transfer, outperforming direct zero-shot transfer.
*   **Deciding Metric:** Delta in F1 Score and FA/24h between zero-shot (Variant A) and Harmonization+Calibration (Variant D) on the Siena evaluation subset.
*   **Status:** [Placeholder: Supported / Not Supported / Partially Supported]

**H4: Explainability Alignment (XAI)**
*   **Statement:** Architecture-agnostic Integrated Gradients (IG) and extracted Attention weights will demonstrate strong alignment (overlap and correlation) with masked regions of clinical importance annotated by human epileptologists.
*   **Deciding Metric:** Jaccard similarity and Spearman rank correlation between XAI outputs and structured clinician annotations across Time, Channel, and Frequency.
*   **Status:** [Placeholder: Supported / Not Supported / Partially Supported]

**H5: Signal Robustness on Secondary Tasks**
*   **Statement:** The core feature extraction backbone is robust enough to separate distinct clinical conditions (Healthy vs. Interictal vs. Ictal) in an isolated, curated experimental setting.
*   **Deciding Metric:** AUPRC and Sensitivity on Bonn Experiments A, B, and C.
*   **Status:** [Placeholder: Supported / Not Supported / Partially Supported]

---

## 2. Datasets

The study is strictly partitioned into three separate datasets to prevent data leakage and ensure rigorous validation of clinical generalizability:

### 2.1 CHB-MIT (Primary Training & Internal Test)
*   **Role:** Model training, hyperparameter tuning, and internal validation.
*   **Evaluation:** Strict Leave-One-Subject-Out (LOSO) cross-validation. No patient overlaps between train and test folds.
*   **Characteristics:** Continuous, multi-channel scalp EEG with annotated seizure intervals.

### 2.2 Siena (Primary External Test)
*   **Role:** Assessing external generalization and domain adaptation algorithms.
*   **Evaluation:** External test set only. A predefined, isolated subset of patients acts as the "calibration" target, and the remaining predefined patients act as the final, immutable evaluation cohort.
*   **Characteristics:** Continuous, multi-channel scalp EEG, originating from different hardware and clinical protocols than CHB-MIT.

### 2.3 Bonn (Secondary Robustness Task)
*   **Role:** Secondary evaluation of learned representations across distinct signal classes (e.g., healthy surface EEG vs. intracranial ictal). Explicitly *not* a clinical continuous-recording benchmark.
*   **Evaluation:** Evaluated entirely separately from the CHB-MIT/Siena cohorts using curated snippets (Experiments A, B, and C). Results are never merged with primary event-level metrics.

---

## 3. Results (Placeholders)

### 3.1 Ablation Sequence (CHB-MIT LOSO)

*Table 3.1: Incremental impact of architectural components on internal validation (LOSO). FA/24h and Event Sensitivity are strictly prioritized.*

| Model Variant | Event Sensitivity | FA/24h | Sensitivity | AUPRC | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline: Linear / RF** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **1D CNN (Spatial Only)** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **CNN + GRU** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **CNN + GNN + GRU** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **Proposed (+ Attention)** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### 3.2 Domain Adaptation on External Cohort (Siena)

*Table 3.2: Effect of calibration and harmonization methodologies on unseen target domain adaptation.*

| Adaptation Variant | Event Sensitivity | FA/24h | Sensitivity | AUPRC | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A) Direct Transfer (No Adapt)** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **B) Normalization (Harmonization)** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **C) Calibration (Temperature)** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **D) Harmonization + Calibration** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### 3.3 Explainability & Clinician Agreement (XAI)

*Table 3.3: Pairwise agreement metrics across XAI methods and human clinician annotations.*

| Comparison Pair | Channel (Jaccard) | Channel (Spearman) | Time (Jaccard) | Freq (Spearman) |
| :--- | :--- | :--- | :--- | :--- |
| **IG vs Attention** | [TBD] | [TBD] | [TBD] | [TBD] |
| **IG vs SHAP** | [TBD] | [TBD] | [TBD] | [TBD] |
| **Attention vs SHAP** | [TBD] | [TBD] | [TBD] | [TBD] |
| **Clinician vs IG** | [TBD] | [TBD] | [TBD] | [TBD] |
| **Clinician vs Attention** | [TBD] | [TBD] | [TBD] | [TBD] |

### 3.4 Secondary Robustness Experiment (Bonn)

*Table 3.4: Discriminative capacity on isolated subsets. Note: these metrics are standard window-level statistics and do not reflect FA/24h clinical performance.*

| Bonn Experiment | Classes | Sensitivity | AUPRC | Accuracy (Ref) |
| :--- | :--- | :--- | :--- | :--- |
| **Exp A** | Healthy vs Seizure | [TBD] | [TBD] | [TBD] |
| **Exp B** | Interictal vs Ictal | [TBD] | [TBD] | [TBD] |
| **Exp C** | Healthy + Interictal vs Ictal | [TBD] | [TBD] | [TBD] |
