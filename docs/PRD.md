# NeuroAegis — Product Requirements Document

| Field        | Value                                                                 |
| ------------ | --------------------------------------------------------------------- |
| **Product**  | NeuroAegis                                                            |
| **Type**     | XAI Framework — Epileptic Seizure Detection Dashboard                 |
| **Status**   | Draft                                                                 |
| **Date**     | 2026-07-18                                                            |

---

## 1 · Vision

NeuroAegis is an Explainable AI (XAI) framework for automated epileptic seizure detection from EEG. The frontend is a **luxury, futuristic, glassmorphic AI command-center dashboard** — a single-pane operating system that surfaces a classification and the reasoning behind it, fast enough to trust under time pressure.

---

## 2 · Problem Statement

Reviewing raw EEG manually to catch seizures is **slow, expertise-gated, and does not scale**. A neurologist must visually parse high-density multi-channel recordings, identify ictal patterns, and make a determination — a process that is both cognitively demanding and time-constrained.

Clinicians and researchers need a **command-center view** that:

1. Visualises EEG signals and their frequency-domain decomposition.
2. Surfaces per-model seizure / non-seizure classification with confidence scores.
3. Presents the *reasoning* behind each prediction via SHAP-based explainability.
4. Exposes evaluation metrics as a trust layer for clinical consideration.

---

## 3 · Product Summary

A **single-pane AI operating system** that shows:

- Live-feeling EEG waveform visualization
- Frequency-band analysis (Gamma / Beta / Alpha / Theta / Delta)
- Per-model seizure classification (Random Forest, XGBoost, LightGBM)
- SHAP-based explainability as a first-class feature
- Evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix)

All panels run on **mock data today**; the architecture is designed so a **real ML backend can drop in tomorrow** by editing one folder (`core/services/model/`).

---

## 4 · Primary Users / Personas

| Persona                              | Core Need                                                                                                                                   |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Neuroscience Researchers**          | Evaluate model performance across multiple classifiers, compare SHAP explanations, and validate that the system's reasoning aligns with domain knowledge. |
| **Clinical AI Reviewers**             | A trust layer (metrics, explainability) to assess whether a model's seizure detection is reliable enough for clinical consideration.         |
| **Demo / Thesis-Committee Stakeholders** | A visually impressive, self-explanatory dashboard that communicates the project's technical depth and rigor at a glance.                   |

---

## 5 · Goals

| #  | Goal                                                                                                          |
| -- | ------------------------------------------------------------------------------------------------------------- |
| G1 | Provide a single-pane command-center view of EEG-based seizure detection.                                     |
| G2 | Surface per-model classification results (Random Forest, XGBoost, LightGBM) with confidence scores.           |
| G3 | Present SHAP-based explainability as a first-class feature (horizontal bar / waterfall of top contributing features). |
| G4 | Display evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix) as the trust layer.   |
| G5 | Visualise EEG waveforms and frequency-band analysis (Gamma / Beta / Alpha / Theta / Delta).                   |
| G6 | Architecture so cleanly separated that a real ML backend can drop in by editing one folder.                    |

---

## 6 · Non-Goals / Explicitly Out of Scope

> [!CAUTION]
> The items below are **intentionally excluded** from the current scope. Any feature or artefact that falls into these categories must not be built.

| #   | Non-Goal                                                                                      |
| --- | --------------------------------------------------------------------------------------------- |
| NG1 | No model training, inference, or preprocessing logic of any kind.                             |
| NG2 | No real patient data.                                                                         |
| NG3 | No real authentication — stub only.                                                           |
| NG4 | No regulatory compliance implementation (HIPAA, etc.) — mention as roadmap only.              |
| NG5 | No real backend or persistence beyond in-memory / mock.                                       |
| NG6 | No invented metrics, panels, or domain language beyond the specified contracts.                |

---

## 7 · Feature List

### 7.1 Dashboard

**Hero view of the command centre.**

- 3D brain visualization as the centrepiece.
- Bottom metric cards:
  - Signal Strength
  - Active Channel Count
  - Current Confidence
  - Session Duration
  - Alert Count
- Neural signal timeline.

### 7.2 Brain Analysis

- Holographic 3D brain centrepiece.
- Animated synapse pulses and glowing neurons.
- Transparent layered shells for depth effect.

### 7.3 EEG Monitor

- Real-time EEG waveform visualization.
- Multi-channel support.

### 7.4 Frequency Analysis

- Frequency bands displayed as thin glowing line charts:
  - **Gamma**
  - **Beta**
  - **Alpha**
  - **Theta**
  - **Delta**

### 7.5 Seizure Prediction

- Per-model seizure / non_seizure classification.
- Confidence gauge per prediction.
- Model selector: Random Forest · XGBoost · LightGBM.

### 7.6 Explainability (SHAP)

- Dedicated panel for SHAP-based explainability.
- Horizontal bar / waterfall chart of top contributing features.
- Features ranked by |SHAP value|.

### 7.7 Reports / Analytics

- Evaluation metrics page:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC curve
  - Confusion matrix heatmap

### 7.8 Patients

- Patient / session management (stub).

### 7.9 Settings

- Application configuration.

### 7.10 Navigation

- **Top nav**: Floating glass bar.
- **Sidebar**: Collapsed icon sidebar.

---

## 8 · Domain Context

> [!IMPORTANT]
> All UI terminology, metric labels, and panel names must originate from this domain context. No terminology outside the specified domain may appear in the interface.

### 8.1 Task

Binary classification of EEG segments → **seizure** / **non_seizure**.

### 8.2 Pipeline (Represented Visually)

```
Raw EEG Acquisition
  → Preprocessing
    → Feature Extraction (temporal + frequency-domain)
      → Classification
        → Explainability (SHAP)
```

### 8.3 Candidate Models

| Model           | Type               |
| --------------- | ------------------ |
| Random Forest   | Ensemble (bagging) |
| XGBoost         | Ensemble (boosting)|
| LightGBM        | Ensemble (boosting)|

All three models are **plural and comparable** — the UI must support side-by-side evaluation.

### 8.4 Trust Layer (Evaluation Metrics)

| Metric           | Format                    |
| ---------------- | ------------------------- |
| Accuracy         | Scalar (0–1 or %)         |
| Precision        | Scalar (0–1 or %)         |
| Recall           | Scalar (0–1 or %)         |
| F1-Score         | Scalar (0–1 or %)         |
| ROC-AUC          | Curve + scalar summary    |
| Confusion Matrix | 2 × 2 heatmap             |

### 8.5 Explainability

SHAP (SHapley Additive exPlanations) values identifying **which EEG features drove a prediction**. Displayed as horizontal bar or waterfall charts ranked by absolute SHAP value.

### 8.6 Frequency Bands

| Band   | Role in Feature Engineering                  |
| ------ | -------------------------------------------- |
| Gamma  | High-frequency neural activity               |
| Beta   | Active / alert state oscillations            |
| Alpha  | Relaxed / idle state oscillations            |
| Theta  | Drowsiness / light sleep oscillations        |
| Delta  | Deep sleep / pathological slow-wave activity |

These bands are **real feature-engineering inputs** to the classification pipeline, not decorative.

---

## 9 · Success Metrics

| #  | Criterion                                                                              |
| -- | -------------------------------------------------------------------------------------- |
| S1 | App runs end-to-end on mock data with **zero console errors**.                         |
| S2 | Every model-facing panel shows correct **loading / empty / error / success** states.   |
| S3 | All model-related code isolated under `core/services/model/`.                          |
| S4 | Every model touchpoint carries a `// TODO: Integrate Trained Model` marker.            |
| S5 | Feature folders **never import from each other** directly.                             |
| S6 | Visual output matches the design spec — **no generic dashboard aesthetic**.            |
| S7 | No terminology outside the specified domain appears in the UI.                         |

---

## 10 · Architecture Boundary

> [!NOTE]
> This section captures the architectural constraint that enables future model integration without frontend changes.

```
src/
├── core/
│   └── services/
│       └── model/          ← All model-related code lives here
│                              Every public function carries:
│                              // TODO: Integrate Trained Model
├── features/
│   ├── dashboard/
│   ├── brain-analysis/
│   ├── eeg-monitor/
│   ├── frequency-analysis/
│   ├── seizure-prediction/
│   ├── explainability/
│   ├── reports/
│   ├── patients/
│   └── settings/
└── ...
```

**Rules:**

1. Feature folders never import from each other directly.
2. All model data flows through `core/services/model/`.
3. Swapping mock data for real inference requires editing **only** `core/services/model/`.

---

## 11 · Design Language

| Attribute         | Specification                                                              |
| ----------------- | -------------------------------------------------------------------------- |
| **Aesthetic**      | Luxury, futuristic, glassmorphic AI command-centre                        |
| **Top Navigation** | Floating glass bar                                                        |
| **Sidebar**        | Collapsed icon sidebar                                                    |
| **Cards / Panels** | Glassmorphic (frosted-glass, backdrop-blur, translucent borders)          |
| **Charts**         | Thin glowing line charts (frequency bands), horizontal bar / waterfall (SHAP) |
| **3D Elements**    | Holographic brain with animated synapse pulses, glowing neurons, transparent layered shells |
| **State Handling** | Every model-facing panel must render loading, empty, error, and success states |

> [!WARNING]
> The visual output must match the design spec. A generic dashboard aesthetic is a **failure criterion** (see S6).

---

## 12 · Roadmap (Design Language Only — Do Not Build)

> [!NOTE]
> These items define future direction only. They must **not** be implemented in the current scope. They may be referenced in UI copy or documentation as forward-looking statements.

| Phase | Item                                        |
| ----- | ------------------------------------------- |
| R1    | Real-time EEG monitoring with live data streams |
| R2    | Wearable EEG device integration             |
| R3    | Remote healthcare deployment                |
| R4    | HIPAA / regulatory compliance               |
| R5    | Multi-patient concurrent monitoring         |

---

## 13 · Constraints & Assumptions

| Type           | Detail                                                                                          |
| -------------- | ----------------------------------------------------------------------------------------------- |
| **Data**       | All data is mock / synthetic. No real patient data is used or stored.                            |
| **Auth**       | Authentication is a stub only — no real identity provider.                                      |
| **Persistence**| In-memory / mock only — no database or file-system persistence.                                 |
| **Compliance** | No regulatory compliance is implemented. HIPAA is a roadmap item only.                          |
| **Models**     | No training, inference, or preprocessing logic exists in the frontend. Model outputs are mocked. |
| **Terminology**| All UI labels and domain terms must originate from the specified contracts — nothing invented.   |

---

*End of Document*
