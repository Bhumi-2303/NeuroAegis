<p align="center">
  <img src="https://img.shields.io/badge/NeuroAegis-Seizure%20Detection%20Platform-00E5FF?style=for-the-badge&labelColor=040814" alt="NeuroAegis" />
</p>

<h1 align="center">🧠 NeuroAegis</h1>

<p align="center">
  <strong>End-to-end Explainable AI platform for real-time epileptic seizure detection &amp; prediction from EEG signals</strong>
</p>

<p align="center">
  <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick_Start-▶-00E5FF?style=flat-square" alt="Quick Start" /></a>
  <a href="#-api-reference"><img src="https://img.shields.io/badge/API_Docs-📖-00E5FF?style=flat-square" alt="API Docs" /></a>
  <a href="#-model-performance"><img src="https://img.shields.io/badge/Model_Card-📊-00E5FF?style=flat-square" alt="Model Card" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10--3.12-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/React_18-61DAFB?style=flat-square&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white" />
  <img src="https://img.shields.io/badge/Three.js-000000?style=flat-square&logo=three.js&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-02569B?style=flat-square" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL_15-4169E1?style=flat-square&logo=postgresql&logoColor=white" />
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Datasets](#-datasets)
- [Signal Processing Pipeline](#-signal-processing-pipeline)
- [Privacy & Compliance](#-privacy--compliance)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔬 Overview

**NeuroAegis** is a full-stack machine learning platform that transforms raw EEG recordings into actionable clinical insights. It combines multi-domain signal processing, state-of-the-art tree-based ensemble models (LightGBM, XGBoost, Random Forest), and SHAP-powered Explainable AI to give clinicians, researchers, and neuroscience experts a transparent, real-time command center for seizure detection.

> **⚠️ Disclaimer:** NeuroAegis is an **experimental research tool** and is **not** an FDA/CE-approved medical device. It should not be used for clinical decision-making without professional medical oversight.

### What makes NeuroAegis different?

| Capability | Description |
|:---|:---|
| **Multi-Format Ingestion** | Upload `.edf`, `.csv`, or `.txt` EEG files — the system handles the rest |
| **Automatic Dataset Detection** | Rule-based engine identifies Bonn vs. CHB-MIT signal characteristics on the fly |
| **57-Feature Extraction** | Time, frequency, and wavelet domain features extracted in a single pass |
| **Explainable Predictions** | Every seizure score comes with a SHAP breakdown showing *exactly* which features drove the decision |
| **Zero-Shot Transfer** | Models trained on single-channel Bonn data generalize to multi-channel CHB-MIT scalp EEG (AUC 0.83) |
| **3D Brain Visualization** | Interactive holographic brain rendered in Three.js with real-time synapse animations |
| **GDPR-Ready** | Raw EEG processed in-memory and immediately discarded; Article 17 erasure endpoint included |

---

## ✨ Key Features

### 🔮 Intelligent Seizure Detection
- Upload multi-format EEG files and receive probability scores within seconds
- Switch between LightGBM, XGBoost, and Random Forest models on the fly
- Automatic dataset origin detection with confidence scoring
- Dynamic probability threshold calibration for clinical-grade sensitivity

### 🧪 Explainable AI (XAI)
- Integrated **SHAP** (SHapley Additive exPlanations) for every prediction
- Feature-level waterfall and bar charts with reference ranges
- Directional impact indicators showing positive/negative seizure contribution
- Clinician-friendly explanations bridging the gap between ML outputs and medical understanding

### 📡 Real-Time EEG Monitoring
- Server-Sent Events (SSE) streaming of multi-channel EEG waveforms
- Configurable sampling rates and playback speed
- Live spectral decomposition across 5 standard EEG bands (Delta, Theta, Alpha, Beta, Gamma)

### 🖥️ Clinical Dark Mode UI
- Futuristic glassmorphic design with frosted glass panels and neon accents
- Interactive 3D holographic brain visualization (Three.js / React Three Fiber)
- Responsive layouts with mobile bottom navigation
- Evaluation analytics: ROC curves, confusion matrices, precision-recall dashboards

### 🏥 Patient Management (v2)
- Doctor Dashboard with integrated patient creation and EEG analysis
- Full clinical metadata tracking (demographics, vitals, medical history)
- Informed consent recording with timestamps
- Printable patient reports with prediction history

---

## 🏗 Architecture

NeuroAegis is organized as a **monorepo** using NPM Workspaces, with clear separation between frontend, backend, and shared contracts.

```
┌─────────────────────────────────────────────────────────────────┐
│                         NGINX (TLS/SSL)                         │
│                    Reverse Proxy + HSTS + SSE                   │
└────────────────┬──────────────────────────┬─────────────────────┘
                 │                          │
       ┌─────────▼──────────┐    ┌──────────▼──────────┐
       │    apps/web         │    │    apps/api          │
       │  ────────────────── │    │  ──────────────────  │
       │  React 18 + TS      │    │  FastAPI (Python)    │
       │  Vite + TailwindCSS │    │  ML Inference Engine │
       │  Three.js (3D)      │    │  SHAP Explainer      │
       │  Framer Motion      │    │  SSE Streaming       │
       │  Zustand + TanStack │    │  Dataset Detection   │
       └─────────────────────┘    └──────────┬──────────┘
                                             │
                                   ┌─────────▼─────────┐
                                   │   PostgreSQL 15    │
                                   │  (SQLite for dev)  │
                                   └───────────────────-┘

       ┌──────────────────────────────────────────────────┐
       │          packages/model-contracts                 │
       │   TypeScript — Single Source of Truth for all     │
       │   data models, predictions, SHAP schemas,         │
       │   confidence scores, and evaluation metrics       │
       └──────────────────────────────────────────────────┘
```

### Request Flow

```mermaid
sequenceDiagram
    participant U as Clinician
    participant W as React Frontend
    participant A as FastAPI Backend
    participant D as Dataset Detector
    participant M as ML Predictor
    participant S as SHAP Explainer
    participant DB as PostgreSQL

    U->>W: Upload EEG file (.edf/.csv/.txt)
    W->>A: POST /api/v2/predict (multipart/form-data)
    A->>DB: Create Patient + PredictionJob
    A->>A: Parse & validate EEG signal
    A->>D: Auto-detect dataset origin
    D-->>A: bonn / chbmit + confidence
    A->>A: Extract 57 multi-domain features
    A->>M: Run inference (LightGBM/XGBoost/RF)
    M-->>A: Seizure probability + label
    A->>S: Generate SHAP explanation
    S-->>A: Top-N feature contributions
    A->>DB: Update job (Completed)
    A-->>W: Prediction + SHAP + confidence band
    W-->>U: Interactive result dashboard
```

---

## 📊 Model Performance

### Bonn University Dataset (Primary Benchmark)

Evaluated on a 20% held-out test set with 57 extracted features and an 80/20 train/test split.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** ⭐ | **0.9000** | **0.9015** | **0.9000** | **0.9001** | **0.9844** |
| XGBoost | 0.8800 | 0.8829 | 0.8800 | 0.8812 | 0.9846 |
| Random Forest | 0.8500 | 0.8546 | 0.8500 | 0.8518 | 0.9794 |

### CHB-MIT Patient-Independent Evaluation

Leave-One-Patient-Out Cross-Validation across 5 patients (`chb01`–`chb05`):

| Metric | Mean ± Std Dev |
|:---|:---:|
| **AUC** | 0.8168 ± 0.1103 |
| **Precision** | 0.2488 ± 0.2607 |
| **Recall** | 0.3398 ± 0.3215 |

### 🌐 Zero-Shot Cross-Dataset Transfer

Models trained on one dataset were evaluated on the other with **zero fine-tuning** — demonstrating remarkable domain generalization after independent standardization.

| Transfer Direction | AUC | Recall | Precision | Highlight |
|:---|:---:|:---:|:---:|:---|
| **Bonn → CHB-MIT** | **0.8282** | 0.7212 | 0.0588 | Outperforms in-domain CHB-MIT LOPO-CV (0.8167) |
| **CHB-MIT → Bonn** | **0.9553** | 0.3300 | **1.0000** | Perfect precision on zero-shot transfer |

> **Key Insight:** Single-channel trained models (Bonn) transfer successfully to continuous multi-channel scalp EEG (CHB-MIT) after physical unit standardization — proving learned physiological signatures are genuinely transferable across recording modalities.

For full hyperparameters, fold-by-fold breakdowns, and threshold analysis, see [`model_card.md`](./model_card.md).

---

## 🛠 Tech Stack

### Backend (Scientific Computing & API)

| Technology | Purpose |
|:---|:---|
| **FastAPI** + Uvicorn | Async REST API with automatic OpenAPI docs |
| **LightGBM** / **XGBoost** / **Scikit-learn** | Ensemble ML models for seizure classification |
| **SHAP** | Explainable AI — feature attribution for every prediction |
| **PyWavelets** (`pywt`) | Discrete Wavelet Transform denoising & feature extraction |
| **SciPy** + **NumPy** + **Pandas** | Signal processing, numerical computation, data wrangling |
| **MNE-Python** | EDF file parsing and EEG data handling |
| **SQLAlchemy** | ORM for PostgreSQL (prod) / SQLite (dev) |
| **PyTorch** | Zero-shot cross-dataset generalization experiments |

### Frontend (Clinical Dashboard)

| Technology | Purpose |
|:---|:---|
| **React 18** + **TypeScript** | Component-based UI with full type safety |
| **Vite** | Lightning-fast HMR and bundling |
| **TailwindCSS** | Utility-first styling with clinical dark mode tokens |
| **Three.js** / **React Three Fiber** / **Drei** | 3D holographic brain visualization |
| **Framer Motion** | Micro-animations and page transitions |
| **Recharts** | Data visualization (ROC curves, spectral plots) |
| **Zustand** | Lightweight global state management |
| **TanStack Query** | Server state caching and synchronization |
| **Zod** | Runtime schema validation |
| **Playwright** + **Vitest** | E2E and unit testing |

### Infrastructure

| Technology | Purpose |
|:---|:---|
| **Docker** + **Docker Compose** | Multi-container orchestration (API + DB + Web) |
| **Nginx** | Reverse proxy with TLS/SSL termination, HSTS, SSE streaming |
| **PostgreSQL 15** | Production database with persistent volumes |
| **GitHub Actions** | CI pipeline for feature parity regression tests |

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
|:---|:---|
| **Node.js** | v18+ |
| **Python** | 3.10 – 3.12 |
| **Git** | Latest |
| **Docker** *(optional)* | Latest |

### Option A — Docker (Recommended)

Spin up the full stack in one command:

```bash
# 1. Clone the repository
git clone https://github.com/Bhumi-2303/NeuroAegis.git
cd NeuroAegis

# 2. Configure environment
cp .env.example .env
# Edit .env — set SECRET_KEY and POSTGRES_PASSWORD

# 3. Launch the stack
docker-compose up --build
```

| Service | URL |
|:---|:---|
| Frontend | [`https://localhost:5173`](https://localhost:5173) |
| Backend API | [`https://localhost:8000`](https://localhost:8000) |
| API Docs (Swagger) | [`https://localhost:8000/docs`](https://localhost:8000/docs) |

### Option B — Local Development

**Terminal 1 — Backend (FastAPI)**

```bash
cd apps/api

# Create and activate virtual environment
python3.12 -m venv ../../.venv
source ../../.venv/bin/activate  # Windows: ..\..\\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Fetch model artifacts (if not already present)
bash ../../scripts/fetch_models.sh

# Start the development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend (React/Vite)**

```bash
cd apps/web

# Install dependencies
npm install

# Start the development server
npm run dev
```

| Service | URL |
|:---|:---|
| Frontend | [`http://localhost:5173`](http://localhost:5173) |
| Backend API | [`http://localhost:8000`](http://localhost:8000) |
| API Docs (Swagger) | [`http://localhost:8000/docs`](http://localhost:8000/docs) |

### Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# Frontend
VITE_API_URL=http://localhost:8000/api/v1

# Backend
MODEL_ASSETS_DIR=apps/api/models/bonn
CORS_ORIGINS=["http://localhost:5173"]
MAX_UPLOAD_SIZE=52428800      # 50MB

# Auth (REQUIRED — generate with: openssl rand -hex 32)
SECRET_KEY=<your-secret-key>

# Database
DATABASE_URL=sqlite:///./neuroaegis.db                            # Dev
# DATABASE_URL=postgresql://neuroaegis:password@localhost:5434/neuroaegis_db  # Prod
```

---

## 📡 API Reference

### v1 — Core Inference

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | Health check with model loading status |
| `GET` | `/model/info` | Active model metadata and feature list |
| `POST` | `/api/v1/predict` | Single-file EEG seizure prediction |
| `GET` | `/api/v1/stream/eeg` | SSE real-time EEG waveform streaming |

### v2 — Doctor Dashboard

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/api/v2/predict` | Patient creation + EEG analysis (background job) |
| `GET` | `/api/v2/predict/status/{job_id}` | Poll job progress (0–100%) |
| `GET` | `/api/v2/history` | Prediction job history |
| `GET` | `/api/v2/report/{job_id}` | Full patient report with prediction results |

### Data Management

| Method | Endpoint | Description |
|:---|:---|:---|
| `DELETE` | `/api/v1/data/patient/{id}` | GDPR Article 17 — permanent data erasure (admin only) |

### Example — Predict Seizure from EEG File

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@sample_eeg_data/S001_seizure.txt" \
  -F "sampling_rate=173.61"
```

<details>
<summary><strong>📋 Response Shape</strong></summary>

```json
{
  "modelName": "lightgbm",
  "prediction": {
    "label": "seizure",
    "probabilities": {
      "seizure": 0.8723,
      "non_seizure": 0.1277
    }
  },
  "confidence": {
    "value": 0.8723,
    "band": "high"
  },
  "explanation": {
    "baseValue": 0.4512,
    "features": [
      {
        "featureName": "wavelet_energy_d3",
        "value": 0.1834,
        "rawValue": 12.45,
        "referenceRange": { "low": 2.1, "high": 8.7 }
      }
    ]
  },
  "generatedAt": "2026-08-11T15:30:00Z"
}
```

</details>

For the complete API contract including Pydantic schemas and TypeScript type mappings, see [`CONTRACT.md`](./CONTRACT.md).

---

## 📂 Project Structure

```
NeuroAegis/
├── apps/
│   ├── api/                        # FastAPI backend
│   │   ├── app/
│   │   │   ├── main.py             # Application entrypoint + lifespan
│   │   │   ├── api/v1/             # v1 endpoints (predict, stream, data_deletion)
│   │   │   ├── api/v2/             # v2 endpoints (doctor dashboard)
│   │   │   ├── core/config.py      # Pydantic settings manager
│   │   │   ├── db/models.py        # SQLAlchemy ORM (Patient, PredictionJob, User)
│   │   │   ├── services/
│   │   │   │   ├── dataset_detection/  # Rule-based Bonn vs CHB-MIT detector
│   │   │   │   └── prediction/         # Strategy pattern: BasePredictor → Bonn/CHBMIT
│   │   │   └── middleware/         # Security headers (HSTS, XSS, CSP)
│   │   ├── models/                 # Serialized .pkl model artifacts
│   │   │   ├── bonn/              # LightGBM, XGBoost, RF + metadata
│   │   │   └── chbmit/            # CHB-MIT-specific models
│   │   └── requirements.txt
│   │
│   └── web/                        # React 18 + TypeScript frontend
│       ├── src/
│       │   ├── components/         # Reusable UI components
│       │   ├── features/           # Feature-sliced modules
│       │   │   ├── dashboard/      # Main dashboard + 3D brain
│       │   │   ├── eeg-monitor/    # Real-time SSE waveform viewer
│       │   │   ├── prediction/     # Upload widget + confidence gauge
│       │   │   ├── explainability/ # SHAP waterfall/bar charts
│       │   │   ├── frequency/      # Spectral band analysis
│       │   │   └── reports/        # ROC curves, confusion matrices
│       │   ├── hooks/              # Custom React hooks
│       │   ├── stores/             # Zustand state stores
│       │   └── api/                # TanStack Query API layer
│       ├── tailwind.config.ts
│       └── vite.config.ts
│
├── packages/
│   └── model-contracts/            # Shared TypeScript data contracts
│       └── src/                    # ModelInput, PredictionResult, ShapExplanation, etc.
│
├── nginx/                          # Reverse proxy configuration
│   ├── nginx.conf                  # TLS, HSTS, SSE proxy rules
│   └── Dockerfile
│
├── scripts/                        # Utility & training scripts
│   ├── fetch_models.sh             # Download/verify model artifacts
│   ├── retrain_chbmit_patient_wise.py
│   └── cross_dataset_generalization.py
│
├── notebooks/
│   └── neuroaegis-v1.ipynb         # Research notebook (EDA, training, SHAP)
│
├── docs/                           # Project documentation
│   ├── PRD.md                      # Product Requirements Document
│   ├── TRD.md                      # Technical Requirements Document
│   ├── ARCHITECTURE.md             # 5-layer architecture specification
│   ├── DESIGN.md                   # Visual design system & tokens
│   └── RULES.md                    # Coding standards & conventions
│
├── sample_eeg_data/                # Test files for quick demos
├── data/                           # Feature datasets (Parquet/CSV)
├── results/                        # Evaluation reports (zero-shot, calibration)
│
├── docker-compose.yml              # Full-stack orchestration
├── CONTRACT.md                     # API contract & schema mapping
├── PRIVACY.md                      # Privacy policy (GDPR compliant)
├── model_card.md                   # Model performance & hyperparameters
└── LICENSE                         # MIT License
```

---

## 🧠 Datasets

### Bonn University EEG Dataset

| Property | Detail |
|:---|:---|
| **Type** | Single-channel intracranial + scalp EEG |
| **Classes** | Sets A–E (Healthy → Seizure continuum) |
| **Segments** | 100 segments × 5 sets = 500 total |
| **Sampling Rate** | 173.61 Hz |
| **Duration** | 23.6 seconds per segment |
| **Use** | Primary training and benchmark evaluation |

### CHB-MIT Scalp EEG Database

| Property | Detail |
|:---|:---|
| **Type** | Continuous multi-channel scalp EEG |
| **Source** | Children's Hospital Boston (PhysioNet) |
| **Patients** | 23 pediatric subjects |
| **Channels** | 23 bipolar EEG channels |
| **Sampling Rate** | 256 Hz |
| **Use** | Patient-independent generalization & zero-shot transfer |

---

## ⚡ Signal Processing Pipeline

Every uploaded EEG signal passes through a standardized multi-domain feature extraction pipeline producing **57 features**:

```mermaid
graph LR
    A["Raw EEG Signal"] --> B["DWT Denoising<br/>(coif3, level 4)"]
    B --> C["Time Domain"]
    B --> D["Frequency Domain"]
    B --> E["Wavelet Domain"]

    C --> C1["Mean, Variance,<br/>Skewness, Kurtosis"]
    C --> C2["Hjorth Parameters<br/>(Activity, Mobility, Complexity)"]
    C --> C3["Zero Crossing Rate,<br/>Line Length"]

    D --> D1["PSD (Welch)"]
    D --> D2["Band Powers<br/>δ θ α β γ"]

    E --> E1["DWT Coefficient<br/>Energies"]
    E --> E2["DWT Coefficient<br/>Entropies"]

    C1 & C2 & C3 & D1 & D2 & E1 & E2 --> F["57-Feature Vector"]
    F --> G["ML Inference +<br/>SHAP Explanation"]
```

| Domain | Features | Count |
|:---|:---|:---:|
| **Time** | Statistical moments, Hjorth parameters, ZCR, Line Length | ~15 |
| **Frequency** | Total PSD, Delta/Theta/Alpha/Beta/Gamma band powers | ~6 |
| **Wavelet** | DWT energies and entropies (coif3, 4 levels of decomposition) | ~36 |

---

## 🔐 Privacy & Compliance

NeuroAegis enforces strict data handling practices:

- **In-Memory Processing** — Raw EEG files are processed in memory and **never persisted** to disk after analysis
- **Transient File Cleanup** — Any temporary `.edf` files are deleted immediately after reading
- **GDPR Article 17** — Dedicated `DELETE /api/v1/data/patient/{id}` endpoint for permanent erasure (admin-only)
- **Informed Consent** — Patient consent tracking with timestamps and opt-in/opt-out fields
- **Security Headers** — HSTS, X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Referrer-Policy
- **CORS Enforcement** — Strict origin allowlisting
- **Role-Based Access** — Admin, Clinician, and Researcher roles

For the complete privacy policy, see [`PRIVACY.md`](./PRIVACY.md).

---

## 🧪 Testing & CI

### Continuous Integration

A GitHub Actions workflow (`.github/workflows/parity.yml`) runs on every push to `main`/`develop` and on all pull requests:

```yaml
# Validates that API feature extraction matches offline notebook calculations
- Runs all test_parity*.py scripts
- Prevents feature calculation regressions
- Ensures floating-point precision parity between notebook and production code
```

### Running Tests Locally

```bash
# Backend parity tests
source .venv/bin/activate
python scripts/test_parity_bonn.py
python scripts/test_parity_chbmit.py

# Frontend tests
cd apps/web
npm run test          # Vitest unit tests
npm run test:e2e      # Playwright E2E tests
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository and create a feature branch from `develop`
2. **Code Standards** — Strict TypeScript (no `any`), kebab-case filenames, presentational/container component split
3. **Architecture** — Respect the 1-directional dependency flow: `ui → hooks → api → state → model`
4. **Testing** — All feature extraction changes must pass parity tests
5. **Commits** — Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `refactor:`, `docs:`)

See [`docs/RULES.md`](./docs/RULES.md) for the full coding standards and [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) for architectural constraints.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

```
MIT License • Copyright (c) 2026 Bhavini Chauhan
```

---

## 📚 Documentation Index

| Document | Description |
|:---|:---|
| [`CONTRACT.md`](./CONTRACT.md) | API contract — endpoints, schemas, type mappings |
| [`PRIVACY.md`](./PRIVACY.md) | Privacy policy — GDPR compliance, data handling |
| [`model_card.md`](./model_card.md) | Model card — hyperparameters, metrics, cross-dataset transfer |
| [`docs/PRD.md`](./docs/PRD.md) | Product Requirements Document |
| [`docs/TRD.md`](./docs/TRD.md) | Technical Requirements Document |
| [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) | System architecture & dependency rules |
| [`docs/DESIGN.md`](./docs/DESIGN.md) | Visual design system & component tokens |
| [`docs/RULES.md`](./docs/RULES.md) | Coding standards & conventions |

---

<p align="center">
  Built with 🧠 by <a href="https://github.com/Bhumi-2303"><strong>Bhavini Chauhan</strong></a>
</p>
