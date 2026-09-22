# NeuroAegis ML Inference & Analysis API

Production-ready FastAPI backend for the NeuroAegis explainable AI EEG seizure detection and prediction platform.

## Architecture

- **FastAPI:** High-performance async web framework.
- **Pydantic:** Type validation and schema contracts mirroring frontend TypeScript definitions (`@neuroaegis/model-contracts`).
- **ML & Scientific Stack:** LightGBM, XGBoost, Scikit-learn, SHAP, PyWavelets, SciPy, NumPy, Pandas, MNE-Python.
- **Database:** SQLAlchemy ORM with SQLite (development/testing) and PostgreSQL (production).

## Key Principles & Guardrails

1. **Inference Only:** The production backend strictly runs pre-trained models. Model weights are never modified during request processing.
2. **One-Time Startup:** Model artifacts, reference ranges, and feature configurations are loaded and cached in memory during the application lifespan startup.
3. **EDF-Only Ingestion:** Only valid European Data Format (`.edf`) files are accepted. Non-EDF uploads are immediately rejected with HTTP 400 and corrective guidance before heavy processing.
4. **Bounded EEG Visualization:** Waveform rendering extracts a bounded window (up to 23 channels, 30 seconds at 256 Hz = 7,680 points per channel) with strict memory bounds and physical unit conversion (µV/mV).
5. **Explainability by Design:** Every seizure prediction includes SHAP feature-level attributions, baseline values, and physiological reference ranges.

## Directory Structure

```
app/backend/
├── app/
│   ├── main.py                    # Application factory and lifespan manager
│   ├── api/
│   │   ├── v1/                    # v1 endpoints (synchronous predict, streaming, GDPR deletion)
│   │   └── v2/                    # v2 endpoints (asynchronous jobs, doctor dashboard)
│   ├── core/                      # Configuration, settings, event bus, and registry
│   ├── db/                        # SQLAlchemy database models, session management
│   ├── schemas/                   # Pydantic schemas (1:1 with @neuroaegis/model-contracts)
│   └── services/                  # Preprocessing, feature extraction, dataset detection, visualization
│       ├── dataset_detection/     # Rule-based detector (CHB-MIT, Siena, Bonn, Unknown)
│       ├── edf_validation.py      # Structural EDF validation and transport limits
│       └── eeg_visualization.py   # Bounded waveform extraction and channel stats
├── config/                        # YAML configurations (models, dataset validation rules)
├── models/                        # Serialized model artifacts (.pkl, metadata.json, reference ranges)
│   ├── bonn/                      # LightGBM full dataset model + SHAP explainer
│   └── chbmit/                    # Patient-wise models + reference ranges
├── tests/                         # Full test suite (96 tests: unit, integration, parity, hardening)
├── Dockerfile                     # Production container specification (Python 3.11-slim)
├── requirements.txt               # Development dependencies
└── requirements-lock.txt          # Multi-platform pinned dependencies with cryptographic hashes
```

## Endpoints

### API v1 (Core Inference)
- `GET /health` / `GET /api/v1/health`: Health check confirming model availability.
- `GET /model/info` / `GET /api/v1/model/info`: Active model metadata and feature definitions.
- `POST /api/v1/predict`: Single-file synchronous EDF seizure prediction with dataset detection, feature extraction, and SHAP explanation.
- `GET /api/v1/stream/eeg`: Server-Sent Events (SSE) EEG streaming endpoint.
- `DELETE /api/v1/data/patient/{id}`: GDPR Article 17 permanent patient data erasure.

### API v2 (Asynchronous Job Processing & Doctor Dashboard)
- `POST /api/v2/predict`: Initiates background EEG analysis job with optional patient demographics and clinical history.
- `GET /api/v2/predict/status/{job_id}`: Polls asynchronous job status (`pending`, `processing`, `completed`, `failed`).
- `GET /api/v2/history`: Returns historical prediction jobs for clinical auditing.
- `GET /api/v2/report/{job_id}`: Comprehensive clinical report with prediction scores and feature importance.

## Environment Variables

| Variable | Default | Purpose |
|:---|:---|:---|
| `MODEL_ASSETS_DIR` | `models/bonn` | Directory containing serialized model artifacts |
| `MAX_EEG_UPLOAD_BYTES` | `201326592` (192 MiB) | Maximum allowed EDF file upload size |
| `DATABASE_URL` | `sqlite:///./neuroaegis.db` | Database connection string (PostgreSQL in production) |
| `SECRET_KEY` | *(Required)* | Secret key for cryptographic signing |
| `CORS_ORIGINS` | `["http://localhost:5173"]` | Allowlisted CORS origins (JSON array or comma-separated) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Running Tests

Run the complete test suite (96 tests across core, unit, integration, dataset detection, EDF validation, visualization, parity, and production hardening):

```bash
# From app/backend directory with active virtual environment:
python -m pytest tests/ -v
```

## Running the API Locally

```bash
# From app/backend directory:
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
