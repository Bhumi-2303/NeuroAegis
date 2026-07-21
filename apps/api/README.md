# NeuroAegis ML Inference API

This is the production-ready FastAPI backend for the NeuroAegis project. It serves as an inference engine for pre-trained LightGBM models.

## Architecture

- **FastAPI:** High-performance async web framework.
- **Pydantic:** Schema validation mirroring the frontend TypeScript contracts (`@neuroaegis/model-contracts`).
- **ML Stack:** LightGBM, SHAP, PyWavelets, SciPy, NumPy, Pandas.

## Key Principles

1. **Inference Only:** This backend is strictly for inference. It will never train or modify the model.
2. **One-Time Startup:** The ML model (`final_lightgbm_full_dataset.pkl`), SHAP explainer, scaler, and selected features list are loaded exactly once during the application lifespan startup.
3. **Exact Training Pipeline:** The preprocessing and feature extraction logic replicates the exact methodology from the training environment, ensuring consistency.

## Directory Structure

```
apps/api/
├── app/
│   ├── main.py                    # App factory and lifespan manager
│   ├── api/v1/                    # API Endpoints
│   ├── core/                      # Configuration and Logging
│   ├── schemas/                   # Pydantic models (1:1 with frontend TS)
│   └── services/                  # Business logic (preprocessing, inference)
├── models/                        # Dataset-specific model assets
│   └── bonn/                      # Default dataset
├── requirements.txt
└── Dockerfile
```

## Model Assets Organization

Model assets should be placed in `apps/api/models/<dataset_name>/`. The `.env` variable `MODEL_ASSETS_DIR` configures the active dataset.

Required files for a dataset:
- `final_lightgbm_full_dataset.pkl`: The trained LightGBM model.
- `selected_features.json`: A list of strings defining the exact subset and order of features the model expects.

## Endpoints

### `GET /api/v1/health`
Health check endpoint. Confirms whether the model was successfully loaded at startup.

### `GET /api/v1/model/info`
Returns metadata about the loaded model and its required features.

### `POST /api/v1/predict`
Uploads EEG signal data and returns a prediction along with SHAP explanations.

- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `file`: The EEG file (.csv supported currently)
  - `sampling_rate`: (Float, default: 256.0) The sampling rate in Hz.
  - `channels`: (String, optional) Comma-separated list of channel names.

**Flow:**
1. File uploaded and read into Pandas DataFrame.
2. **Preprocessing:** Wavelet denoising applied.
3. **Feature Extraction:** Extracts Statistical, Hjorth, Spectral Entropy, Band Power, and Wavelet features.
4. **Feature Selection:** Filters and orders features strictly according to `selected_features.json`.
5. **Inference:** LightGBM predicts probability of seizure.
6. **Explanation:** SHAP TreeExplainer generates top 10 feature contributions.

## Local Development

1. Install dependencies: `pip install -r requirements.txt`
2. Run server: `uvicorn app.main:app --reload --port 8000`

For Docker, use the root-level `docker-compose.yml`.
