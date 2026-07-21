# API Contract

## Existing Endpoints

### 1. `GET /health` (v1)
- **Status**: [EXISTS]
- **Ownership**: Already exists
- **Method**: `GET`
- **Path**: `/health` (via v1 router)
- **Request Body**: None
- **Response Shape**: 
  - **Pydantic Schema** (`HealthCheckSchema`):
    - `status`: str
    - `model_loaded`: bool
    - `version`: str
    - `details`: Optional[Dict[str, Any]]
  - **TypeScript Type**: Not defined in `model-contracts`

### 2. `GET /model/info` (v1)
- **Status**: [EXISTS]
- **Ownership**: Already exists
- **Method**: `GET`
- **Path**: `/model/info`
- **Request Body**: None
- **Response Shape**: 
  - **Pydantic Schema**: None explicit (returns `Dict[str, Any]`)
  - **TypeScript Type**: Not defined in `model-contracts`

### 3. `POST /predict` (v1)
- **Status**: [EXISTS]
- **Ownership**: Already exists
- **Method**: `POST`
- **Path**: `/predict` (via v1 router)
- **Request Body Shape**: `multipart/form-data`
  - `file`: UploadFile (required)
  - `sampling_rate`: float (default 256.0)
  - `channels`: Optional[str]
  *(Note: There is a `ModelInputSchema` defined in `schemas/prediction.py` and a `ModelInput` type in TS, but the v1 endpoint actually uses multipart form data.)*
- **Response Shape**:
  - **Pydantic Schema** (`ModelOutputSchema`):
    - `modelName`: Literal["random_forest", "xgboost", "lightgbm"]
    - `prediction`: `PredictionResultSchema` (label: Literal["seizure", "non_seizure"], probabilities: Dict[Literal["seizure", "non_seizure"], float])
    - `confidence`: `ConfidenceScoreSchema` (value: float, band: Literal["low", "medium", "high"])
    - `explanation`: `ShapExplanationSchema` (baseValue: float, features: List[`ShapFeatureContributionSchema`])
    - `generatedAt`: str
  - **TypeScript Type** (`ModelOutput`):
    - Maps exactly to `ModelOutputSchema`.

### 4. `POST /predict` (v2)
- **Status**: [EXISTS]
- **Ownership**: Backend owner work (needs schemas & types)
- **Method**: `POST`
- **Path**: `/predict` (via v2 router)
- **Request Body Shape**: `multipart/form-data`
  - `name`: str
  - `age`: int
  - `gender`: str
  - `weight`: float
  - `height`: float
  - `medical_history`: str (JSON string)
  - `vital_signs`: str (JSON string)
  - `file`: UploadFile
  - `sampling_rate`: float (default 256.0)
  - `channels`: Optional[str]
- **Response Shape**: 
  - **Pydantic Schema**: None explicit (returns raw dict `{"job_id": str, "patient_id": str}`)
  - **TypeScript Type**: Not defined

### 5. `GET /predict/status/{job_id}` (v2)
- **Status**: [EXISTS]
- **Ownership**: Backend owner work (needs schemas & types)
- **Method**: `GET`
- **Path**: `/predict/status/{job_id}`
- **Request Body**: None
- **Response Shape**: 
  - **Pydantic Schema**: None explicit (returns raw dict with `job_id`, `status`, `progress`, and conditionally `result` containing prediction labels and shap explanation)
  - **TypeScript Type**: Not defined

### 6. `GET /history` (v2)
- **Status**: [EXISTS]
- **Ownership**: Backend owner work (needs schemas & types)
- **Method**: `GET`
- **Path**: `/history` (via v2 router)
- **Request Body**: None
- **Response Shape**: 
  - **Pydantic Schema**: None explicit (returns raw list of dicts with job history)
  - **TypeScript Type**: Not defined

### 7. `GET /report/{job_id}` (v2)
- **Status**: [EXISTS]
- **Ownership**: Backend owner work (needs schemas & types)
- **Method**: `GET`
- **Path**: `/report/{job_id}`
- **Request Body**: None
- **Response Shape**: 
  - **Pydantic Schema**: None explicit (returns raw dict with job and patient details)
  - **TypeScript Type**: Not defined

## Unused / Unmapped Schemas and Types

- **`ModelInputSchema` / `ModelInput`**: Exists in Pydantic and TypeScript, but no endpoint currently accepts JSON representing this schema. Both v1 and v2 `/predict` use `multipart/form-data`.
- **`ReportSchema` / `Report`**: Defined in `schemas/metrics.py` and TS `metrics.types.ts`, but no endpoint returns this shape (the v2 `/report/{job_id}` returns a completely different raw dictionary).
- **`EvaluationMetricsSchema` / `EvaluationMetrics`**: Defined in schemas/types but not returned by any existing endpoint.
- **`GraphDataPointSchema` / `FrequencyBandDataSchema` / TS equivalents**: Defined but not returned by any existing endpoint.
- **`AlertSchema` / `Alert`**: Defined but not returned by any existing endpoint.

## OPEN QUESTIONS

1. **Prediction Request Discrepancy**: The frontend hook `usePrediction.ts` uses the `ModelInput` TS interface (which implies sending a JSON payload with `sessionId`, `signalWindow` array, etc.), but both `v1` and `v2` Python `predict.py` endpoints expect `multipart/form-data` with file uploads. How should this be resolved? Are we changing the backend to accept JSON arrays, or changing the frontend to upload files?
2. **Missing v2 Schemas**: The v2 endpoints (`/predict`, `/predict/status/{job_id}`, `/history`, `/report/{job_id}`) currently return raw Python dictionaries. Should Pydantic schemas and corresponding TS types be built for these? (Assuming yes, since they are missing).
3. **Report Schema Mismatch**: The v2 `/report/{job_id}` endpoint returns patient and job data, which does not match the `ReportSchema` (which contains `metrics: List[EvaluationMetricsSchema]`). Should the endpoint be updated to match the schema, or the schema updated to match the endpoint?
4. **Missing Endpoints for Unused Schemas**: There are Pydantic schemas and TS types for `metrics`, `eeg` (frequency bands), and `alerts`, but no endpoints in `v1` or `v2` serve them. Should these endpoints be built, or are the schemas extraneous?
