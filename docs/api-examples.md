# API Examples

## POST `/api/v1/predict/`

This endpoint accepts an EEG file (CSV, TXT, or EDF) and starts a prediction job asynchronously.

**Content-Type**: `multipart/form-data`

### Sample Request

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_eeg.csv;type=text/csv' \
  -F 'sampling_rate=256'
```

### Sample Response

```json
{
  "job_id": "c6131f6a-4951-4e41-b847-0d5b4a053c89",
  "detected_dataset": "chbmit",
  "confidence": 1.0,
  "matched_rules": [
    "chbmit_channels"
  ],
  "selected_model": "lightgbm"
}
```

The frontend can then poll the `/api/v1/predict/status/{job_id}` endpoint (or the v2 equivalent) using this `job_id` to get the final prediction results.
