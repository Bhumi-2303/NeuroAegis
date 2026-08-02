# NeuroAegis Privacy & Security Policy

## 1. Ephemeral Processing
All EEG data uploaded to NeuroAegis is processed **ephemerally**. The uploaded samples are loaded into memory for inference and are **deleted immediately** after the prediction pipeline completes. We do not store, log, or persist raw EEG recordings on our servers.

## 2. Filename Sanitization
To prevent directory traversal and injection attacks, all uploaded filenames are strictly sanitized. Any paths, special characters, or shell metacharacters are stripped out. Only standard alphanumeric characters, dashes, underscores, and supported extensions (`.csv`, `.txt`, `.edf`) are permitted.

## 3. Upload Size Limits
To prevent denial-of-service (DoS) attacks and ensure responsive service, all file uploads are capped at a maximum upload size. Files exceeding this limit are rejected at the edge layer before processing begins. Currently, the limit is typically 50 MB, as configured in the server settings (`MAX_UPLOAD_SIZE`).

## 4. Medical Disclaimer
**Not for clinical use — research prototype.** NeuroAegis is an experimental tool designed for research and demonstration purposes only. It is not intended for diagnosis, treatment, cure, or prevention of any disease. Always consult a qualified healthcare professional for medical advice and interpretation of EEG data.
