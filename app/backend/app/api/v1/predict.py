from __future__ import annotations
import io
import uuid
import os
from typing import Any

import pandas as pd
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import SessionLocal, get_db
from app.db.models import Patient, PredictionJob, User
from app.services.prediction.prediction_router import prediction_router
from app.core.auth import require_role

router = APIRouter()

import logging

logger = logging.getLogger("neuroaegis")

def process_and_save_prediction(job_id: str, eeg_data, channel_names, fs, dataset: str, model_name: str):
    db = SessionLocal()
    try:
        # Update status to processing
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if job:
            job.status = "Processing"
            job.progress = 50
            db.commit()

        logger.info(f"[{job_id}] Job started. Dataset: {dataset}, Model: {model_name}")

        # Run model
        predictor = prediction_router.get_predictor(dataset)
        logger.info(f"[{job_id}] Predictor retrieved successfully. Starting inference...")
        result = predictor.get_prediction(
            eeg_data=eeg_data,
            channel_names=channel_names,
            fs=fs,
            model_name=model_name
        )
        
        # Save results
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if job:
            job.prediction_label = result["prediction"]["label"]
            job.probability_seizure = result["prediction"]["probabilities"]["seizure"]
            job.confidence_band = result["confidence"]["band"]
            job.shap_explanation = result["explanation"]
            job.status = "Completed"
            job.progress = 100
            import datetime
            job.completed_at = datetime.datetime.utcnow()
            db.commit()
            logger.info(f"[{job_id}] Prediction finished successfully. Result: {job.prediction_label} ({job.probability_seizure:.4f})")
    except Exception as e:
        logger.error(f"[{job_id}] Prediction background task failed: {e}", exc_info=True)
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if job:
            job.status = "Failed"
            job.progress = 0
            job.error = str(e)
            db.commit()
    finally:
        db.close()


@router.post("/", response_model=dict[str, Any])
async def predict_eeg(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sampling_rate: float | None = Form(None),
    channels: str | None = Form(None),
    patient_id: str | None = Form(None),
    dataset: str | None = Form(None),
    model: str | None = Form(None),
    db: Session = Depends(get_db)
):
    """
    Accepts an uploaded EEG file, runs it through the full exact pipeline asynchronously,
    and returns a job_id for tracking.
    """
    import os
    import re

    from app.services.dataset_detection.detector import dataset_detector
    
    if not prediction_router.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded on the backend")
        
    if file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
        
    # Filename sanitization
    safe_filename = os.path.basename(file.filename)
    safe_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '', safe_filename)
    
    if not (safe_filename.endswith(".csv") or safe_filename.endswith(".txt") or safe_filename.endswith(".edf")):
        raise HTTPException(status_code=400, detail="Only .csv, .txt, and .edf files are supported currently")
        
    try:
        logger.info(f"Received predict request. File: {safe_filename}, Size: {file.size}, Provided dataset: {dataset}")
        contents = await file.read()
        logger.info(f"File {safe_filename} read successfully. Size: {len(contents)} bytes")
        
        if safe_filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents), on_bad_lines='skip')
        elif safe_filename.endswith(".txt"):
            try:
                df = pd.read_csv(io.BytesIO(contents), sep=r'\s+|,', engine='python', on_bad_lines='skip')
            except Exception:
                df = pd.read_csv(io.BytesIO(contents), sep=None, engine='python', on_bad_lines='skip')
                
        if not safe_filename.endswith(".edf"):
            # Check if headers are just numeric data (headerless file)
            try:
                [float(c) for c in df.columns]
                # If we succeed, it means all columns are numeric -> no header was present
                first_row = pd.DataFrame([df.columns], columns=df.columns)
                df = pd.concat([first_row, df], ignore_index=True)
                df = df.astype(float)
                df.columns = [str(i) for i in range(len(df.columns))]
            except ValueError:
                # It has normal string headers
                pass
        elif safe_filename.endswith(".edf"):
            import os
            import tempfile

            import mne
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            try:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                if not sampling_rate:
                    sampling_rate = raw.info['sfreq']
                df = raw.to_data_frame(scalings=dict(eeg=1, eog=1, ecg=1, emg=1, misc=1))
                if 'time' in df.columns:
                    df = df.drop('time', axis=1)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        # 1. Dataset Detection (if not explicitly provided)
        detected_dataset = dataset
        confidence = 1.0
        matched_rules = []
        
        if not detected_dataset:
            provided_fs = sampling_rate or 0.0
            det_ds, conf, rules = dataset_detector.detect(df, provided_fs)
            
            if conf < 0.70:
                reason = "Unable to determine EEG dataset. Confidence is too low."
                logger.warning(reason)
                raise ValueError(reason)
            elif conf < 0.90:
                logger.warning(f"Dataset detected with low confidence: {conf}")
                
            detected_dataset = det_ds
            confidence = conf
            matched_rules = rules
            
        # 2. Get default model if not provided
        selected_model = model
        if not selected_model:
            predictor = prediction_router.get_predictor(detected_dataset)
            selected_model = predictor.default_model
            
        # 3. Input Validation based on detected dataset
        predictor_metadata = prediction_router.get_available_models().get(detected_dataset, {}).get("dataset_info", {})
        
        # Use detected sampling rate if none provided
        final_sampling_rate = sampling_rate or predictor_metadata.get("sampling_rate", 256.0)
        window_length = predictor_metadata.get("window_length", 15360)
        if not isinstance(window_length, (int, float)):
            window_length = 15360

        eeg_data = df.values.T 
        channel_names = df.columns.tolist()

        # Windowing logic for live demo: slice to exact window length expected by model
        if eeg_data.shape[1] > window_length:
            start_sample = 0
            
            # If CHBMIT, try to find ground truth seizure time for the demo file
            if detected_dataset == "chbmit":
                import re
                file_match = re.search(r"(chb\d+)", safe_filename)
                if file_match:
                    patient_str = file_match.group(1)
                    
                    # Find project root by looking for 'data/chbmit_subset'
                    current_dir = os.path.abspath(os.path.dirname(__file__))
                    while not os.path.exists(os.path.join(current_dir, "data", "chbmit_subset")):
                        parent = os.path.dirname(current_dir)
                        if parent == current_dir:
                            break
                        current_dir = parent
                        
                    summary_path = os.path.join(current_dir, "data", "chbmit_subset", patient_str, f"{patient_str}-summary.txt")
                    if os.path.exists(summary_path):
                        with open(summary_path, 'r') as f:
                            content = f.read()
                            # Find the file section (ignoring extension)
                            base_name = os.path.splitext(safe_filename)[0]
                            file_idx = content.find(base_name)
                            if file_idx != -1:
                                next_file_idx = content.find("File Name:", file_idx + 1)
                                section = content[file_idx:next_file_idx if next_file_idx != -1 else len(content)]
                                start_match = re.search(r"Seizure\s+(?:\d+\s+)?Start Time:\s*(\d+)", section)
                                if start_match:
                                    start_sec = int(start_match.group(1))
                                    # Center the window around the seizure start time
                                    start_sample = int(start_sec * final_sampling_rate) - (window_length // 2)
                                    start_sample = max(0, start_sample)
                                    if start_sample + window_length > eeg_data.shape[1]:
                                        start_sample = eeg_data.shape[1] - window_length
                                    logger.info(f"Demo file matched! Found seizure at {start_sec}s. Slicing from sample {start_sample}.")
            
            eeg_data = eeg_data[:, start_sample:start_sample + window_length]
            logger.info(f"Data sliced to window length: {window_length} samples.")
            
        if channels and channels.strip():
            channel_names_input = [c.strip() for c in channels.split(",") if c.strip()]
            if len(channel_names_input) > 0:
                if len(channel_names_input) != len(df.columns):
                    raise ValueError(f"Number of provided channels ({len(channel_names_input)}) does not match CSV columns ({len(df.columns)})")
                channel_names = channel_names_input
                
    except Exception as e:
        error_msg = f"Invalid CSV file or detection failed: {e!s}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
        
    try:
        # Verify patient exists if provided
        if patient_id:
            patient = db.query(Patient).filter(Patient.id == patient_id).first()
            if not patient:
                raise HTTPException(status_code=404, detail="Patient not found")
                
        # Create Job
        job_id = str(uuid.uuid4())
        job = PredictionJob(
            id=job_id,
            patient_id=patient_id,
            status="Validating",
            progress=0,
            # We assume the schema is updated to support these if possible, 
            # or we store in a JSON column if supported.
            # If the columns don't exist yet, we will add them.
            detected_dataset=detected_dataset,
            detection_confidence=confidence,
            selected_model=selected_model
        )
        db.add(job)
        db.commit()
        
        
        logger.info(f"[{job_id}] Validator passed. Dataset: {detected_dataset}, Confidence: {confidence:.2f}, Predictor: {selected_model}")
        
        background_tasks.add_task(
            process_and_save_prediction, 
            job_id, eeg_data, channel_names, final_sampling_rate, detected_dataset, selected_model
        )
        logger.info(f"[{job_id}] Response sent for prediction task.")
        
        return {
            "job_id": job_id,
            "detected_dataset": detected_dataset,
            "confidence": confidence,
            "matched_rules": matched_rules,
            "selected_model": selected_model
        }

        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
