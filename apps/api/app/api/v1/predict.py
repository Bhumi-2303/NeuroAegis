from fastapi import APIRouter, File, UploadFile, HTTPException, Form, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from typing import Optional
import pandas as pd
import io
import uuid
from typing import Dict, Any

from app.db.database import get_db
from app.db.models import PredictionJob, Patient
from app.services.prediction.prediction_router import prediction_router
from app.core.config import settings

router = APIRouter()

import logging

logger = logging.getLogger("neuroaegis")

def process_and_save_prediction(job_id: str, eeg_data, channel_names, fs, dataset: str, model_name: str, db: Session):
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


@router.post("/", response_model=Dict[str, Any])
async def predict_eeg(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sampling_rate: Optional[float] = Form(None),
    channels: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None),
    dataset: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Accepts an uploaded EEG file, runs it through the full exact pipeline asynchronously,
    and returns a job_id for tracking.
    """
    from app.services.dataset_detection.detector import dataset_detector

    if not prediction_router.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded on the backend")
        
    if file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
        
    if not (file.filename.endswith(".csv") or file.filename.endswith(".txt") or file.filename.endswith(".edf")):
        raise HTTPException(status_code=400, detail="Only .csv, .txt, and .edf files are supported currently")
        
    try:
        logger.info(f"Received predict request. File: {file.filename}, Size: {file.size}, Provided dataset: {dataset}")
        contents = await file.read()
        logger.info(f"File {file.filename} read successfully. Size: {len(contents)} bytes")
        
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents), on_bad_lines='skip')
        elif file.filename.endswith(".txt"):
            try:
                df = pd.read_csv(io.BytesIO(contents), sep=r'\s+|,', engine='python', on_bad_lines='skip')
            except Exception:
                df = pd.read_csv(io.BytesIO(contents), sep=None, engine='python', on_bad_lines='skip')
        elif file.filename.endswith(".edf"):
            import mne
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            try:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                df = raw.to_data_frame()
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

        eeg_data = df.values.T 
        channel_names = df.columns.tolist()
            
        if channels:
            channel_names = channels.split(",")
            if len(channel_names) != len(df.columns):
                raise ValueError("Number of provided channels does not match CSV columns")
                
    except Exception as e:
        error_msg = f"Invalid CSV file or detection failed: {str(e)}"
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
            job_id, eeg_data, channel_names, final_sampling_rate, detected_dataset, selected_model, db
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
