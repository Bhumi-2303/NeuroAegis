from __future__ import annotations
import io
import json
import uuid

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
from app.db.database import get_db
from app.db.models import Patient, PredictionJob, User
from app.services.dataset_detection import dataset_detector
from app.services.job_service import run_prediction_pipeline
from app.services.model_service import ml_model_service
from app.services.prediction.prediction_router import prediction_router
from app.core.auth import require_role

router = APIRouter()

@router.post("/predict")
async def create_prediction_job(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    weight: float = Form(...),
    height: float = Form(...),
    medical_history: str = Form(...), # JSON string
    vital_signs: str = Form(...), # JSON string
    file: UploadFile = File(...),
    sampling_rate: float = Form(256.0),
    channels: str | None = Form(None),
    db: Session = Depends(get_db)
):
    if not ml_model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded on the backend")
        
    if file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
        
    import os
    import re
    safe_filename = os.path.basename(file.filename)
    safe_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '', safe_filename)
    
    try:
        contents = await file.read()
        if safe_filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents), on_bad_lines='skip')
        elif safe_filename.endswith(".txt"):
            try:
                df = pd.read_csv(io.BytesIO(contents), sep=r'\s+|,', engine='python', on_bad_lines='skip')
            except Exception:
                df = pd.read_csv(io.BytesIO(contents), sep=None, engine='python', on_bad_lines='skip')
        else:
            raise HTTPException(status_code=400, detail="Only .csv and .txt files are supported currently")
            
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

        # Dataset Detection
        provided_fs = sampling_rate or 0.0
        det_ds, conf, rules = dataset_detector.detect(df, provided_fs)
        if conf < 0.70:
            raise HTTPException(status_code=400, detail="Unable to determine EEG dataset. Confidence is too low.")
        detected_dataset = det_ds
        
        predictor_metadata = prediction_router.get_available_models().get(detected_dataset, {}).get("dataset_info", {})
        final_sampling_rate = sampling_rate or predictor_metadata.get("sampling_rate", 256.0)
        window_length = predictor_metadata.get("window_length", 15360)
        if not isinstance(window_length, (int, float)):
            window_length = 15360
            
        eeg_data = df.values.T 
        channel_names = df.columns.tolist()

        if eeg_data.shape[1] > window_length:
            start_sample = 0
            if detected_dataset == "chbmit":
                file_match = re.search(r"(chb\d+)", safe_filename)
                if file_match:
                    patient_str = file_match.group(1)
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
                            base_name = os.path.splitext(safe_filename)[0]
                            file_idx = content.find(base_name)
                            if file_idx != -1:
                                next_file_idx = content.find("File Name:", file_idx + 1)
                                section = content[file_idx:next_file_idx if next_file_idx != -1 else len(content)]
                                start_match = re.search(r"Seizure\s+(?:\d+\s+)?Start Time:\s*(\d+)", section)
                                if start_match:
                                    start_sec = int(start_match.group(1))
                                    start_sample = int(start_sec * final_sampling_rate) - (window_length // 2)
                                    start_sample = max(0, start_sample)
                                    if start_sample + window_length > eeg_data.shape[1]:
                                        start_sample = eeg_data.shape[1] - window_length
            eeg_data = eeg_data[:, start_sample:start_sample + window_length]
            
        if channels and channels.strip():
            channel_names_input = [c.strip() for c in channels.split(",") if c.strip()]
            if len(channel_names_input) > 0:
                if len(channel_names_input) != len(df.columns):
                    raise HTTPException(status_code=400, detail=f"Number of provided channels ({len(channel_names_input)}) does not match CSV columns ({len(df.columns)})")
                channel_names = channel_names_input
        # Parse JSON fields
        try:
            parsed_medical_history = json.loads(medical_history)
            parsed_vital_signs = json.loads(vital_signs)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in medical_history or vital_signs")
            
        # Create Patient
        patient_id = str(uuid.uuid4())
        patient = Patient(
            id=patient_id,
            name=name,
            age=age,
            gender=gender,
            weight=weight,
            height=height,
            medical_history=parsed_medical_history,
            vital_signs=parsed_vital_signs
        )
        db.add(patient)
        
        # Create Job
        job_id = str(uuid.uuid4())
        job = PredictionJob(
            id=job_id,
            patient_id=patient_id,
            status="Validating",
            progress=0
        )
        db.add(job)
        db.commit()
        
        # Start background task
        background_tasks.add_task(run_prediction_pipeline, job_id, eeg_data, channel_names, final_sampling_rate, detected_dataset)
        
        return {"job_id": job_id, "patient_id": patient_id}
        
    except Exception as e:
        import logging
        logging.getLogger("neuroaegis").error(f"Failed to start prediction job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/status/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    response = {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress
    }
    
    if job.status == "Completed":
        response["result"] = {
            "prediction_label": job.prediction_label,
            "probability_seizure": job.probability_seizure,
            "confidence_band": job.confidence_band,
            "shap_explanation": job.shap_explanation
        }
        
    return response

@router.get("/history")
async def get_history(db: Session = Depends(get_db)):
    jobs = db.query(PredictionJob).order_by(PredictionJob.created_at.desc()).all()
    results = []
    for job in jobs:
        results.append({
            "job_id": job.id,
            "patient_name": job.patient.name,
            "created_at": job.created_at,
            "status": job.status,
            "prediction_label": job.prediction_label
        })
    return results

@router.get("/report/{job_id}")
async def get_report(job_id: str, db: Session = Depends(get_db)):
    job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job": {
            "id": job.id,
            "status": job.status,
            "prediction_label": job.prediction_label,
            "probability_seizure": job.probability_seizure,
            "confidence_band": job.confidence_band,
            "shap_explanation": job.shap_explanation,
            "created_at": job.created_at,
            "completed_at": job.completed_at
        },
        "patient": {
            "name": job.patient.name,
            "age": job.patient.age,
            "gender": job.patient.gender,
            "weight": job.patient.weight,
            "height": job.patient.height,
            "medical_history": job.patient.medical_history,
            "vital_signs": job.patient.vital_signs
        }
    }
