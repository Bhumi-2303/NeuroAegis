from fastapi import APIRouter, File, UploadFile, HTTPException, Form, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from typing import Optional, List
import pandas as pd
import numpy as np
import io
import uuid
import json

from app.db.database import get_db
from app.db.models import Patient, PredictionJob
from app.services.job_service import run_prediction_pipeline
from app.core.config import settings
from app.services.model_service import ml_model_service

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
    channels: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    if not ml_model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded on the backend")
        
    if file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
        
    try:
        contents = await file.read()
        if file.filename.endswith((".csv", ".txt")):
            df = pd.read_csv(io.BytesIO(contents), sep=None, engine='python')
            eeg_data = df.values.T 
            channel_names = df.columns.tolist()
        else:
            raise HTTPException(status_code=400, detail="Only .csv and .txt files are supported currently")
            
        if channels:
            channel_names = channels.split(",")
            
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
        background_tasks.add_task(run_prediction_pipeline, job_id, eeg_data, channel_names, sampling_rate, db)
        
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
