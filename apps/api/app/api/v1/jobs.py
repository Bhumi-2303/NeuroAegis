from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.db.database import get_db
from app.db.models import PredictionJob

router = APIRouter()

@router.get("/latest", response_model=Dict[str, Any])
def get_latest_job(db: Session = Depends(get_db)):
    job = db.query(PredictionJob).order_by(PredictionJob.created_at.desc()).first()
    if not job:
        raise HTTPException(status_code=404, detail="No jobs found")
        
    response = {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "modelName": "random_forest", # Required by frontend ModelOutput schema
        "generatedAt": job.completed_at.isoformat() if job.completed_at else job.created_at.isoformat()
    }
    
    if job.status == "Completed":
        response["prediction"] = {
            "label": job.prediction_label,
            "probabilities": {
                "seizure": job.probability_seizure,
                "non_seizure": 1.0 - job.probability_seizure if job.probability_seizure is not None else 0.0
            }
        }
        response["confidence"] = {
            "value": job.probability_seizure if job.probability_seizure is not None else 0.0,
            "band": job.confidence_band
        }
        response["explanation"] = job.shap_explanation
        
        # Also include raw result field for compatibility
        response["result"] = {
            "prediction_label": job.prediction_label,
            "probability_seizure": job.probability_seizure,
            "confidence_band": job.confidence_band,
            "shap_explanation": job.shap_explanation
        }
    elif job.status == "Failed":
        response["error"] = "Job failed during processing"
        
    return response

@router.get("/{job_id}", response_model=Dict[str, Any])
def get_job(job_id: str, db: Session = Depends(get_db)):
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
    elif job.status == "Failed":
        response["error"] = "Job failed during processing"
        
    return response
