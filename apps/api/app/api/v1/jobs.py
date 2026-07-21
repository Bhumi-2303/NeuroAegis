from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.db.database import get_db
from app.db.models import PredictionJob

router = APIRouter()

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
