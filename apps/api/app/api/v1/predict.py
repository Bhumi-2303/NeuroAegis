from fastapi import APIRouter, File, UploadFile, HTTPException, Form, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from typing import Optional
import pandas as pd
import io
import uuid
from typing import Dict, Any

from app.db.database import get_db
from app.db.models import PredictionJob, Patient
from app.services.model_service import ml_model_service
from app.core.config import settings

router = APIRouter()

def process_and_save_prediction(job_id: str, eeg_data, channel_names, fs, db: Session):
    try:
        # Update status to processing
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if job:
            job.status = "Processing"
            job.progress = 50
            db.commit()

        # Run model
        result = ml_model_service.get_prediction(
            eeg_data=eeg_data,
            channel_names=channel_names,
            fs=fs
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
    except Exception as e:
        import logging
        logging.getLogger("neuroaegis").error(f"Prediction background task failed: {e}", exc_info=True)
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if job:
            job.status = "Failed"
            job.progress = 0
            db.commit()


@router.post("/", response_model=Dict[str, Any])
async def predict_eeg(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sampling_rate: float = Form(256.0),
    channels: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Accepts an uploaded EEG file, runs it through the full exact pipeline asynchronously,
    and returns a job_id for tracking.
    """
    if not ml_model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded on the backend")
        
    if file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
        
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported currently")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Input Validation: Check if there's any non-numeric data
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
            raise ValueError("All columns in the CSV must be numeric EEG data")
            
        if len(df) == 0:
            raise ValueError("CSV file is empty")
            
        eeg_data = df.values.T 
        channel_names = df.columns.tolist()
            
        if channels:
            channel_names = channels.split(",")
            if len(channel_names) != len(df.columns):
                raise ValueError("Number of provided channels does not match CSV columns")
                
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
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
            progress=0
        )
        db.add(job)
        db.commit()
        
        background_tasks.add_task(
            process_and_save_prediction, 
            job_id, eeg_data, channel_names, sampling_rate, db
        )
        
        return {"job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.getLogger("neuroaegis").error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
