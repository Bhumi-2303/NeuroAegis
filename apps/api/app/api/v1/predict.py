from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import Optional, List
import pandas as pd
import numpy as np
import io

from app.schemas.prediction import ModelOutputSchema
from app.services.model_service import ml_model_service
from app.core.config import settings

router = APIRouter()

@router.post("/predict", response_model=ModelOutputSchema)
async def predict_eeg(
    file: UploadFile = File(...),
    sampling_rate: float = Form(256.0),
    channels: Optional[str] = Form(None)
):
    """
    Accepts an uploaded EEG file (CSV format for now), runs it through 
    the full exact pipeline, and returns the prediction and SHAP explanation.
    """
    if not ml_model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded on the backend")
        
    if file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
        
    try:
        contents = await file.read()
        
        # For simplicity, assuming CSV with channels as columns and time as rows.
        # Can easily be extended for .edf using mne
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
            # Shape should be (n_channels, n_samples) for our preprocessing
            eeg_data = df.values.T 
            channel_names = df.columns.tolist()
        else:
            raise HTTPException(status_code=400, detail="Only .csv files are supported currently")
            
        if channels:
            channel_names = channels.split(",")
            
        return ml_model_service.get_prediction(
            eeg_data=eeg_data,
            channel_names=channel_names,
            fs=sampling_rate
        )
        
    except Exception as e:
        import logging
        logging.getLogger("neuroaegis").error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
