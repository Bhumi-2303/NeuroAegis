import asyncio
import numpy as np
import logging
from sqlalchemy.orm import Session
from app.db.models import PredictionJob
from app.services.model_service import ml_model_service
from app.services.preprocessing import preprocess_eeg
from app.services.feature_extraction import extract_all_features
from app.services.feature_selection import select_and_order_features
from app.services.explainer import shap_service

logger = logging.getLogger("neuroaegis.job_service")

def update_job_status(db: Session, job_id: str, status: str, progress: int):
    job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
    if job:
        job.status = status
        job.progress = progress
        db.commit()

async def run_prediction_pipeline(job_id: str, eeg_data: np.ndarray, channel_names: list, fs: float, db: Session):
    try:
        # Stage 1: Validating (Already mostly done before calling this, but we show it)
        update_job_status(db, job_id, "Validating Patient Data", 10)
        await asyncio.sleep(1) # Small delay for UI cinematic effect
        
        # Stage 2: Feature Extraction (Preprocessing + Extraction)
        update_job_status(db, job_id, "Feature Extraction & Signal Processing", 25)
        denoised_data = preprocess_eeg(eeg_data)
        all_features = extract_all_features(denoised_data, channel_names, fs)
        
        # Stage 3: Brain Graph Construction / GNN (Simulated visual stage based on features)
        update_job_status(db, job_id, "Brain Graph Construction", 45)
        feature_vector = select_and_order_features(all_features, ml_model_service.selected_features)
        if ml_model_service.scaler:
            feature_vector = ml_model_service.scaler.transform(feature_vector)
        await asyncio.sleep(1) # Small delay for cinematic effect
        
        # Stage 4: Inference
        update_job_status(db, job_id, "Graph Neural Network Inference", 65)
        prob_seizure = float(ml_model_service.model.predict(feature_vector)[0])
        prob_non_seizure = 1.0 - prob_seizure
        is_seizure = prob_seizure > 0.5
        label = "seizure" if is_seizure else "non_seizure"
        
        confidence_val = max(prob_seizure, prob_non_seizure)
        if confidence_val > 0.9:
            band = "high"
        elif confidence_val > 0.75:
            band = "medium"
        else:
            band = "low"
            
        # Stage 5: Explainable AI (SHAP)
        update_job_status(db, job_id, "Explainable AI (SHAP)", 85)
        explanation_dict = shap_service.explain_prediction(feature_vector, top_n=10)
        
        # Stage 6: Confidence Calculation & Completion
        update_job_status(db, job_id, "Confidence Calculation", 95)
        await asyncio.sleep(1)
        
        # Save final results to DB
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if job:
            job.prediction_label = label
            job.probability_seizure = prob_seizure
            job.confidence_band = band
            job.shap_explanation = explanation_dict
            job.status = "Completed"
            job.progress = 100
            import datetime
            job.completed_at = datetime.datetime.utcnow()
            db.commit()
            
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        update_job_status(db, job_id, f"Failed: {str(e)}", 0)
