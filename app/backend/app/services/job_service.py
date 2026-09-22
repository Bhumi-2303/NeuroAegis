from __future__ import annotations
import asyncio
import datetime
import logging
from functools import partial

import numpy as np
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import PredictionJob
from app.services.explainer import shap_service
from app.services.features import extract_features as extract_all_features
from app.services.features import preprocess_eeg, select_and_order_features
from app.services.prediction.prediction_router import prediction_router

logger = logging.getLogger("neuroaegis.job_service")

def update_job_status(job_id: str, status: str, progress: int):
    """Update job status using a short-lived DB session."""
    db = SessionLocal()
    try:
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if job:
            job.status = status
            job.progress = progress
            db.commit()
    finally:
        db.close()


def _run_inference(eeg_data: np.ndarray, channel_names: list, fs: float, dataset_name: str) -> dict:
    """CPU-bound inference work — runs in a thread executor."""
    denoised_data = preprocess_eeg(eeg_data)
    all_features = extract_all_features(denoised_data, channel_names, fs)

    predictor = prediction_router.get_predictor(dataset_name)
    feature_vector = select_and_order_features(all_features, predictor.selected_features)
    if predictor.scaler:
        feature_vector = predictor.scaler.transform(feature_vector)

    model = predictor.models.get('lightgbm') or predictor.models.get(predictor.default_model)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feature_vector)
        prob_seizure = float(probs[0][1]) if len(probs[0]) > 1 else float(probs[0][0])
    else:
        preds = model.predict(feature_vector)
        val = preds[0]
        if isinstance(val, (np.ndarray, list)):
            prob_seizure = float(val[1]) if len(val) > 1 else float(val[0])
        else:
            prob_seizure = float(val)

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

    explanation_dict = shap_service.explain_prediction(feature_vector, top_n=10)

    for feat in explanation_dict.get("features", []):
        name = feat["featureName"]
        feat["rawValue"] = all_features.get(name)

        if predictor.scaler and name in predictor.selected_features:
            idx = predictor.selected_features.index(name)
            if hasattr(predictor.scaler, 'mean_') and hasattr(predictor.scaler, 'scale_'):
                mean = float(predictor.scaler.mean_[idx])
                std = float(predictor.scaler.scale_[idx])
                feat["referenceRange"] = [mean - std, mean + std]

    return {
        "label": label,
        "prob_seizure": prob_seizure,
        "band": band,
        "explanation": explanation_dict,
    }


async def run_prediction_pipeline(job_id: str, eeg_data: np.ndarray, channel_names: list, fs: float, dataset_name: str = "bonn"):
    try:
        # Stage 1: Validating
        update_job_status(job_id, "Validating Patient Data", 10)

        # Stage 2-5: Run CPU-bound ML inference in a thread executor
        update_job_status(job_id, "Feature Extraction & Signal Processing", 25)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(_run_inference, eeg_data, channel_names, fs, dataset_name),
        )

        # Stage 6: Save final results
        update_job_status(job_id, "Confidence Calculation", 95)
        db = SessionLocal()
        try:
            job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
            if job:
                job.prediction_label = result["label"]
                job.probability_seizure = result["prob_seizure"]
                job.confidence_band = result["band"]
                job.shap_explanation = result["explanation"]
                job.status = "Completed"
                job.progress = 100
                job.completed_at = datetime.datetime.utcnow()
                db.commit()
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        db = SessionLocal()
        try:
            job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
            if job:
                job.status = "Failed"
                job.progress = 0
                job.error = str(e)
                db.commit()
        finally:
            db.close()
