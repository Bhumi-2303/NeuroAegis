from __future__ import annotations
import asyncio
import datetime
import logging
from functools import partial

import numpy as np
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import PredictionJob
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
    predictor = prediction_router.get_predictor(dataset_name)
    prediction_result = predictor.get_prediction(
        eeg_data=eeg_data,
        channel_names=channel_names,
        fs=fs,
        model_name=predictor.default_model,
    )
    return {
        "label": prediction_result["prediction"]["label"],
        "prob_seizure": float(prediction_result["prediction"]["probabilities"]["seizure"]),
        "band": prediction_result["confidence"]["band"],
        "explanation": prediction_result["explanation"],
    }


async def run_prediction_pipeline(
    job_id: str,
    eeg_data: np.ndarray,
    channel_names: list,
    fs: float,
    dataset_name: str = "bonn",
    eeg_visualization: dict | None = None,
):
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
                if eeg_visualization is not None:
                    job.eeg_visualization = eeg_visualization
                elif job.eeg_visualization is None:
                    from app.services.eeg_visualization import build_window_visualization
                    job.eeg_visualization = build_window_visualization(
                        eeg_data=eeg_data,
                        channel_names=channel_names,
                        fs=fs,
                    )
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
