from __future__ import annotations
import asyncio
import contextvars
import datetime
import logging
import time
from functools import partial

import numpy as np
from sqlalchemy.orm import Session

from app.core.errors import categorize_exception, sanitize_for_log
from app.core.logging import reset_logging_context, set_logging_context
from app.core.telemetry import (
    JOB_EXECUTION_SECONDS,
    JOBS_COMPLETED_TOTAL,
    JOBS_FAILED_TOTAL,
    JOBS_IN_PROGRESS,
    safe_telemetry_op,
    track_stage_latency,
)
from app.db.database import SessionLocal
from app.db.models import PredictionJob
from app.services.prediction.prediction_router import prediction_router

logger = logging.getLogger("neuroaegis.job_service")

def update_job_status(job_id: str, status: str, progress: int, expected_worker_id: str | None = None):
    """Update job status using a short-lived DB session."""
    db = SessionLocal()
    try:
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if job:
            if expected_worker_id is not None and job.worker_id != expected_worker_id:
                logger.warning(
                    f"Skipping status update for job {job_id}: worker ownership mismatch "
                    f"(expected '{expected_worker_id}', current owner is '{job.worker_id}')"
                )
                return
            if getattr(job, "is_deleted", False) or job.status in ("Completed", "Failed"):
                logger.warning(
                    f"Skipping status update for job {job_id}: job is deleted or already in terminal state '{job.status}'"
                )
                return
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
    expected_worker_id: str | None = None,
    request_id: str | None = None,
):
    tokens = set_logging_context(request_id=request_id, job_id=job_id, worker_id=expected_worker_id)
    exec_mode = "distributed" if expected_worker_id is not None else "in_process"
    safe_dataset = dataset_name if dataset_name in ("bonn", "chbmit") else "unknown"
    start_time = time.perf_counter()
    safe_telemetry_op(JOBS_IN_PROGRESS.inc, labels={"execution_mode": exec_mode})
    try:
        # Stage 1: Validating
        update_job_status(job_id, "Validating Patient Data", 10, expected_worker_id=expected_worker_id)

        # Stage 2-5: Run CPU-bound ML inference in a thread executor with propagated contextvars
        update_job_status(job_id, "Feature Extraction & Signal Processing", 25, expected_worker_id=expected_worker_id)
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        result = await loop.run_in_executor(
            None,
            ctx.run,
            partial(_run_inference, eeg_data, channel_names, fs, dataset_name),
        )

        # Stage 6: Save final results
        update_job_status(job_id, "Confidence Calculation", 95, expected_worker_id=expected_worker_id)
        with track_stage_latency("persistence", dataset=safe_dataset):
            db = SessionLocal()
            try:
                job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
                if job:
                    if expected_worker_id is not None and job.worker_id != expected_worker_id:
                        logger.warning(
                            f"Refusing to finalize job {job_id} as Completed: worker ownership mismatch "
                            f"(expected '{expected_worker_id}', current owner is '{job.worker_id}')"
                        )
                        return
                    if getattr(job, "is_deleted", False):
                        logger.warning(
                            f"Refusing to finalize job {job_id} as Completed: job was soft-deleted"
                        )
                        return
                    if job.status == "Failed":
                        logger.warning(
                            f"Refusing to finalize job {job_id} as Completed: job was reaped or marked Failed"
                        )
                        return
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
                    from app.core.audit import EVENT_CLINICAL_PREDICTION_COMPLETED, record_audit_event
                    record_audit_event(
                        db=db,
                        event_type=EVENT_CLINICAL_PREDICTION_COMPLETED,
                        outcome="success",
                        actor_type="worker" if expected_worker_id else "system",
                        actor_id=expected_worker_id or "in_process",
                        tenant_id=job.tenant_id,
                        patient_id=job.patient_id,
                        job_id=job.id,
                        resource_type="prediction_job",
                        resource_id=job.id,
                        request_id=request_id,
                        metadata={"dataset": safe_dataset, "execution_mode": exec_mode},
                    )
                    db.commit()
            finally:
                db.close()

        duration = max(0.0, time.perf_counter() - start_time)
        safe_telemetry_op(
            JOB_EXECUTION_SECONDS.observe,
            duration,
            labels={"dataset": safe_dataset, "execution_mode": exec_mode, "status": "success"},
        )
        safe_telemetry_op(
            JOBS_COMPLETED_TOTAL.inc,
            labels={"dataset": safe_dataset, "execution_mode": exec_mode},
        )

    except Exception as e:
        safe_error = categorize_exception(e)
        logger.error(f"Job {job_id} failed [{safe_error}]: {sanitize_for_log(str(e))}", exc_info=True)
        duration = max(0.0, time.perf_counter() - start_time)
        safe_telemetry_op(
            JOB_EXECUTION_SECONDS.observe,
            duration,
            labels={"dataset": safe_dataset, "execution_mode": exec_mode, "status": "failed"},
        )
        safe_telemetry_op(
            JOBS_FAILED_TOTAL.inc,
            labels={"dataset": safe_dataset, "execution_mode": exec_mode, "error_category": safe_error},
        )
        db = SessionLocal()
        try:
            job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
            if job:
                if expected_worker_id is not None and job.worker_id != expected_worker_id:
                    logger.warning(
                        f"Skipping failure record for job {job_id}: worker ownership mismatch "
                        f"(expected '{expected_worker_id}', current owner is '{job.worker_id}')"
                    )
                    return
                job.status = "Failed"
                job.progress = 0
                job.error = safe_error
                from app.core.audit import EVENT_CLINICAL_PREDICTION_FAILED, record_audit_event
                record_audit_event(
                    db=db,
                    event_type=EVENT_CLINICAL_PREDICTION_FAILED,
                    outcome="failure",
                    actor_type="worker" if expected_worker_id else "system",
                    actor_id=expected_worker_id or "in_process",
                    tenant_id=job.tenant_id,
                    patient_id=job.patient_id,
                    job_id=job.id,
                    resource_type="prediction_job",
                    resource_id=job.id,
                    request_id=request_id,
                    error_category=safe_error,
                    metadata={"dataset": safe_dataset, "execution_mode": exec_mode},
                )
                db.commit()
        finally:
            db.close()
    finally:
        safe_telemetry_op(JOBS_IN_PROGRESS.dec, labels={"execution_mode": exec_mode})
        reset_logging_context(tokens)
