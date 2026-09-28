from __future__ import annotations
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.auth import get_tenant_job, require_roles
from app.db.database import get_db
from app.db.models import PredictionJob, User

router = APIRouter()


@router.get("/latest", response_model=dict[str, Any])
def get_latest_job(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("clinician", "admin")),
):
    job = (
        db.query(PredictionJob)
        .filter(
            PredictionJob.tenant_id == current_user.tenant_id,
            PredictionJob.is_deleted == False,
        )
        .order_by(PredictionJob.created_at.desc())
        .first()
    )
    if not job:
        raise HTTPException(status_code=404, detail="No jobs found")

    response = {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "modelName": job.selected_model or "random_forest",  # Required by frontend ModelOutput schema
        "datasetName": job.detected_dataset,
        "detectionConfidence": job.detection_confidence,
        "generatedAt": job.completed_at.isoformat() if job.completed_at else job.created_at.isoformat(),
    }

    if job.status == "Completed":
        response["prediction"] = {
            "label": job.prediction_label,
            "probabilities": {
                "seizure": job.probability_seizure,
                "non_seizure": 1.0 - job.probability_seizure if job.probability_seizure is not None else 0.0,
            },
        }
        response["confidence"] = {
            "value": job.probability_seizure if job.probability_seizure is not None else 0.0,
            "band": job.confidence_band,
        }
        response["explanation"] = job.shap_explanation

        # Also include raw result field for compatibility
        response["result"] = {
            "prediction_label": job.prediction_label,
            "probability_seizure": job.probability_seizure,
            "confidence_band": job.confidence_band,
            "shap_explanation": job.shap_explanation,
            "eeg_visualization": job.eeg_visualization,
        }
    elif job.status == "Failed":
        response["error"] = job.error or "Job failed during processing"

    return response


@router.get("/{job_id}", response_model=dict[str, Any])
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("clinician", "admin")),
):
    job = get_tenant_job(job_id, db, current_user.tenant_id)

    response = {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "datasetName": job.detected_dataset,
        "detectionConfidence": job.detection_confidence,
        "modelName": job.selected_model,
    }

    if job.status == "Completed":
        response["result"] = {
            "prediction_label": job.prediction_label,
            "probability_seizure": job.probability_seizure,
            "confidence_band": job.confidence_band,
            "shap_explanation": job.shap_explanation,
            "eeg_visualization": job.eeg_visualization,
        }
    elif job.status == "Failed":
        response["error"] = job.error or "Job failed during processing"

    return response
