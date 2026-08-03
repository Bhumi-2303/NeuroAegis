from __future__ import annotations
import platform

from fastapi import APIRouter

from app.schemas.health import HealthCheckSchema
from app.services.model_service import ml_model_service
from app.services.prediction.prediction_router import prediction_router

router = APIRouter()

@router.get("/health", response_model=HealthCheckSchema)
async def health_check():
    """Basic health check and model loaded status"""

    # Derive model_version and dataset_name from the first loaded predictor
    model_version = None
    dataset_name = None

    if prediction_router.is_loaded and prediction_router._predictors:
        # Report on the first loaded dataset (primary)
        first_dataset = next(iter(prediction_router._predictors))
        dataset_name = first_dataset
        predictor = prediction_router._predictors[first_dataset]
        metadata = getattr(predictor, "metadata", {}) or {}
        model_version = metadata.get("version") or metadata.get("model", None)

    last_load_time = getattr(prediction_router, "last_load_time", None)

    return HealthCheckSchema(
        status="ok",
        model_loaded=ml_model_service.is_loaded,
        version="0.1.0",
        model_version=model_version,
        dataset_name=dataset_name,
        last_load_time=last_load_time,
        details={
            "python_version": platform.python_version()
        }
    )
