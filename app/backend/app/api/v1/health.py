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

    predictors_info = {}
    if prediction_router.is_loaded and prediction_router._predictors:
        for ds_name, predictor in prediction_router._predictors.items():
            meta = getattr(predictor, "metadata", {}) or {}
            predictors_info[ds_name] = {
                "model": meta.get("model") or meta.get("version"),
                "dataset": meta.get("dataset", ds_name),
                "features": meta.get("features")
            }
        # Primary loaded dataset
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
            "python_version": platform.python_version(),
            "loaded_datasets": list(prediction_router._predictors.keys()) if prediction_router._predictors else [],
            "predictors": predictors_info
        }
    )
