import logging
import platform

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.db.database import engine
from app.schemas.health import HealthCheckSchema
from app.services.model_service import ml_model_service
from app.services.prediction.prediction_router import prediction_router

router = APIRouter()
logger = logging.getLogger("neuroaegis.health")


@router.get("/health", response_model=HealthCheckSchema)
async def health_check():
    """Basic health check and model loaded status"""
    predictors_info = {}
    model_version: str | None = None
    dataset_name: str | None = None

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


@router.get("/liveness")
async def liveness_probe():
    """Lightweight liveness probe indicating process is active."""
    return {"status": "alive"}


@router.get("/readiness")
async def readiness_probe():
    """Readiness probe verifying DB connectivity and model readiness."""
    db_healthy = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_healthy = True
    except Exception as exc:
        logger.error(f"Readiness DB check failed: {exc}")

    models_ready = bool(ml_model_service.is_loaded or prediction_router.is_loaded)

    redis_healthy = True
    from app.core.config import settings
    if settings.ENABLE_DISTRIBUTED_QUEUE:
        try:
            from app.services.queue import prediction_queue
            pool = await prediction_queue.get_pool()
            await pool.ping()
        except Exception as exc:
            logger.error(f"Readiness check Redis error: {exc}")
            redis_healthy = False

    if not db_healthy or not redis_healthy:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "database": "healthy" if db_healthy else "unhealthy",
                "redis": "healthy" if redis_healthy else "unhealthy" if settings.ENABLE_DISTRIBUTED_QUEUE else "disabled",
                "models": "loaded" if models_ready else "unloaded",
            },
        )

    return {
        "status": "ready",
        "database": "healthy",
        "redis": "healthy" if settings.ENABLE_DISTRIBUTED_QUEUE else "disabled",
        "models": "loaded" if models_ready else "degraded",
    }
