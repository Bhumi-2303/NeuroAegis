from fastapi import APIRouter
from app.schemas.health import HealthCheckSchema
from app.services.model_service import ml_model_service
import platform

router = APIRouter()

@router.get("/health", response_model=HealthCheckSchema)
async def health_check():
    """Basic health check and model loaded status"""
    return HealthCheckSchema(
        status="ok",
        model_loaded=ml_model_service.is_loaded,
        version="0.1.0",
        details={
            "python_version": platform.python_version()
        }
    )
