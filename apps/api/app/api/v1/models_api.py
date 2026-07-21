from fastapi import APIRouter
from app.services.prediction.prediction_router import prediction_router
from typing import Dict, Any

router = APIRouter()

@router.get("/", response_model=Dict[str, Any])
async def get_models():
    """Returns metadata about all loaded models and datasets."""
    return prediction_router.get_available_models()
