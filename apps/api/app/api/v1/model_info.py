from fastapi import APIRouter
from app.services.model_service import ml_model_service
from typing import Dict, Any

router = APIRouter()

@router.get("/info", response_model=Dict[str, Any])
async def get_model_info():
    """Returns metadata about the loaded model and its required features"""
    return {
        "model_loaded": ml_model_service.is_loaded,
        "metadata": ml_model_service.metadata,
        "feature_count": len(ml_model_service.selected_features),
        "features": ml_model_service.selected_features,
    }
