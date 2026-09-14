from __future__ import annotations
from typing import Any

from fastapi import APIRouter

from app.services.model_service import ml_model_service

from app.services.prediction.prediction_router import prediction_router

router = APIRouter()

@router.get("/info", response_model=dict[str, Any])
async def get_model_info(dataset: str = "chbmit", model_name: str = "lightgbm"):
    """Returns metadata about the loaded model and its required features"""
    try:
        predictor = prediction_router.get_predictor(dataset)
        importances = predictor.get_feature_importances(model_name)
    except Exception:
        importances = []

    return {
        "model_loaded": ml_model_service.is_loaded,
        "metadata": ml_model_service.metadata if hasattr(ml_model_service, "metadata") else {},
        "feature_count": len(importances),
        "features": [imp["name"] for imp in importances] if importances else [],
        "feature_importances": importances
    }
