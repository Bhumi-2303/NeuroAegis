from __future__ import annotations
from typing import Any

from fastapi import APIRouter

from app.services.prediction.prediction_router import prediction_router

router = APIRouter()

@router.get("/", response_model=dict[str, Any])
async def get_models():
    """Returns metadata about all loaded models and datasets."""
    return prediction_router.get_available_models()
