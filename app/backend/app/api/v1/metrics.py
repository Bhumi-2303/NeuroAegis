import json
import logging
import os
from typing import Any
from fastapi import APIRouter, HTTPException
from app.core.config import settings

logger = logging.getLogger("neuroaegis.metrics")
router = APIRouter()

@router.get("", response_model=dict[str, Any])
async def get_metrics():
    """Returns stored validation metrics"""
    metrics_path = os.path.join(settings.BASE_DIR, "data", "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics not found")
        
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load metrics")
