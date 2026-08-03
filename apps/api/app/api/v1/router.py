from __future__ import annotations
from fastapi import APIRouter

from app.api.v1 import health, jobs, model_info, models_api, patients, predict

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(predict.router, prefix="/predict", tags=["predict"])
api_router.include_router(model_info.router, prefix="/model", tags=["model_info"])
api_router.include_router(models_api.router, prefix="/models", tags=["models"])
api_router.include_router(patients.router, prefix="/patients", tags=["patients"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])

