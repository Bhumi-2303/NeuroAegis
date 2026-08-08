from __future__ import annotations
from fastapi import APIRouter

from app.api.v1 import health, jobs, model_info, models_api, patients, predict, metrics, stream, data_deletion
from app.api.auth import router as auth_router

api_router = APIRouter()
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(health.router, tags=["health"])
api_router.include_router(predict.router, prefix="/predict", tags=["predict"])
api_router.include_router(model_info.router, prefix="/model", tags=["model_info"])
api_router.include_router(models_api.router, prefix="/models", tags=["models"])
api_router.include_router(patients.router, prefix="/patients", tags=["patients"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
api_router.include_router(stream.router, prefix="/stream", tags=["stream"])
api_router.include_router(data_deletion.router, prefix="/data", tags=["data_privacy"])

