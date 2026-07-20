from fastapi import APIRouter
from app.api.v1 import health, predict, model_info

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(predict.router, tags=["predict"])
api_router.include_router(model_info.router, prefix="/model", tags=["model_info"])
