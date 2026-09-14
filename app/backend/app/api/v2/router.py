from __future__ import annotations
from fastapi import APIRouter

from app.api.v2.demo import router as demo_router
from app.api.v2.predict import router as predict_router

api_router = APIRouter()
api_router.include_router(predict_router, tags=["v2_predict"])
api_router.include_router(demo_router, prefix="/demo", tags=["v2_demo"])
