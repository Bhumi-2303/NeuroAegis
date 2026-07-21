from fastapi import APIRouter
from app.api.v2.predict import router as predict_router

api_router = APIRouter()
api_router.include_router(predict_router, tags=["v2_predict"])
