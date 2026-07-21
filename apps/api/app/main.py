from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.api.v1.router import api_router as api_v1_router
try:
    from app.api.v2.router import api_router as api_v2_router
except ImportError:
    api_v2_router = None # Will implement v2 router soon
from app.services.model_service import ml_model_service
from app.db.database import engine, Base
from app.db.models import Patient, PredictionJob

logger = logging.getLogger("neuroaegis")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI app.
    Loads the ML model and SHAP explainer exactly once during startup.
    """
    logger.info("Application startup: Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    logger.info("Application startup: Loading ML artifacts...")
    success = ml_model_service.load_artifacts()
    if success:
        logger.info("ML artifacts loaded successfully.")
    else:
        logger.warning("Failed to load ML artifacts. API will start in degraded mode.")
        
    yield
    
    logger.info("Application shutdown: Cleaning up resources...")
    # Add cleanup logic if necessary

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Set all CORS enabled origins
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_v1_router, prefix=settings.API_V1_STR)

if api_v2_router:
    app.include_router(api_v2_router, prefix=settings.API_V2_STR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
