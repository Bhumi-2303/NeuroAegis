from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1.router import api_router as api_v1_router
from app.core.config import settings

try:
    from app.api.v2.router import api_router as api_v2_router
except ImportError:
    api_v2_router = None # Will implement v2 router soon
from app.db.database import Base, engine

logger = logging.getLogger("neuroaegis")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds security headers to every response."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if settings.ENVIRONMENT != "development":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI app.
    Loads the ML models and SHAP explainers exactly once during startup.
    """
    logger.info("Application startup: Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    logger.info("Application startup: Loading ML artifacts via Prediction Router...")
    from app.services.prediction.prediction_router import prediction_router
    success = prediction_router.load_all_models()
    if success:
        prediction_router.last_load_time = datetime.now(timezone.utc).isoformat()
        logger.info("ML models loaded successfully.")
    else:
        prediction_router.last_load_time = None
        logger.warning("Failed to load some or all ML models. API will start in degraded mode.")
        
    yield
    
    logger.info("Application shutdown: Cleaning up resources...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Security middleware
app.add_middleware(SecurityHeadersMiddleware)

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
