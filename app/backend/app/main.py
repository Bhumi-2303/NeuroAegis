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
    Ensures database schema compatibility and runs background stale job reaper.
    """
    logger.info("Application startup: Creating database tables...")
    Base.metadata.create_all(bind=engine)
    from app.db.database import ensure_schema_compatibility
    ensure_schema_compatibility(engine)
    
    logger.info("Application startup: Loading ML artifacts via Prediction Router...")
    from app.services.prediction.prediction_router import prediction_router
    success = prediction_router.load_all_models()
    if success:
        prediction_router.last_load_time = datetime.now(timezone.utc).isoformat()
        logger.info("ML models loaded successfully.")
    else:
        prediction_router.last_load_time = None
        logger.warning("Failed to load some or all ML models. API will start in degraded mode.")

    # Background reaper for stale jobs and orphaned payloads
    import asyncio
    reaper_stop = asyncio.Event()

    async def _api_reaper_loop():
        from app.services.job_recovery import cleanup_orphaned_staged_files, reap_stale_jobs
        from app.db.database import SessionLocal
        while not reaper_stop.is_set():
            try:
                await asyncio.wait_for(reaper_stop.wait(), timeout=settings.REAPER_INTERVAL_SECONDS)
                break
            except asyncio.TimeoutError:
                pass
            if reaper_stop.is_set():
                break
            try:
                db = SessionLocal()
                try:
                    reap_stale_jobs(db)
                    cleanup_orphaned_staged_files(db=db)
                finally:
                    db.close()
            except Exception as exc:
                logger.error(f"API background reaper notice: {exc}")

    reaper_task = asyncio.create_task(_api_reaper_loop())
        
    yield
    
    logger.info("Application shutdown: Cleaning up resources...")
    reaper_stop.set()
    try:
        await asyncio.wait_for(reaper_task, timeout=2.0)
    except Exception:
        pass


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
