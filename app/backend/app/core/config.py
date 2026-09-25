from __future__ import annotations
import os
import tempfile
from typing import Any

import yaml
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# The largest locally observed CHB-MIT EDF is 177,285,376 bytes. The 192 MiB
# default leaves 24,041,216 bytes of margin for legitimate dataset variation.
DEFAULT_MAX_EEG_UPLOAD_BYTES = 192 * 1024 * 1024
DEFAULT_DEV_SECRET_KEY = "ff948120d6c9b84a43343111fb6c417098725c0603c22d40742cc7aa8037efa1"


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    API_V2_STR: str = "/api/v2"
    PROJECT_NAME: str = "NeuroAegis API"
    
    # Environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "info"
    ENABLE_DOCS: bool | None = None
    
    # Database
    DATABASE_URL: str = "sqlite:///./neuroaegis.db"
    
    # Redis / Queue Infrastructure (Prompt 7 & 7.4)
    REDIS_URL: str = "redis://redis:6379/0"
    QUEUE_NAME: str = "neuroaegis:jobs"
    WORKER_MAX_JOBS: int = 2
    JOB_TIMEOUT_SECONDS: int = 120

    @property
    def WORKER_CONCURRENCY(self) -> int:
        """Backward compatibility alias for canonical WORKER_MAX_JOBS."""
        return self.WORKER_MAX_JOBS

    # Worker Lifecycle & Leases (Prompt 7.4)
    WORKER_HEARTBEAT_INTERVAL_SECONDS: int = 5
    JOB_LEASE_TIMEOUT_SECONDS: int = 30
    STORAGE_ORPHAN_GRACE_SECONDS: int = 300
    REAPER_INTERVAL_SECONDS: int = 10

    # Shared Staged Storage (Prompt 7)
    STORAGE_DIR: str = os.environ.get(
        "STORAGE_DIR",
        "/app/storage" if os.path.exists("/app") else os.path.join(tempfile.gettempdir(), "neuroaegis_storage")
    )

    # Controlled Queue Dispatch (Prompt 7.3)
    ENABLE_DISTRIBUTED_QUEUE: bool = False

    # Auth
    SECRET_KEY: str = DEFAULT_DEV_SECRET_KEY
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # CORS
    CORS_ALLOWED_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    
    # Model Configuration
    # Uses absolute path calculation based on project root if running locally
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Kept for backwards compatibility temporarily
    MODEL_ASSETS_DIR: str = os.path.join(BASE_DIR, "models", "bonn")
    
    # Transport limit only; EDF validity and signal-memory limits are separate.
    MAX_EEG_UPLOAD_BYTES: int = DEFAULT_MAX_EEG_UPLOAD_BYTES
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @field_validator("CORS_ALLOWED_ORIGINS", "CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v_trimmed = v.strip()
            if v_trimmed.startswith("[") and v_trimmed.endswith("]"):
                import json
                try:
                    return json.loads(v_trimmed)
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        if isinstance(v, (list, tuple, set)):
            return list(v)
        return []

    @model_validator(mode="after")
    def validate_production_and_sync(self) -> Settings:
        # Default docs enabled state
        if self.ENABLE_DOCS is None:
            self.ENABLE_DOCS = self.ENVIRONMENT.lower() != "production"

        # Sync CORS aliases
        if self.CORS_ORIGINS != self.CORS_ALLOWED_ORIGINS:
            default_origins = [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ]
            if self.CORS_ALLOWED_ORIGINS != default_origins:
                self.CORS_ORIGINS = self.CORS_ALLOWED_ORIGINS
            else:
                self.CORS_ALLOWED_ORIGINS = self.CORS_ORIGINS

        # Production security validations
        if self.ENVIRONMENT.lower() == "production":
            if (
                not self.SECRET_KEY
                or self.SECRET_KEY == DEFAULT_DEV_SECRET_KEY
                or len(self.SECRET_KEY) < 32
                or self.SECRET_KEY.lower() in ("secret", "changeme", "default")
            ):
                raise ValueError(
                    "Production configuration error: SECRET_KEY must be a strong, unique secret (at least 32 characters) set via the environment."
                )
            if "*" in self.CORS_ALLOWED_ORIGINS and self.CORS_ALLOW_CREDENTIALS:
                raise ValueError(
                    "Production configuration error: Wildcard '*' in CORS allowed origins cannot be combined with CORS_ALLOW_CREDENTIALS."
                )

        return self

    @property
    def MODELS_CONFIG(self) -> dict[str, Any]:
        config_path = os.path.join(self.BASE_DIR, "config", "models.yaml")
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            import logging
            logging.getLogger("neuroaegis").error(f"Failed to load models.yaml: {e}")
            return {"models": {}}

settings = Settings()
