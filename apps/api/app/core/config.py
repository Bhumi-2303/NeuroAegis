from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict, Any
import os
import yaml

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    API_V2_STR: str = "/api/v2"
    PROJECT_NAME: str = "NeuroAegis API"
    
    # Environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "info"
    
    # Database
    DATABASE_URL: str = "sqlite:///./neuroaegis.db"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Model Configuration
    # Uses absolute path calculation based on project root if running locally
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Kept for backwards compatibility temporarily
    MODEL_ASSETS_DIR: str = os.path.join(BASE_DIR, "models", "bonn")
    
    # Upload limits
    MAX_UPLOAD_SIZE: int = 52428800 # 50 MB
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @property
    def MODELS_CONFIG(self) -> Dict[str, Any]:
        config_path = os.path.join(self.BASE_DIR, "config", "models.yaml")
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            import logging
            logging.getLogger("neuroaegis").error(f"Failed to load models.yaml: {e}")
            return {"models": {}}

settings = Settings()
