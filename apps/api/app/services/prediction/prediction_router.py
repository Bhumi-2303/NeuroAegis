import os
import logging
from typing import Dict, Any, Type
from app.core.config import settings
from .base_predictor import BasePredictor

logger = logging.getLogger("neuroaegis.router")

class PredictionRouter:
    def __init__(self):
        self._predictors: Dict[str, BasePredictor] = {}
        self.is_loaded = False

    def load_all_models(self):
        """Loads all enabled models defined in models.yaml"""
        logger.info("Loading all models from registry...")
        
        config = settings.MODELS_CONFIG.get("models", {})
        
        # Avoid circular imports by lazy loading
        from .bonn_predictor import BonnPredictor
        from .chbmit_predictor import CHBMITPredictor
        
        registry_map: Dict[str, Type[BasePredictor]] = {
            "bonn": BonnPredictor,
            "chbmit": CHBMITPredictor
        }
        
        for dataset, cfg in config.items():
            if not cfg.get("enabled", False):
                continue
                
            if dataset not in registry_map:
                logger.warning(f"Predictor for dataset '{dataset}' not implemented.")
                continue
                
            model_dir = os.path.join(settings.BASE_DIR, cfg.get("path", ""))
            predictor_class = registry_map[dataset]
            predictor = predictor_class(
                model_dir=model_dir, 
                default_model=cfg.get("default_model", "lightgbm")
            )
            
            success = predictor.load_model()
            if success:
                self._predictors[dataset] = predictor
                logger.info(f"Successfully loaded '{dataset}' predictor.")
            else:
                logger.error(f"Failed to load '{dataset}' predictor.")
                
        self.is_loaded = len(self._predictors) > 0
        return self.is_loaded

    def get_predictor(self, dataset: str) -> BasePredictor:
        if dataset not in self._predictors:
            raise ValueError(f"Dataset '{dataset}' is not supported or not loaded.")
        return self._predictors[dataset]

    def get_available_models(self) -> Dict[str, Any]:
        """Returns metadata about loaded models for the frontend."""
        available = {}
        config = settings.MODELS_CONFIG.get("models", {})
        for dataset, predictor in self._predictors.items():
            dataset_cfg = config.get(dataset, {})
            available[dataset] = {
                "loaded": predictor.is_loaded,
                "models": list(predictor.models.keys()),
                "default_model": predictor.default_model,
                "metadata": predictor.metadata,
                "dataset_info": {
                    "display_name": dataset_cfg.get("display_name", dataset),
                    "expected_channels": dataset_cfg.get("expected_channels", 1),
                    "sampling_rate": dataset_cfg.get("sampling_rate", 256.0),
                    "feature_count": dataset_cfg.get("feature_count", 0),
                    "window_length": dataset_cfg.get("window_length", 4097)
                }
            }
        return available

# Singleton instance
prediction_router = PredictionRouter()
