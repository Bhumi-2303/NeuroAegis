import logging
from typing import List, Dict, Any
import numpy as np

from app.schemas.prediction import ModelOutputSchema, PredictionResultSchema, ConfidenceScoreSchema, ShapExplanationSchema
from app.services.prediction.prediction_router import prediction_router

logger = logging.getLogger("neuroaegis.model_service")

class ModelService:
    """
    Backwards compatibility wrapper for the original ModelService.
    Delegates calls to the new PredictionRouter.
    """
    def __init__(self):
        pass
        
    @property
    def is_loaded(self):
        return prediction_router.is_loaded

    def load_artifacts(self) -> bool:
        return prediction_router.load_all_models()

    def get_prediction(self, eeg_data: np.ndarray, channel_names: List[str], fs: float = 256.0) -> ModelOutputSchema:
        predictor = prediction_router.get_predictor("bonn")
        res = predictor.get_prediction(eeg_data, channel_names, fs, model_name="lightgbm")
        
        return ModelOutputSchema(
            modelName=res["modelName"],
            prediction=PredictionResultSchema(**res["prediction"]),
            confidence=ConfidenceScoreSchema(**res["confidence"]),
            explanation=ShapExplanationSchema(**res["explanation"]),
            generatedAt=res["generatedAt"]
        )

# Singleton instance
ml_model_service = ModelService()
