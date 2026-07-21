from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

class BasePredictor(ABC):
    def __init__(self, model_dir: str, default_model: str):
        self.model_dir = model_dir
        self.default_model = default_model
        self.model = None
        self.scaler = None
        self.selected_features = []
        self.feature_names = []
        self.metadata = {}
        self.is_loaded = False
        self.models = {}  # E.g., {'lightgbm': model1, 'random_forest': model2}

    @abstractmethod
    def load_model(self) -> bool:
        """Loads model artifacts."""
        pass

    @abstractmethod
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocesses the raw EEG data."""
        pass

    @abstractmethod
    def extract_features(self, data: np.ndarray, channel_names: List[str], fs: float) -> np.ndarray:
        """Extracts features from preprocessed data."""
        pass

    @abstractmethod
    def predict(self, feature_vector: np.ndarray, model_name: str = None) -> Dict[str, Any]:
        """Runs inference and returns probabilities and labels."""
        pass

    @abstractmethod
    def generate_explanation(self, feature_vector: np.ndarray, model_name: str = None) -> Dict[str, Any]:
        """Generates SHAP explanation."""
        pass

    def get_prediction(self, eeg_data: np.ndarray, channel_names: List[str], fs: float, model_name: str = None) -> Dict[str, Any]:
        """Runs the full pipeline."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")
            
        denoised_data = self.preprocess(eeg_data)
        feature_vector = self.extract_features(denoised_data, channel_names, fs)
        prediction_res = self.predict(feature_vector, model_name)
        explanation = self.generate_explanation(feature_vector, model_name)
        
        confidence_val = max(prediction_res['probabilities'].values())
        band = "high" if confidence_val > 0.9 else "medium" if confidence_val > 0.75 else "low"
        
        import datetime
        return {
            "modelName": model_name or self.default_model,
            "prediction": prediction_res,
            "confidence": {
                "value": confidence_val,
                "band": band
            },
            "explanation": explanation,
            "generatedAt": datetime.datetime.utcnow().isoformat() + "Z"
        }
