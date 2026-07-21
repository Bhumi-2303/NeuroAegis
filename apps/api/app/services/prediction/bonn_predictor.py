import os
import json
import joblib
import logging
import numpy as np
from typing import Dict, Any, List

from .base_predictor import BasePredictor
from app.services.pipelines.bonn.preprocessing import preprocess_eeg
from app.services.pipelines.bonn.feature_extraction import extract_all_features
from app.services.pipelines.bonn.feature_selection import select_and_order_features
from app.services.explainer import shap_service

logger = logging.getLogger("neuroaegis.bonn_predictor")

class BonnPredictor(BasePredictor):
    def load_model(self) -> bool:
        try:
            metadata_path = os.path.join(self.model_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
            
            features_path = os.path.join(self.model_dir, "selected_features.json")
            if not os.path.exists(features_path):
                logger.error(f"Bonn: selected_features.json not found at {features_path}.")
                return False
            with open(features_path, "r") as f:
                self.selected_features = json.load(f)

            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

            model_path = os.path.join(self.model_dir, "final_lightgbm_full_dataset.pkl")
            if not os.path.exists(model_path):
                logger.error(f"Bonn: Model file not found at {model_path}.")
                return False
                
            self.models['lightgbm'] = joblib.load(model_path)
            shap_service.initialize(self.models['lightgbm'], self.selected_features)
            
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"Bonn: Error loading artifacts: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        return preprocess_eeg(data)

    def extract_features(self, data: np.ndarray, channel_names: List[str], fs: float) -> np.ndarray:
        all_features = extract_all_features(data, channel_names, fs)
        feature_vector = select_and_order_features(all_features, self.selected_features)
        if self.scaler:
            feature_vector = self.scaler.transform(feature_vector)
        return feature_vector

    def predict(self, feature_vector: np.ndarray, model_name: str = None) -> Dict[str, Any]:
        model_name = model_name or self.default_model
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available for Bonn")
            
        model = self.models[model_name]
        prob_seizure = float(model.predict(feature_vector)[0])
        prob_non_seizure = 1.0 - prob_seizure
        
        is_seizure = prob_seizure > 0.5
        return {
            "label": "seizure" if is_seizure else "non_seizure",
            "probabilities": {"seizure": prob_seizure, "non_seizure": prob_non_seizure}
        }

    def generate_explanation(self, feature_vector: np.ndarray, model_name: str = None) -> Dict[str, Any]:
        # Using the shared shap_service as it was initialized in load_model
        # Ideally, each predictor should have its own SHAP instance if they can be called concurrently.
        # But for now, we'll keep the logic the same as previous.
        return shap_service.explain_prediction(feature_vector, top_n=10)
