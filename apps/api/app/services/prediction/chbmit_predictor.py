import os
import json
import joblib
import logging
import numpy as np
import shap
from typing import Dict, Any, List

from .base_predictor import BasePredictor
from app.services.pipelines.chbmit.preprocessing import preprocess_eeg
from app.services.pipelines.chbmit.feature_extraction import extract_all_features

logger = logging.getLogger("neuroaegis.chbmit_predictor")

class CHBMITPredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explainers = {}

    def load_model(self) -> bool:
        try:
            metadata_path = os.path.join(self.model_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
            
            features_path = os.path.join(self.model_dir, "selected_features.json")
            if not os.path.exists(features_path):
                logger.error(f"CHBMIT: selected_features.json not found at {features_path}.")
                return False
            with open(features_path, "r") as f:
                self.selected_features = json.load(f)
                
            self.feature_names = self.selected_features

            # Load models
            lgb_path = os.path.join(self.model_dir, "lightgbm_baseline.pkl")
            if os.path.exists(lgb_path):
                self.models['lightgbm'] = joblib.load(lgb_path)
                
            rf_path = os.path.join(self.model_dir, "random_forest_baseline.pkl")
            if os.path.exists(rf_path):
                self.models['random_forest'] = joblib.load(rf_path)
                
            if not self.models:
                logger.error(f"CHBMIT: No models found in {self.model_dir}.")
                return False

            # Initialize SHAP explainers specific to CHB-MIT
            for m_name, m in self.models.items():
                try:
                    self.explainers[m_name] = shap.TreeExplainer(m)
                except Exception as e:
                    logger.warning(f"Could not initialize SHAP for CHBMIT {m_name}: {e}")
            
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"CHBMIT: Error loading artifacts: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        return preprocess_eeg(data)

    def extract_features(self, data: np.ndarray, channel_names: List[str], fs: float) -> np.ndarray:
        feature_dict = extract_all_features(data, channel_names, fs)
        # Ensure correct order based on selected_features
        vector = []
        for feat in self.selected_features:
            vector.append(feature_dict.get(feat, 0.0))
        return np.array([vector])

    def predict(self, feature_vector: np.ndarray, model_name: str = None) -> Dict[str, Any]:
        model_name = model_name or self.default_model
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available for CHBMIT")
            
        model = self.models[model_name]
        
        # Depending on the model type, `predict_proba` might be needed instead of `predict` for prob
        # Assuming model.predict returns probabilities, or using predict_proba if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feature_vector)[0]
            prob_seizure = float(probs[1]) if len(probs) > 1 else float(probs[0])
        else:
            prob_seizure = float(model.predict(feature_vector)[0])
            
        prob_non_seizure = 1.0 - prob_seizure
        is_seizure = prob_seizure > 0.5
        
        return {
            "label": "seizure" if is_seizure else "non_seizure",
            "probabilities": {"seizure": prob_seizure, "non_seizure": prob_non_seizure}
        }

    def generate_explanation(self, feature_vector: np.ndarray, model_name: str = None) -> Dict[str, Any]:
        model_name = model_name or self.default_model
        explainer = self.explainers.get(model_name)
        if not explainer:
            return {"error": "SHAP not available for this model"}
            
        shap_values = explainer.shap_values(feature_vector)
        # Assuming binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            sv = shap_values[0]
            
        # Top 10 features
        importances = np.abs(sv)
        top_indices = np.argsort(importances)[-10:][::-1]
        
        top_features = []
        for idx in top_indices:
            feat_name = self.selected_features[idx] if idx < len(self.selected_features) else f"Feature_{idx}"
            top_features.append({
                "name": feat_name,
                "value": float(feature_vector[0][idx]),
                "shap_value": float(sv[idx]),
                "importance": float(importances[idx])
            })
            
        return {
            "features": top_features,
            "base_value": float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value)
        }
