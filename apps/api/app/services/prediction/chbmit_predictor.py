from __future__ import annotations
import json
import logging
import os
from typing import Any

import joblib
import numpy as np
import shap

from app.services.pipelines.chbmit.feature_extraction import extract_all_features
from app.services.pipelines.chbmit.preprocessing import preprocess_eeg

from .base_predictor import BasePredictor

logger = logging.getLogger("neuroaegis.chbmit_predictor")

class CHBMITPredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explainers = {}

    def load_model(self) -> bool:
        try:
            self.reference_ranges = {}
            ref_path = os.path.join(self.model_dir, "reference_ranges.json")
            if os.path.exists(ref_path):
                with open(ref_path, "r") as f:
                    self.reference_ranges = json.load(f)

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
                loaded = joblib.load(lgb_path)
                if isinstance(loaded, dict) and 'model' in loaded:
                    self.models['lightgbm'] = loaded['model']
                else:
                    self.models['lightgbm'] = loaded

                
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

    def extract_features(self, data: np.ndarray, channel_names: list[str], fs: float) -> tuple[np.ndarray, dict[str, float]]:
        feature_dict = extract_all_features(data, channel_names, fs)
        # Ensure correct order based on selected_features
        vector = []
        for feat in self.selected_features:
            vector.append(feature_dict.get(feat, 0.0))
        return np.array([vector]), feature_dict

    def predict(self, feature_vector: np.ndarray, model_name: str = None) -> dict[str, Any]:
        model_name = model_name or self.default_model
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available for CHBMIT")
            
        model = self.models[model_name]
        
        # Depending on the model type, `predict_proba` might be needed instead of `predict` for prob
        # Assuming model.predict returns probabilities, or using predict_proba if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feature_vector)
            prob_seizure = float(probs[0][1]) if len(probs[0]) > 1 else float(probs[0][0])
        else:
            preds = model.predict(feature_vector)
            val = preds[0]
            if isinstance(val, (np.ndarray, list)):
                prob_seizure = float(val[1]) if len(val) > 1 else float(val[0])
            else:
                prob_seizure = float(val)
            
        prob_non_seizure = 1.0 - prob_seizure
        is_seizure = prob_seizure > 0.5
        
        return {
            "label": "seizure" if is_seizure else "non_seizure",
            "probabilities": {"seizure": prob_seizure, "non_seizure": prob_non_seizure}
        }

    def generate_explanation(self, feature_vector: np.ndarray, raw_features: dict[str, float], model_name: str = None) -> dict[str, Any]:
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
            feat_data = {
                "featureName": feat_name,
                "value": float(sv[idx]),
                "rawValue": raw_features.get(feat_name, float(feature_vector[0][idx])),
                "importance": float(importances[idx])
            }
            if feat_name in self.reference_ranges:
                feat_data["referenceRange"] = self.reference_ranges[feat_name]
                
            top_features.append(feat_data)
            
        return {
            "features": top_features,
            "baseValue": float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value)
        }

    def get_feature_importances(self, model_name: str = None) -> list[dict[str, Any]]:
        model_name = model_name or self.default_model
        if model_name not in self.models:
            return []
            
        model = self.models[model_name]
        importances = None
        
        # Try to extract from the model directly
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "booster_"):  # LightGBM
            importances = model.booster_.feature_importance()
        elif hasattr(model, "named_steps"):  # Sklearn Pipeline
            last_step = list(model.named_steps.values())[-1]
            if hasattr(last_step, "feature_importances_"):
                importances = last_step.feature_importances_
                
        if importances is None:
            return []
            
        # Normalize to percentage
        total = sum(importances)
        if total > 0:
            importances = [float(i) / total * 100 for i in importances]
            
        results = []
        for i, name in enumerate(self.selected_features):
            val = float(importances[i]) if i < len(importances) else 0.0
            
            # Simple heuristic for category mapping since we just have the name string
            category = "Temporal"
            lower_name = name.lower()
            if "freq" in lower_name or "psd" in lower_name or "band" in lower_name:
                category = "Frequency"
            elif "wavelet" in lower_name or "wt" in lower_name:
                category = "Wavelet"
            elif "entropy" in lower_name or "svd" in lower_name:
                category = "Entropy"
            elif "hjorth" in lower_name or "complexity" in lower_name or "mobility" in lower_name:
                category = "Hjorth"
                
            results.append({
                "name": name,
                "value": val,
                "category": category
            })
            
        # Sort by value descending
        results.sort(key=lambda x: x["value"], reverse=True)
        return results
