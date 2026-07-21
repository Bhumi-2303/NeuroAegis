import shap
import numpy as np
import lightgbm as lgb
from typing import List, Dict, Any

class ShapExplainerService:
    def __init__(self):
        self.explainer = None
        self.feature_names = []

    def initialize(self, model: lgb.Booster, feature_names: List[str]):
        """Initialize the TreeExplainer once at startup"""
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names

    def explain_prediction(self, feature_vector: np.ndarray, top_n: int = 10) -> Dict[str, Any]:
        """
        Generate SHAP values for a single prediction and return the top_n most important features.
        """
        if not self.explainer:
            raise ValueError("ShapExplainerService not initialized")

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(feature_vector)
        
        if isinstance(shap_values, list):
            # Typically index 1 is the positive class (seizure)
            shap_values_to_use = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            # shap_values_to_use is now (n_samples, n_features)
            shap_values_to_use = shap_values_to_use[0]
            
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
                base_value = base_value[1]
        else:
            # It could be a 3D array (n_samples, n_features, n_classes) or 2D (n_samples, n_features)
            shap_values_to_use = shap_values[0] # now (n_features, n_classes) or (n_features,)
            if isinstance(shap_values_to_use, np.ndarray) and shap_values_to_use.ndim > 1:
                # Take the positive class
                shap_values_to_use = shap_values_to_use[:, 1] if shap_values_to_use.shape[1] > 1 else shap_values_to_use[:, 0]
                
            base_value = self.explainer.expected_value
            
        # Ensure base_value is a scalar
        if isinstance(base_value, (np.ndarray, list)):
            if len(base_value) > 1:
                base_value = base_value[1]
            elif len(base_value) == 1:
                base_value = base_value[0]
                
        # Ensure base_value is fully unwrapped
        while isinstance(base_value, (np.ndarray, list)):
            base_value = base_value[0]

        # Combine feature names with their SHAP values
        feature_contributions = []
        for name, val in zip(self.feature_names, shap_values_to_use):
            # Ensure val is a scalar
            while isinstance(val, (np.ndarray, list)):
                val = val[0]
            feature_contributions.append({"featureName": name, "value": float(val)})
        
        # Sort by absolute SHAP value (importance)
        feature_contributions.sort(key=lambda x: abs(x["value"]), reverse=True)
        
        return {
            "baseValue": float(base_value),
            "features": feature_contributions[:top_n]
        }

# Singleton instance
shap_service = ShapExplainerService()
