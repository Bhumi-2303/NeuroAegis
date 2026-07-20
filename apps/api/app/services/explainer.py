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
        
        # For binary classification in LightGBM, shap_values might be a list of arrays (one for each class)
        # or a single array representing the positive class. We'll handle both.
        if isinstance(shap_values, list):
            # Typically index 1 is the positive class (seizure)
            shap_values_to_use = shap_values[1][0]
            base_value = self.explainer.expected_value[1]
        else:
            shap_values_to_use = shap_values[0]
            # Handle array expected_value
            base_value = self.explainer.expected_value
            if isinstance(base_value, (np.ndarray, list)):
                base_value = base_value[0]
            
        # Combine feature names with their SHAP values
        feature_contributions = [
            {"featureName": name, "value": float(val)}
            for name, val in zip(self.feature_names, shap_values_to_use)
        ]
        
        # Sort by absolute SHAP value (importance)
        feature_contributions.sort(key=lambda x: abs(x["value"]), reverse=True)
        
        return {
            "baseValue": float(base_value),
            "features": feature_contributions[:top_n]
        }

# Singleton instance
shap_service = ShapExplainerService()
