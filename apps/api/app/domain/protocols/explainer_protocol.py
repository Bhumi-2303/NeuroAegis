from typing import Protocol, runtime_checkable, Dict, Any, List
import numpy as np
from .plugin_protocol import PluginProtocol
from .model_protocol import ModelProtocol

@runtime_checkable
class ExplainerProtocol(PluginProtocol, Protocol):
    """
    Protocol for explainability (XAI) plugins.
    Generates feature importance metrics (e.g., SHAP, LIME).
    """
    
    def explain(
        self, 
        model: ModelProtocol, 
        features: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Generate feature importance scores.
        
        Args:
            model (ModelProtocol): The loaded model plugin used for prediction.
            features (np.ndarray): The feature vector to explain.
            feature_names (List[str]): Names of the features.
            
        Returns:
            Dict[str, float]: Mapping of feature names to their importance/contribution scores.
        """
        ...
