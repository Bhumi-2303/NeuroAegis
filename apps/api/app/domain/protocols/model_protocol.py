from typing import Protocol, runtime_checkable, Dict, Any
import numpy as np
from .plugin_protocol import PluginProtocol

@runtime_checkable
class ModelProtocol(PluginProtocol, Protocol):
    """
    Protocol for machine learning inference plugins.
    Handles loading weights, resource management, and making predictions.
    """
    
    def load(self, model_path: str) -> None:
        """
        Load model weights into memory.
        
        Args:
            model_path (str): Path to the model artifacts.
        """
        ...
        
    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Perform inference on extracted features.
        
        Args:
            features (np.ndarray): Extracted feature vector.
            
        Returns:
            Dict[str, float]: Mapping of class labels to probability scores summing to 1.0.
        """
        ...
        
    def unload(self) -> None:
        """
        Free model resources from memory (e.g., clear GPU cache).
        """
        ...
