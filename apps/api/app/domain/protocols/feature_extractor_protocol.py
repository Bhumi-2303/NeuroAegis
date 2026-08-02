from typing import Protocol, runtime_checkable, Dict, Any, Optional
import numpy as np
from .plugin_protocol import PluginProtocol

@runtime_checkable
class FeatureExtractorProtocol(PluginProtocol, Protocol):
    """
    Protocol for feature extraction plugins.
    Converts raw signal data into mathematical feature vectors suitable for inference.
    """
    
    def extract_features(self, data: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Extract features from the raw signal.
        
        Args:
            data (np.ndarray): The raw signal data.
            metadata (Optional[Dict[str, Any]]): Additional recording metadata (e.g., sampling rate).
            
        Returns:
            np.ndarray: The extracted feature vector.
        """
        ...
