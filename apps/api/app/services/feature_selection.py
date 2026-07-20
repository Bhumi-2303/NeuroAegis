import numpy as np
from typing import List, Dict

def select_and_order_features(extracted_features: Dict[str, float], selected_features: List[str]) -> np.ndarray:
    """
    STRICT VALIDATION:
    - Verifies feature count matches exactly.
    - Verifies no missing features.
    - Verifies no extra features.
    - Preserves exact ordering from selected_features.json.
    """
    extracted_keys = set(extracted_features.keys())
    selected_keys = set(selected_features)
    
    # 1. Verify no missing features
    missing_features = selected_keys - extracted_keys
    if missing_features:
        raise ValueError(f"Feature Validation Error: Missing {len(missing_features)} required features: {missing_features}")
        
    # 2. Verify no extra features
    extra_features = extracted_keys - selected_keys
    if extra_features:
        raise ValueError(f"Feature Validation Error: Found {len(extra_features)} extra features not expected by the model: {extra_features}")
        
    # 3. Verify counts match (Redundant but explicit)
    if len(extracted_features) != len(selected_features):
        raise ValueError(f"Feature Validation Error: Count mismatch. Expected {len(selected_features)}, got {len(extracted_features)}.")
    
    # 4. Extract in exact order
    feature_vector = [extracted_features[feature_name] for feature_name in selected_features]
        
    return np.array(feature_vector).reshape(1, -1)
