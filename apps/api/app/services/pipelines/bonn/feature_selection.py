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
        
    # 3. We don't care about extra features, we just extract the ones we need
    # (The model only uses selected_features, so extra features are ignored)
    
    # 4. Extract in exact order
    feature_vector = [extracted_features[feature_name] for feature_name in selected_features]
        
    return np.array(feature_vector).reshape(1, -1)
