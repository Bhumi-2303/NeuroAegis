import numpy as np
import json
from apps.api.app.services.feature_extraction import extract_features

def test_parity():
    denoised_data = np.load('NeuroAegis_Complete_Project/features/coif3_denoised_dataset.npz')
    signals = denoised_data['signals']
    
    feature_matrix = np.load('NeuroAegis_Complete_Project/features/feature_matrix.npz')['X']
    with open('NeuroAegis_Complete_Project/features/feature_names.json', 'r') as f:
        feature_names = json.load(f)
        
    idx = 0
    signal = signals[idx]
    
    exported_features = dict(zip(feature_names, feature_matrix[idx]))
    
    backend_features = extract_features(signal)
    
    mismatches = 0
    for feat_name, exported_val in exported_features.items():
        if feat_name not in backend_features:
            continue
        backend_val = backend_features[feat_name]
        if not np.isclose(np.float32(backend_val), exported_val, rtol=1e-5, atol=1e-7):
            print(f"MISMATCH feature '{feat_name}': Backend={backend_val}, Exported={exported_val}")
            mismatches += 1
            
    if mismatches == 0:
        print("Feature extraction alone matches perfectly!")

if __name__ == "__main__":
    test_parity()
