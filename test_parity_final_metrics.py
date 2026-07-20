import numpy as np
import json
from apps.api.app.services.feature_extraction import extract_features

def compute_metrics():
    # Read the exact denoised signals output by the notebook's preprocessing pipeline
    denoised_data = np.load('NeuroAegis_Complete_Project/features/coif3_denoised_dataset.npz')
    signals = denoised_data['signals']
    
    feature_matrix = np.load('NeuroAegis_Complete_Project/features/feature_matrix.npz')['X']
    with open('NeuroAegis_Complete_Project/features/feature_names.json', 'r') as f:
        feature_names = json.load(f)
        
    # Test all 11500 samples
    total_samples = len(signals)
    sample_indices = np.arange(total_samples)
    
    rtol = 1e-5
    atol = 1e-7
    
    max_abs_error = 0.0
    max_rel_error = 0.0
    mismatched_features_count = 0
    total_features_compared = 0
    
    print(f"Testing parity for all {total_samples} samples...")
    for idx in sample_indices:
        signal = signals[idx]
        exported_features = dict(zip(feature_names, feature_matrix[idx]))
        
        # Backend feature extraction
        backend_features = extract_features(signal)
        
        for feat_name, exported_val in exported_features.items():
            if feat_name not in backend_features:
                continue
            backend_val = np.float32(backend_features[feat_name])
            total_features_compared += 1
            
            abs_err = np.abs(backend_val - exported_val)
            if exported_val != 0:
                rel_err = abs_err / np.abs(exported_val)
            else:
                rel_err = abs_err
                
            if abs_err > max_abs_error:
                max_abs_error = abs_err
            if rel_err > max_rel_error:
                max_rel_error = rel_err
                
            if not np.isclose(backend_val, exported_val, rtol=rtol, atol=atol):
                mismatched_features_count += 1
                
    print("--- PARITY TEST METRICS ---")
    print(f"Number of samples tested: {len(sample_indices)}")
    print(f"Number of features compared: {total_features_compared}")
    print(f"rtol value: {rtol}")
    print(f"atol value: {atol}")
    print(f"Maximum absolute error: {max_abs_error:.12f}")
    print(f"Maximum relative error: {max_rel_error:.12f}")
    print(f"Number of mismatched features: {mismatched_features_count}")
    print("---------------------------")

if __name__ == "__main__":
    compute_metrics()
