import numpy as np
import json
from apps.api.app.services.preprocessing import wavelet_denoise
from apps.api.app.services.feature_extraction import extract_features

def test_parity():
    raw_data = np.load('NeuroAegis_Complete_Project/features/bonn_raw_dataset.npz')
    signals = raw_data['signals']
    
    feature_matrix = np.load('NeuroAegis_Complete_Project/features/feature_matrix.npz')['X']
    with open('NeuroAegis_Complete_Project/features/feature_names.json', 'r') as f:
        feature_names = json.load(f)
        
    sample_indices = [0, 100, 500, 1000, -1]
    
    for idx in sample_indices:
        signal = signals[idx]
        
        # Build expected dict
        exported_features = dict(zip(feature_names, feature_matrix[idx]))
        
        denoised_signal = wavelet_denoise(signal, wavelet="coif3", level=4)
        denoised_signal = denoised_signal.astype(np.float32)
        
        backend_features = extract_features(denoised_signal)
        
        mismatches = 0
        for feat_name, exported_val in exported_features.items():
            if feat_name not in backend_features:
                continue
            backend_val = backend_features[feat_name]
            # Since exported is float32, we cast backend to float32 before comparing
            if not np.isclose(np.float32(backend_val), exported_val, rtol=1e-5, atol=1e-7):
                print(f"MISMATCH at idx {idx}, feature '{feat_name}':\nBackend  : {backend_val} ({np.float32(backend_val)})\nExported : {exported_val}\n")
                mismatches += 1
        if mismatches > 0:
            print(f"Failed at idx {idx}")
            return
    print("Matches perfectly with float32 casting!")

if __name__ == "__main__":
    test_parity()
