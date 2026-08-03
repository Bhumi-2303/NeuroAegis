import numpy as np
import pandas as pd
from apps.api.app.services.preprocessing import wavelet_denoise
from apps.api.app.services.feature_extraction import extract_features

def test_parity():
    print("Loading data...")
    raw_data = np.load('NeuroAegis_Complete_Project/features/bonn_raw_dataset.npz')
    signals = raw_data['signals']
    
    exported_df = pd.read_csv('NeuroAegis_Complete_Project/features/features.csv')
    
    print("Testing random samples for parity...")
    sample_indices = [0, 100, 500, 1000, -1]
    
    for idx in sample_indices:
        signal = signals[idx]
        exported_features = exported_df.iloc[idx].to_dict()
        
        # Test just the features on raw vs denoised
        denoised_signal = wavelet_denoise(signal, wavelet="coif3", level=4)
        backend_features = extract_features(denoised_signal)
        
        mismatches = 0
        for feat_name, exported_val in exported_features.items():
            if feat_name not in backend_features:
                if feat_name != 'label':
                    print(f"Missing feature in backend: {feat_name}")
                continue
                
            backend_val = backend_features[feat_name]
            
            if not np.isclose(backend_val, exported_val, rtol=1e-5, atol=1e-8):
                print(f"MISMATCH at idx {idx}, feature '{feat_name}':\nBackend  : {backend_val}\nExported : {exported_val}\n")
                mismatches += 1
        if mismatches > 0:
            print(f"Found {mismatches} mismatches at idx {idx}")
            return
            
    print("SUCCESS: Numerical parity verified!")

if __name__ == "__main__":
    test_parity()
