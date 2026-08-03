import numpy as np
import json
from apps.api.app.services.feature_extraction import extract_features

denoised_data = np.load('NeuroAegis_Complete_Project/features/coif3_denoised_dataset.npz')
signals = denoised_data['signals']

feature_matrix = np.load('NeuroAegis_Complete_Project/features/feature_matrix.npz')['X']
with open('NeuroAegis_Complete_Project/features/feature_names.json', 'r') as f:
    feature_names = json.load(f)
    
rtol = 1e-5
atol = 1e-7

for idx in range(len(signals)):
    signal = signals[idx]
    exported_features = dict(zip(feature_names, feature_matrix[idx]))
    backend_features = extract_features(signal)
    
    for feat_name, exported_val in exported_features.items():
        if feat_name not in backend_features:
            continue
        backend_val = np.float32(backend_features[feat_name])
        
        if not np.isclose(backend_val, exported_val, rtol=rtol, atol=atol):
            abs_err = np.abs(backend_val - exported_val)
            rel_err = abs_err / np.abs(exported_val) if exported_val != 0 else abs_err
            print(f"Sample {idx}, Feature '{feat_name}': Backend={backend_val}, Exported={exported_val}, abs_err={abs_err}, rel_err={rel_err}")
