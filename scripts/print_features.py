import numpy as np
import pandas as pd
from apps.api.app.services.preprocessing import wavelet_denoise
from apps.api.app.services.feature_extraction import extract_features

raw_data = np.load('NeuroAegis_Complete_Project/features/bonn_raw_dataset.npz')
signals = raw_data['signals']
exported_df = pd.read_csv('NeuroAegis_Complete_Project/features/features.csv')

idx = 0
signal = signals[idx]
exported_features = exported_df.iloc[idx].to_dict()
denoised_signal = wavelet_denoise(signal, wavelet="coif3", level=4)
backend_features = extract_features(denoised_signal)

for k in sorted(backend_features.keys()):
    print(f"{k}: Backend={backend_features[k]} | Exported={exported_features.get(k)}")
