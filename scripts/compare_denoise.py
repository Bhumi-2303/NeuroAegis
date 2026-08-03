import numpy as np
from apps.api.app.services.preprocessing import wavelet_denoise

raw_data = np.load('NeuroAegis_Complete_Project/features/bonn_raw_dataset.npz')
signals = raw_data['signals']

denoised_data = np.load('NeuroAegis_Complete_Project/features/coif3_denoised_dataset.npz')
notebook_denoised = denoised_data['signals']

idx = 0
signal = signals[idx]
my_denoised = wavelet_denoise(signal, wavelet="coif3", level=4)
my_denoised_f32 = my_denoised.astype(np.float32)

notebook_signal = notebook_denoised[idx]

print(f"Max diff (f64): {np.max(np.abs(my_denoised - notebook_signal))}")
print(f"Max diff (f32): {np.max(np.abs(my_denoised_f32 - notebook_signal))}")

diff = np.abs(my_denoised_f32 - notebook_signal)
print(f"Number of mismatches > 1e-6: {np.sum(diff > 1e-6)}")

