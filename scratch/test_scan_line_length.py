import sys
import numpy as np
import mne
import warnings
from app.services.prediction.chbmit_predictor import CHBMITPredictor

warnings.filterwarnings('ignore')

filepath = "data/chbmit_subset/chb01/chb01_01.edf"
raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
df = raw.to_data_frame(scalings=dict(eeg=1, eog=1, ecg=1, emg=1, misc=1))
if 'time' in df.columns:
    df = df.drop('time', axis=1)

eeg_data = df.values.T 
channel_names = df.columns.tolist()

fs = 256.0
window_length = 15360

predictor = CHBMITPredictor("apps/api/models/chbmit", "lightgbm")
if not predictor.load_model():
    print("Failed to load model")
    exit(1)

print("Scanning file in 60s windows...")
line_lengths = []
probs = []
for i in range(0, eeg_data.shape[1] - window_length, window_length):
    bg_data = eeg_data[:, i:i + window_length]
    features, raw_feats = predictor.extract_features(bg_data, channel_names, fs)
    res = predictor.predict(features)
    prob = res['probabilities']['seizure']
    ll = raw_feats['Ch0_line_length']
    line_lengths.append(ll)
    probs.append(prob)

print(f"Total windows: {len(line_lengths)}")
print(f"Mean line length: {np.mean(line_lengths):.4f}")
print(f"Min line length: {np.min(line_lengths):.4f}")
print(f"Max line length: {np.max(line_lengths):.4f}")
print(f"Mean probability: {np.mean(probs):.4f}")
print(f"Min probability: {np.min(probs):.4f}")
print(f"Max probability: {np.max(probs):.4f}")
