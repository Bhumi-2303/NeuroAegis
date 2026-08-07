import sys
import numpy as np
import mne
import warnings
import json
from app.services.prediction.chbmit_predictor import CHBMITPredictor

warnings.filterwarnings('ignore')

# Load the file
filepath = "data/chbmit_subset/chb01/chb01_01.edf"
raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
df = raw.to_data_frame(scalings=dict(eeg=1, eog=1, ecg=1, emg=1, misc=1))
if 'time' in df.columns:
    df = df.drop('time', axis=1)

eeg_data = df.values.T 
channel_names = df.columns.tolist()

# The seizure in chb06_01 is at 1724s.
# Let's test a background window far away, e.g. at 500s.
fs = 256.0
window_length = 15360

print("Testing background window at 500s:")
start_sample = int(500 * fs)
bg_data = eeg_data[:, start_sample:start_sample + window_length]

predictor = CHBMITPredictor("apps/api/models/chbmit", "lightgbm")
if not predictor.load_model():
    print("Failed to load model")
    exit(1)
    
features, raw_feats = predictor.extract_features(bg_data, channel_names, fs)
res = predictor.predict(features)
print(f"Background Prediction: {res['label']}")
print(f"Probabilities: {res['probabilities']}")
expl = predictor.generate_explanation(features, raw_feats)
top = sorted(expl['features'], key=lambda x: abs(x['value']), reverse=True)[:3]
for t in top:
    print(f"  {t['featureName']}: SHAP={t['value']:.4f}, Raw={t['rawValue']:.4f}")
    
print("\nTesting seizure window at 1724s:")
start_sample = int(1724 * fs) - (window_length // 2)
seiz_data = eeg_data[:, start_sample:start_sample + window_length]
features, raw_feats = predictor.extract_features(seiz_data, channel_names, fs)
res = predictor.predict(features)
print(f"Seizure Prediction: {res['label']}")
print(f"Probabilities: {res['probabilities']}")
expl = predictor.generate_explanation(features, raw_feats)
top = sorted(expl['features'], key=lambda x: abs(x['value']), reverse=True)[:3]
for t in top:
    print(f"  {t['featureName']}: SHAP={t['value']:.4f}, Raw={t['rawValue']:.4f}")
