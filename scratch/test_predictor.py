import mne
import sys
import numpy as np
from app.services.prediction.chbmit_predictor import CHBMITPredictor

predictor = CHBMITPredictor(model_dir="apps/api/models/chbmit", default_model="lightgbm")
predictor.load_model()

raw = mne.io.read_raw_edf("data/chbmit_subset/chb01/chb01_01.edf", preload=True, verbose=False)
df = raw.to_data_frame()
df.drop(columns=["time"], inplace=True, errors="ignore")
eeg_data = df.values.T

start_sample = 433664
eeg_data = eeg_data[:, start_sample:start_sample + 15360]

print("Sliced shape:", eeg_data.shape)

feature_vector, raw_features = predictor.extract_features(eeg_data, ["Ch0"], 256.0)

print("raw_features keys:", list(raw_features.keys())[:5])
print("vector zeros:", list(feature_vector[0]).count(0.0))
print("Ch0_line_length in raw:", raw_features.get("Ch0_line_length"))

pred = predictor.predict(feature_vector)
print("Prediction:", pred)

expl = predictor.generate_explanation(feature_vector, raw_features)
print("Ch0_line_length raw in expl:", expl["features"][0].get("rawValue"))

