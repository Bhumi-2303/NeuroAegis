import pandas as pd
import numpy as np

# Load test predictions
df = pd.read_csv("research/results/phase_4b/final_test_predictions.csv")
df.sort_values(by=["patient_id", "recording_id", "window_start_sec"], inplace=True)

# Generate alarms (simple thresholding first)
threshold = 0.5
df["positive"] = df["predicted_probability"] >= threshold

# Event logic: continuous positive windows = alarm?
# Let's count FA/day
total_hours = 152.82
# ... Need to know existing alert protocol.
