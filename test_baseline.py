import pandas as pd
import numpy as np

# Load predictions
df = pd.read_csv("research/results/phase_4b/final_test_predictions.csv")
threshold = 0.5
df["pred"] = (df["predicted_probability"] >= threshold).astype(int)

# True labels
df["label"] = df["label_50pct_overlap"].astype(int)

# Calculate metrics
tp = ((df["label"] == 1) & (df["pred"] == 1)).sum()
fp = ((df["label"] == 0) & (df["pred"] == 1)).sum()
tn = ((df["label"] == 0) & (df["pred"] == 0)).sum()
fn = ((df["label"] == 1) & (df["pred"] == 0)).sum()

print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
total_hours = len(df) * 2.5 / 3600
fa_per_24h = (fp / total_hours) * 24

print(f"FA/24h: {fa_per_24h}")

# Event sensitivity
# We don't have seizure_event_ids in this csv. Let's see how events are defined.
# If they are just contiguous blocks of label==1.
events = []
in_event = False
event_windows = []
for idx, row in df.iterrows():
    if row["label"] == 1:
        event_windows.append(row["pred"])
        in_event = True
    else:
        if in_event:
            events.append(event_windows)
            event_windows = []
            in_event = False
if in_event:
    events.append(event_windows)

detected = 0
for e in events:
    if sum(e) > 0:
        detected += 1

print(f"Total events: {len(events)}")
print(f"Detected: {detected}")
if len(events) > 0:
    print(f"Sensitivity: {detected / len(events) * 100:.2f}%")
