import pandas as pd
import json

print("Calculating for CHB-MIT...")
chb_df = pd.read_parquet("chbmit_subset.parquet")
chb_non_seizure = chb_df[chb_df['target'] == 0]
chb_non_seizure_numeric = chb_non_seizure.select_dtypes(include=['number'])

chb_ranges = {}
for col in chb_non_seizure_numeric.columns:
    if col != 'target':
        chb_ranges[col] = [
            float(chb_non_seizure_numeric[col].quantile(0.05)),
            float(chb_non_seizure_numeric[col].quantile(0.95))
        ]

with open("apps/api/models/chbmit/reference_ranges.json", "w") as f:
    json.dump(chb_ranges, f, indent=2)

print("Calculating for Bonn...")
bonn_df = pd.read_csv("NeuroAegis_bonn_dataset_model/features/features.csv")
bonn_non_seizure = bonn_df[bonn_df['binary_label'] == 0]
bonn_non_seizure_numeric = bonn_non_seizure.select_dtypes(include=['number'])

bonn_ranges = {}
for col in bonn_non_seizure_numeric.columns:
    if col not in ['file', 'class', 'label', 'label_name', 'binary_label']:
        bonn_ranges[col] = [
            float(bonn_non_seizure_numeric[col].quantile(0.05)),
            float(bonn_non_seizure_numeric[col].quantile(0.95))
        ]

with open("apps/api/models/bonn/reference_ranges.json", "w") as f:
    json.dump(bonn_ranges, f, indent=2)

print("Done!")
