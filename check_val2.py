import pandas as pd
from research.experiments.imbalance.patient_splitter import PatientDataSplitter

splitter = PatientDataSplitter(
    window_index_path="data/manifests/chbmit_window_index.csv.gz",
    seizure_events_path="data/manifests/chbmit_seizure_events.csv"
)
_, val_df, _ = splitter.get_splits()
print("val_df shape:", val_df.shape)
