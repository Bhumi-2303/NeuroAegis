import pandas as pd
from research.experiments.model_c.sequence_dataset import SequenceBuilder
from research.experiments.imbalance.patient_splitter import PatientDataSplitter
import numpy as np

splitter = PatientDataSplitter(
    window_index_path="data/manifests/chbmit_window_index.csv.gz",
    seizure_events_path="data/manifests/chbmit_seizure_events.csv"
)
_, val_df, _ = splitter.get_splits()

builder = SequenceBuilder(label_column="label_50pct_overlap")
val_seq_df = builder.build_sequences(val_df, seq_len=8)
print("val_seq_df shape:", val_seq_df.shape)

f = np.load("research/experiments/model_c/experiments/L8/val_predictions.npz")
print("y_prob shape:", f["y_prob"].shape)
