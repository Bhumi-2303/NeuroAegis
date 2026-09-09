import sys
import os
import time

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import pandas as pd
import numpy as np

from research.imbalance.patient_splitter import PatientDataSplitter
from research.phase_4b.embedding_cache import EmbeddingCacheManager

print("Initializing PatientDataSplitter for training set...")
splitter = PatientDataSplitter(label_column="label_50pct_overlap")
train_df, val_df, test_df = splitter.get_splits()
print(f"Training split: {len(train_df):,} windows across {train_df['recording_id'].nunique()} recordings")

cache_mgr = EmbeddingCacheManager()
t0 = time.time()
train_embs = cache_mgr.get_split_embeddings(
    train_df,
    split_name="train",
    progress_callback=lambda cur, tot, r: print(f"[{cur}/{tot}] Cached {r} ({time.time()-t0:.1f}s elapsed)", flush=True) if cur % 20 == 0 or cur == tot else None
)
t1 = time.time()
print(f"DONE: Cached all training embeddings in {t1-t0:.1f}s. Shape: {train_embs.shape}")
