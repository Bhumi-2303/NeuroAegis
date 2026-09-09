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

print("Initializing PatientDataSplitter...")
splitter = PatientDataSplitter(label_column="label_50pct_overlap")
train_df, val_df, test_df = splitter.get_splits()
print(f"Validation split: {len(val_df):,} windows across {val_df['recording_id'].nunique()} recordings")

cache_mgr = EmbeddingCacheManager()
t0 = time.time()
val_embs = cache_mgr.get_split_embeddings(
    val_df,
    split_name="val",
    progress_callback=lambda cur, tot, r: print(f"[{cur}/{tot}] Cached {r} ({time.time()-t0:.1f}s elapsed)", flush=True) if cur % 10 == 0 or cur == tot else None
)
t1 = time.time()
print(f"DONE: Cached all validation embeddings in {t1-t0:.1f}s. Shape: {val_embs.shape}")
