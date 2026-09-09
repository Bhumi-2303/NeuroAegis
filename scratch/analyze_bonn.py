#!/usr/bin/env python3
"""Comprehensive analysis of the Bonn University EEG Dataset."""

import os
import glob
import pandas as pd
import numpy as np
from collections import Counter

RAW_DIR = "/Volumes/BLACK-BOX/NeuroAegis/data/bonn_raw"
FEATURES_CSV = "/Volumes/BLACK-BOX/NeuroAegis/data/bonn_features.csv"
SETS = ["Z", "O", "N", "F", "S"]

sep = "=" * 70

# ──────────────────────────────────────────────────────────────────────
# 1. Count files in each subfolder
# ──────────────────────────────────────────────────────────────────────
print(sep)
print("1. FILE COUNTS PER SUBFOLDER")
print(sep)
total_files = 0
for s in SETS:
    folder = os.path.join(RAW_DIR, s)
    if os.path.isdir(folder):
        # list only actual files, skip ._ macOS resource forks
        files = [f for f in os.listdir(folder)
                 if os.path.isfile(os.path.join(folder, f)) and not f.startswith("._")]
        count = len(files)
        total_files += count
        print(f"  Set {s}: {count} files")
    else:
        print(f"  Set {s}: DIRECTORY NOT FOUND")
print(f"  TOTAL: {total_files} files")

# ──────────────────────────────────────────────────────────────────────
# 2. File formats and sizes
# ──────────────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("2. FILE FORMATS AND SIZES")
print(sep)
for s in SETS:
    folder = os.path.join(RAW_DIR, s)
    if not os.path.isdir(folder):
        continue
    files = [f for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder, f)) and not f.startswith("._")]
    extensions = Counter(os.path.splitext(f)[1] for f in files)
    sizes = [os.path.getsize(os.path.join(folder, f)) for f in files]
    print(f"  Set {s}:")
    print(f"    Extensions: {dict(extensions)}")
    if sizes:
        print(f"    Size range: {min(sizes)} – {max(sizes)} bytes")
        print(f"    Mean size:  {np.mean(sizes):.1f} bytes")
    # Show first 5 filenames
    print(f"    Sample filenames: {sorted(files)[:5]}")

# ──────────────────────────────────────────────────────────────────────
# 3. Feature CSV analysis
# ──────────────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("3. FEATURE CSV ANALYSIS")
print(sep)

df = pd.read_csv(FEATURES_CSV)

print(f"  Total rows:    {len(df)}")
print(f"  Total columns: {len(df.columns)}")
print(f"\n  Column names ({len(df.columns)}):")
for i, col in enumerate(df.columns):
    print(f"    [{i:>2}] {col}  (dtype={df[col].dtype})")

# 3a. Label distribution
print(f"\n  --- Label distribution (column 'label') ---")
if "label" in df.columns:
    label_counts = df["label"].value_counts().sort_index()
    for lbl, cnt in label_counts.items():
        print(f"    {lbl}: {cnt}  ({cnt/len(df)*100:.1f}%)")
    print(f"    Unique labels: {df['label'].nunique()}")
else:
    print("    WARNING: 'label' column not found!")

# 3b. Set distribution
print(f"\n  --- Set distribution (column 'set') ---")
if "set" in df.columns:
    set_counts = df["set"].value_counts().sort_index()
    for s_val, cnt in set_counts.items():
        print(f"    {s_val}: {cnt}  ({cnt/len(df)*100:.1f}%)")
    print(f"    Unique sets: {df['set'].nunique()}")
else:
    print("    WARNING: 'set' column not found!")

# 3c. Cross-tabulation label × set
print(f"\n  --- Cross-tabulation: label × set ---")
if "label" in df.columns and "set" in df.columns:
    ct = pd.crosstab(df["label"], df["set"], margins=True)
    print(ct.to_string(index=True).replace("\n", "\n    "))
else:
    print("    Cannot compute (missing label/set column)")

# 3d. Missing values
print(f"\n  --- Missing values ---")
missing = df.isnull().sum()
total_missing = missing.sum()
print(f"    Total missing values: {total_missing}")
if total_missing > 0:
    cols_with_missing = missing[missing > 0]
    for col, cnt in cols_with_missing.items():
        print(f"    {col}: {cnt} missing ({cnt/len(df)*100:.2f}%)")
else:
    print("    No missing values found ✓")

# 3e. Duplicate rows
print(f"\n  --- Duplicate rows ---")
n_dupes = df.duplicated().sum()
print(f"    Exact duplicate rows: {n_dupes}")
if n_dupes > 0:
    print("    WARNING: duplicates detected!")
else:
    print("    No duplicates ✓")

# 3f. Basic statistics of features
print(f"\n  --- Basic statistics of numeric features ---")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"    Number of numeric columns: {len(numeric_cols)}")
desc = df[numeric_cols].describe().T
# Show key stats
print(f"\n    {'Feature':<30} {'count':>8} {'mean':>12} {'std':>12} {'min':>12} {'max':>12}")
print("    " + "-" * 86)
for col in desc.index:
    row = desc.loc[col]
    print(f"    {col:<30} {row['count']:>8.0f} {row['mean']:>12.4f} {row['std']:>12.4f} {row['min']:>12.4f} {row['max']:>12.4f}")

# 3g. epoch_id uniqueness
print(f"\n  --- epoch_id analysis ---")
if "epoch_id" in df.columns:
    n_unique = df["epoch_id"].nunique()
    n_total = len(df)
    print(f"    Total epoch_ids:  {n_total}")
    print(f"    Unique epoch_ids: {n_unique}")
    if n_unique == n_total:
        print("    All epoch_ids are unique ✓")
    else:
        n_dup = n_total - n_unique
        print(f"    WARNING: {n_dup} duplicate epoch_ids found!")
        dup_ids = df[df["epoch_id"].duplicated(keep=False)]["epoch_id"].value_counts().head(10)
        print("    Top duplicated epoch_ids:")
        for eid, cnt in dup_ids.items():
            print(f"      {eid}: appears {cnt} times")
    # Show sample epoch_ids
    print(f"    Sample epoch_ids: {df['epoch_id'].head(5).tolist()}")
else:
    print("    WARNING: 'epoch_id' column not found!")

# ──────────────────────────────────────────────────────────────────────
# 4. Verify raw text files – sample from each set
# ──────────────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("4. RAW TEXT FILE VERIFICATION (sample from each set)")
print(sep)
for s in SETS:
    folder = os.path.join(RAW_DIR, s)
    if not os.path.isdir(folder):
        continue
    files = sorted([f for f in os.listdir(folder)
                    if os.path.isfile(os.path.join(folder, f)) and not f.startswith("._")])
    if not files:
        print(f"  Set {s}: NO FILES")
        continue

    # Check data-point count consistency across ALL files
    point_counts = {}
    for f in files:
        fpath = os.path.join(folder, f)
        try:
            with open(fpath, "r") as fh:
                lines = fh.readlines()
            # Count non-empty lines (each line is one data point)
            n_points = len([l for l in lines if l.strip()])
            point_counts[f] = n_points
        except Exception as e:
            point_counts[f] = f"ERROR: {e}"

    counts = Counter(v for v in point_counts.values() if isinstance(v, int))
    print(f"  Set {s}:")
    print(f"    Files checked: {len(files)}")
    print(f"    Data-point counts: {dict(counts)}")

    # Show sample of first file
    sample_file = files[0]
    sample_path = os.path.join(folder, sample_file)
    with open(sample_path, "r") as fh:
        sample_lines = fh.readlines()
    print(f"    Sample file: {sample_file}")
    print(f"    First 5 values: {[l.strip() for l in sample_lines[:5]]}")

# ──────────────────────────────────────────────────────────────────────
# 5. Check for corrupted or empty files
# ──────────────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("5. CORRUPTED / EMPTY FILE CHECK")
print(sep)
empty_files = []
error_files = []
non_numeric_files = []

for s in SETS:
    folder = os.path.join(RAW_DIR, s)
    if not os.path.isdir(folder):
        continue
    files = [f for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder, f)) and not f.startswith("._")]
    for f in files:
        fpath = os.path.join(folder, f)
        size = os.path.getsize(fpath)
        if size == 0:
            empty_files.append(f"{s}/{f}")
            continue
        try:
            with open(fpath, "r") as fh:
                content = fh.read()
            values = content.strip().split()
            # Check if all values are numeric
            for v in values:
                try:
                    float(v)
                except ValueError:
                    non_numeric_files.append((f"{s}/{f}", v))
                    break
        except Exception as e:
            error_files.append((f"{s}/{f}", str(e)))

print(f"  Empty files:       {len(empty_files)}")
if empty_files:
    for ef in empty_files:
        print(f"    - {ef}")
else:
    print("    None ✓")

print(f"  Unreadable files:  {len(error_files)}")
if error_files:
    for ef, err in error_files:
        print(f"    - {ef}: {err}")
else:
    print("    None ✓")

print(f"  Non-numeric files: {len(non_numeric_files)}")
if non_numeric_files:
    for nf, val in non_numeric_files:
        print(f"    - {nf}: found non-numeric value '{val}'")
else:
    print("    None ✓")

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("SUMMARY")
print(sep)
print(f"  Raw data: {total_files} files across {len(SETS)} sets (Z, O, N, F, S)")
print(f"  Feature CSV: {len(df)} rows × {len(df.columns)} columns")
print(f"  Missing values: {total_missing}")
print(f"  Duplicate rows: {n_dupes}")
print(f"  Empty raw files: {len(empty_files)}")
print(f"  Corrupted raw files: {len(error_files) + len(non_numeric_files)}")
print(f"\nAnalysis complete.")
