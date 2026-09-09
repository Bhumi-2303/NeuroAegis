#!/usr/bin/env python3
"""
Comprehensive CHB-MIT Scalp EEG Dataset Analysis
Analyzes three dataset locations, parquet files, seizure annotations, and demographics.
"""

import os
import re
import glob
import json
import sys
from collections import defaultdict, OrderedDict

import pandas as pd

# ============================================================
# PATHS
# ============================================================
BASE = "/Volumes/BLACK-BOX/NeuroAegis"
ORIGINAL_DIR = os.path.join(BASE, "CHB-MIT Dataset")
EDF_DIR = os.path.join(BASE, "data", "chbmit_edf")
SUBSET_DIR = os.path.join(BASE, "data", "chbmit_subset")

PARQUET_FILES = [
    os.path.join(BASE, "data", "chbmit_multichannel.parquet"),
    os.path.join(BASE, "data", "chbmit_multichannel_full.parquet"),
    os.path.join(BASE, "chbmit_subset.parquet"),
]

OUTPUT_FILE = os.path.join(BASE, "scratch", "chbmit_analysis_report.txt")

# Redirect output to both file and stdout
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

report_file = open(OUTPUT_FILE, "w")
sys.stdout = Tee(sys.__stdout__, report_file)

PATIENTS = [f"chb{i:02d}" for i in range(1, 25)]

def separator(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

# ============================================================
# 1. COUNT EDF FILES PER PATIENT
# ============================================================
separator("1. EDF FILE COUNTS PER PATIENT")

def count_edf_per_patient(base_dir, label):
    """Count .edf files per patient folder in a directory."""
    print(f"\n--- {label}: {base_dir} ---")
    patient_counts = OrderedDict()
    total = 0
    for pat in PATIENTS:
        pat_dir = os.path.join(base_dir, pat)
        if os.path.isdir(pat_dir):
            edfs = glob.glob(os.path.join(pat_dir, "*.edf"))
            # Exclude macOS resource fork files
            edfs = [f for f in edfs if not os.path.basename(f).startswith("._")]
            count = len(edfs)
            patient_counts[pat] = count
            total += count
        else:
            patient_counts[pat] = "DIR NOT FOUND"
    
    for pat, cnt in patient_counts.items():
        print(f"  {pat}: {cnt} EDF files")
    print(f"\n  TOTAL: {total} EDF files across {sum(1 for v in patient_counts.values() if isinstance(v, int) and v > 0)} patients")
    return patient_counts

orig_counts = count_edf_per_patient(ORIGINAL_DIR, "CHB-MIT Dataset (Original)")
edf_counts = count_edf_per_patient(EDF_DIR, "data/chbmit_edf")

# ============================================================
# 2. PARSE SEIZURE SUMMARY FILES
# ============================================================
separator("2. SEIZURE ANNOTATIONS FROM SUMMARY FILES")

def parse_summary_file(filepath):
    """Parse a chbXX-summary.txt file to extract seizure info per file."""
    with open(filepath, "r") as f:
        content = f.read()
    
    results = []
    current_file = None
    current_seizures = []
    num_seizures_in_file = 0
    sampling_rate = None
    channels = []
    
    # Extract sampling rate
    sr_match = re.search(r'Data Sampling Rate:\s*(\d+)\s*Hz', content)
    if sr_match:
        sampling_rate = int(sr_match.group(1))
    
    # Extract channels
    for ch_match in re.finditer(r'Channel\s+\d+:\s*(.+)', content):
        ch_name = ch_match.group(1).strip()
        if ch_name not in channels:
            channels.append(ch_name)
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        
        # Match file name
        file_match = re.match(r'File Name:\s*(.+\.edf)', line)
        if file_match:
            # Save previous file info
            if current_file is not None:
                results.append({
                    'filename': current_file,
                    'num_seizures': num_seizures_in_file,
                    'seizures': current_seizures
                })
            current_file = file_match.group(1)
            current_seizures = []
            num_seizures_in_file = 0
            continue
        
        # Match number of seizures
        nsz_match = re.match(r'Number of Seizures in File:\s*(\d+)', line)
        if nsz_match:
            num_seizures_in_file = int(nsz_match.group(1))
            continue
        
        # Match seizure start time - handles both "Seizure Start Time" and "Seizure N Start Time"
        start_match = re.match(r'Seizure\s*\d*\s*Start Time:\s*(\d+)\s*seconds', line)
        if start_match:
            start_sec = int(start_match.group(1))
            current_seizures.append({'start': start_sec, 'end': None})
            continue
        
        # Match seizure end time
        end_match = re.match(r'Seizure\s*\d*\s*End Time:\s*(\d+)\s*seconds', line)
        if end_match:
            end_sec = int(end_match.group(1))
            if current_seizures and current_seizures[-1]['end'] is None:
                current_seizures[-1]['end'] = end_sec
            continue
    
    # Save last file
    if current_file is not None:
        results.append({
            'filename': current_file,
            'num_seizures': num_seizures_in_file,
            'seizures': current_seizures
        })
    
    return results, sampling_rate, channels

all_patient_seizure_data = OrderedDict()
total_seizures_all = 0
total_files_with_seizures = 0
total_files_without_seizures = 0
total_seizure_duration_all = 0

for pat in PATIENTS:
    summary_path = os.path.join(ORIGINAL_DIR, pat, f"{pat}-summary.txt")
    if not os.path.exists(summary_path):
        print(f"  WARNING: {summary_path} NOT FOUND")
        continue
    
    file_results, srate, channels = parse_summary_file(summary_path)
    
    num_seizures = 0
    files_with_sz = 0
    files_without_sz = 0
    total_duration = 0
    seizure_details = []
    
    for fr in file_results:
        if fr['num_seizures'] > 0:
            files_with_sz += 1
            for sz in fr['seizures']:
                num_seizures += 1
                if sz['end'] is not None:
                    dur = sz['end'] - sz['start']
                    total_duration += dur
                    seizure_details.append({
                        'file': fr['filename'],
                        'start': sz['start'],
                        'end': sz['end'],
                        'duration': dur
                    })
        else:
            files_without_sz += 1
    
    all_patient_seizure_data[pat] = {
        'sampling_rate': srate,
        'num_channels': len(channels),
        'channels': channels,
        'total_files_in_summary': len(file_results),
        'files_with_seizures': files_with_sz,
        'files_without_seizures': files_without_sz,
        'total_seizures': num_seizures,
        'total_seizure_duration_sec': total_duration,
        'seizure_details': seizure_details,
    }
    
    total_seizures_all += num_seizures
    total_files_with_seizures += files_with_sz
    total_files_without_seizures += files_without_sz
    total_seizure_duration_all += total_duration

# Print per-patient summary
print(f"{'Patient':<10} {'SRate':<8} {'#Ch':<6} {'#Files':<8} {'w/Sz':<8} {'wo/Sz':<8} {'#Seizures':<12} {'Tot Duration (s)':<18}")
print("-" * 90)
for pat, data in all_patient_seizure_data.items():
    print(f"{pat:<10} {data['sampling_rate']:<8} {data['num_channels']:<6} {data['total_files_in_summary']:<8} "
          f"{data['files_with_seizures']:<8} {data['files_without_seizures']:<8} {data['total_seizures']:<12} "
          f"{data['total_seizure_duration_sec']:<18}")

print(f"\n--- GRAND TOTALS ---")
print(f"  Total seizures across all patients: {total_seizures_all}")
print(f"  Total files with seizures: {total_files_with_seizures}")
print(f"  Total files without seizures: {total_files_without_seizures}")
print(f"  Total seizure duration: {total_seizure_duration_all} seconds ({total_seizure_duration_all/60:.1f} minutes)")

# Print detailed seizure list per patient
separator("2b. DETAILED SEIZURE LIST PER PATIENT")
for pat, data in all_patient_seizure_data.items():
    if data['seizure_details']:
        print(f"\n  {pat} ({data['total_seizures']} seizures, total {data['total_seizure_duration_sec']}s):")
        for i, sz in enumerate(data['seizure_details'], 1):
            print(f"    Seizure {i}: {sz['file']} | Start={sz['start']}s, End={sz['end']}s, Duration={sz['duration']}s")

# ============================================================
# 3. PARSE RECORDS AND RECORDS-WITH-SEIZURES FOR CROSS-VALIDATION
# ============================================================
separator("3. CROSS-VALIDATION: RECORDS vs RECORDS-WITH-SEIZURES vs SUMMARIES")

def parse_records_file(filepath):
    """Parse RECORDS or RECORDS-WITH-SEIZURES file."""
    with open(filepath, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    patient_files = defaultdict(list)
    for line in lines:
        parts = line.split("/")
        if len(parts) == 2:
            patient_files[parts[0]].append(parts[1])
    return patient_files

records_path = os.path.join(ORIGINAL_DIR, "RECORDS")
records_sz_path = os.path.join(ORIGINAL_DIR, "RECORDS-WITH-SEIZURES")

records_data = parse_records_file(records_path)
records_sz_data = parse_records_file(records_sz_path)

print("--- RECORDS file ---")
print(f"  Total files listed: {sum(len(v) for v in records_data.values())}")
print(f"  Patients: {len(records_data)}")
for pat in PATIENTS:
    if pat in records_data:
        print(f"    {pat}: {len(records_data[pat])} files")

print(f"\n--- RECORDS-WITH-SEIZURES file ---")
print(f"  Total files listed: {sum(len(v) for v in records_sz_data.values())}")
print(f"  Patients: {len(records_sz_data)}")
for pat in PATIENTS:
    if pat in records_sz_data:
        print(f"    {pat}: {len(records_sz_data[pat])} files")

# Cross-validate summaries vs RECORDS-WITH-SEIZURES
print(f"\n--- Cross-Validation ---")
mismatches = []
for pat in PATIENTS:
    if pat not in all_patient_seizure_data:
        continue
    
    # Files with seizures from summary
    summary_sz_files = set()
    for detail in all_patient_seizure_data[pat]['seizure_details']:
        summary_sz_files.add(detail['file'])
    
    # Files with seizures from RECORDS-WITH-SEIZURES
    records_sz_files = set(records_sz_data.get(pat, []))
    
    if summary_sz_files != records_sz_files:
        only_in_summary = summary_sz_files - records_sz_files
        only_in_records = records_sz_files - summary_sz_files
        mismatches.append({
            'patient': pat,
            'only_in_summary': only_in_summary,
            'only_in_records': only_in_records,
        })
        print(f"  MISMATCH for {pat}:")
        if only_in_summary:
            print(f"    Only in summary: {only_in_summary}")
        if only_in_records:
            print(f"    Only in RECORDS-WITH-SEIZURES: {only_in_records}")
    else:
        pass  # Match is good

if not mismatches:
    print("  ✓ All patients: Summary seizure files match RECORDS-WITH-SEIZURES exactly")
else:
    print(f"\n  Total mismatches: {len(mismatches)} patients")

# Cross-validate RECORDS vs actual files on disk
print(f"\n--- RECORDS vs Actual Files on Disk (Original Dir) ---")
records_vs_disk_issues = []
for pat in PATIENTS:
    records_files = set(records_data.get(pat, []))
    pat_dir = os.path.join(ORIGINAL_DIR, pat)
    if os.path.isdir(pat_dir):
        disk_edfs = set(f for f in os.listdir(pat_dir) if f.endswith('.edf') and not f.startswith('._'))
        only_records = records_files - disk_edfs
        only_disk = disk_edfs - records_files
        if only_records or only_disk:
            records_vs_disk_issues.append(pat)
            print(f"  MISMATCH for {pat}:")
            if only_records:
                print(f"    In RECORDS but not on disk: {only_records}")
            if only_disk:
                print(f"    On disk but not in RECORDS: {only_disk}")

if not records_vs_disk_issues:
    print("  ✓ All RECORDS entries match actual files on disk")

# ============================================================
# 4. PARQUET FILE ANALYSIS
# ============================================================
separator("4. PARQUET FILE ANALYSIS")

for pq_path in PARQUET_FILES:
    print(f"\n--- {os.path.basename(pq_path)} ---")
    print(f"  Path: {pq_path}")
    
    if not os.path.exists(pq_path):
        print(f"  *** FILE NOT FOUND ***")
        continue
    
    file_size_mb = os.path.getsize(pq_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    try:
        df = pd.read_parquet(pq_path)
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Column names: {list(df.columns)}")
        print(f"  Dtypes:")
        for col in df.columns:
            print(f"    {col}: {df[col].dtype}")
        
        # Missing values
        print(f"\n  Missing values:")
        missing = df.isnull().sum()
        total_missing = missing.sum()
        if total_missing == 0:
            print(f"    ✓ No missing values")
        else:
            for col in df.columns:
                if missing[col] > 0:
                    print(f"    {col}: {missing[col]:,} ({missing[col]/len(df)*100:.2f}%)")
        
        # Memory usage
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"\n  In-memory size: {mem_mb:.2f} MB")
        
        # Label/class distribution
        label_cols = [c for c in df.columns if c.lower() in ('label', 'class', 'target', 'seizure', 'y', 'is_seizure')]
        if not label_cols:
            # Try partial matches
            label_cols = [c for c in df.columns if any(kw in c.lower() for kw in ('label', 'class', 'target', 'seizure'))]
        
        if label_cols:
            for lc in label_cols:
                print(f"\n  Label/Class distribution ('{lc}'):")
                vc = df[lc].value_counts()
                for val, cnt in vc.items():
                    print(f"    {val}: {cnt:,} ({cnt/len(df)*100:.2f}%)")
        else:
            print(f"\n  No obvious label/class column found among: {list(df.columns)}")
        
        # Patient distribution
        patient_cols = [c for c in df.columns if c.lower() in ('patient', 'patient_id', 'subject', 'subject_id', 'case')]
        if not patient_cols:
            patient_cols = [c for c in df.columns if any(kw in c.lower() for kw in ('patient', 'subject', 'case'))]
        
        if patient_cols:
            for pc in patient_cols:
                print(f"\n  Patient distribution ('{pc}'):")
                vc = df[pc].value_counts().sort_index()
                for val, cnt in vc.items():
                    print(f"    {val}: {cnt:,} ({cnt/len(df)*100:.2f}%)")
        else:
            print(f"\n  No obvious patient column found among: {list(df.columns)}")
        
        # First few rows sample
        print(f"\n  First 3 rows (sample):")
        print(df.head(3).to_string(max_cols=15))
        
        # Basic stats for numeric columns (first 5)
        numeric_cols = df.select_dtypes(include=['number']).columns[:5]
        if len(numeric_cols) > 0:
            print(f"\n  Basic stats (first {len(numeric_cols)} numeric columns):")
            print(df[numeric_cols].describe().to_string())
        
        del df  # Free memory
        
    except Exception as e:
        print(f"  ERROR loading parquet: {e}")

# ============================================================
# 5. SUBSET DIRECTORY ANALYSIS
# ============================================================
separator("5. SUBSET DIRECTORY ANALYSIS (data/chbmit_subset/)")

subset_counts = OrderedDict()
total_subset = 0
for pat in PATIENTS:
    pat_dir = os.path.join(SUBSET_DIR, pat)
    if os.path.isdir(pat_dir):
        all_files = [f for f in os.listdir(pat_dir) if not f.startswith('.') and not f.startswith('._')]
        edfs = [f for f in all_files if f.endswith('.edf')]
        non_edfs = [f for f in all_files if not f.endswith('.edf')]
        subset_counts[pat] = {'edf': len(edfs), 'other': len(non_edfs), 'total': len(all_files)}
        total_subset += len(edfs)
    else:
        subset_counts[pat] = None

print(f"{'Patient':<10} {'EDF':<8} {'Other':<8} {'Total':<8}")
print("-" * 34)
for pat, data in subset_counts.items():
    if data is not None:
        print(f"{pat:<10} {data['edf']:<8} {data['other']:<8} {data['total']:<8}")
    else:
        print(f"{pat:<10} DIR NOT FOUND")

print(f"\n  Total EDF files in subset: {total_subset}")
print(f"  Patients in subset: {sum(1 for v in subset_counts.values() if v is not None and v['edf'] > 0)}")

# Check for extra files in subset (RECORDS, SUBJECT-INFO, etc.)
extra_files = [f for f in os.listdir(SUBSET_DIR) if not f.startswith('.') and not f.startswith('._') and not f.startswith('chb')]
if extra_files:
    print(f"\n  Extra top-level files in subset dir: {extra_files}")

# ============================================================
# 6. CROSS-LOCATION CONSISTENCY CHECK
# ============================================================
separator("6. CROSS-LOCATION CONSISTENCY CHECK")

print("--- Original vs data/chbmit_edf (EDF counts per patient) ---")
all_match = True
for pat in PATIENTS:
    orig = orig_counts.get(pat, 0)
    edf = edf_counts.get(pat, 0)
    if isinstance(orig, str) or isinstance(edf, str):
        print(f"  {pat}: SKIPPED (dir not found)")
        continue
    if orig != edf:
        print(f"  MISMATCH {pat}: Original={orig}, chbmit_edf={edf}")
        all_match = False
if all_match:
    print("  ✓ EDF counts match between Original and data/chbmit_edf for all patients")

print(f"\n--- Original vs data/chbmit_subset (subset is expected to be smaller) ---")
for pat in PATIENTS:
    orig = orig_counts.get(pat, 0)
    sub = subset_counts.get(pat, {})
    if sub is None:
        continue
    sub_edf = sub.get('edf', 0)
    if isinstance(orig, str):
        continue
    if sub_edf > orig:
        print(f"  WARNING {pat}: Subset ({sub_edf}) has MORE files than Original ({orig})")

# Check if subset files exist in original
print(f"\n--- Subset files existence check in Original ---")
missing_in_orig = []
for pat in PATIENTS:
    sub_dir = os.path.join(SUBSET_DIR, pat)
    orig_dir = os.path.join(ORIGINAL_DIR, pat)
    if not os.path.isdir(sub_dir) or not os.path.isdir(orig_dir):
        continue
    sub_edfs = [f for f in os.listdir(sub_dir) if f.endswith('.edf') and not f.startswith('._')]
    orig_edfs = set(f for f in os.listdir(orig_dir) if f.endswith('.edf') and not f.startswith('._'))
    for ef in sub_edfs:
        if ef not in orig_edfs:
            missing_in_orig.append(f"{pat}/{ef}")

if missing_in_orig:
    print(f"  Files in subset but not in original: {missing_in_orig}")
else:
    print("  ✓ All subset EDF files exist in the original dataset")

# ============================================================
# 7. SUBJECT-INFO DEMOGRAPHICS
# ============================================================
separator("7. SUBJECT-INFO DEMOGRAPHICS")

subject_info_path = os.path.join(ORIGINAL_DIR, "SUBJECT-INFO")
if os.path.exists(subject_info_path):
    with open(subject_info_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    print(f"  Raw content:")
    for l in lines:
        print(f"    {l}")
    
    # Parse demographics
    demographics = []
    for l in lines[1:]:  # Skip header
        parts = l.split('\t')
        if len(parts) >= 3:
            case = parts[0].strip()
            gender = parts[1].strip()
            age = parts[2].strip()
            demographics.append({'case': case, 'gender': gender, 'age': age})
    
    print(f"\n  Total subjects: {len(demographics)}")
    
    # Gender distribution
    genders = [d['gender'] for d in demographics]
    print(f"  Gender distribution:")
    print(f"    Female (F): {genders.count('F')}")
    print(f"    Male (M): {genders.count('M')}")
    
    # Age statistics
    ages = []
    for d in demographics:
        try:
            ages.append(float(d['age']))
        except ValueError:
            pass
    
    if ages:
        print(f"  Age statistics:")
        print(f"    Min: {min(ages)} years")
        print(f"    Max: {max(ages)} years")
        print(f"    Mean: {sum(ages)/len(ages):.1f} years")
        print(f"    Median: {sorted(ages)[len(ages)//2]} years")
        
        # Age groups
        pediatric = sum(1 for a in ages if a < 18)
        adult = sum(1 for a in ages if a >= 18)
        print(f"    Pediatric (<18): {pediatric}")
        print(f"    Adult (≥18): {adult}")
    
    # Note: chb24 is missing from SUBJECT-INFO
    subjects_in_info = set(d['case'] for d in demographics)
    subjects_in_data = set(PATIENTS)
    missing_subjects = subjects_in_data - subjects_in_info
    extra_subjects = subjects_in_info - subjects_in_data
    if missing_subjects:
        print(f"\n  Patients in dataset but NOT in SUBJECT-INFO: {missing_subjects}")
    if extra_subjects:
        print(f"  Patients in SUBJECT-INFO but NOT in dataset: {extra_subjects}")
else:
    print("  SUBJECT-INFO file NOT FOUND")

# ============================================================
# SUMMARY
# ============================================================
separator("FINAL SUMMARY")

print(f"  Dataset: CHB-MIT Scalp EEG Database")
print(f"  Total patients: {len(PATIENTS)}")
print(f"  Total EDF files (Original): {sum(v for v in orig_counts.values() if isinstance(v, int))}")
print(f"  Total EDF files (chbmit_edf): {sum(v for v in edf_counts.values() if isinstance(v, int))}")
print(f"  Total EDF files (chbmit_subset): {total_subset}")
print(f"  Total seizures annotated: {total_seizures_all}")
print(f"  Total seizure duration: {total_seizure_duration_all}s ({total_seizure_duration_all/60:.1f} min)")
print(f"  Files with seizures: {total_files_with_seizures}")
print(f"  Files without seizures: {total_files_without_seizures}")
print(f"  Parquet files analyzed: {len(PARQUET_FILES)}")
print(f"\n  Report saved to: {OUTPUT_FILE}")

report_file.close()
sys.stdout = sys.__stdout__
print(f"\nDone! Report saved to: {OUTPUT_FILE}")
