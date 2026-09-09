import os
import json
import re
import pandas as pd
import hashlib

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
PHASE6_DIR = os.path.join(BASE_DIR, "research/phase_6")
MANIFEST_DIR = os.path.join(PHASE6_DIR, "manifests")
SCRATCH_META_DIR = "/Users/tirthkosambia/.gemini/antigravity/brain/cdc5b036-6469-4204-8ef9-d71f4ac69623/scratch/siena_meta"

os.makedirs(MANIFEST_DIR, exist_ok=True)

# 1. Load EDF headers & sizes
with open(os.path.join(SCRATCH_META_DIR, "all_edf_headers.json"), "r") as f:
    headers = json.load(f)

with open(os.path.join(SCRATCH_META_DIR, "edf_sizes.json"), "r") as f:
    sizes = json.load(f)

with open(os.path.join(SCRATCH_META_DIR, "RECORDS"), "r") as f:
    records = [line.strip() for line in f if line.strip()]

df_subj = pd.read_csv(os.path.join(SCRATCH_META_DIR, "subject_info.csv"))
df_subj.columns = [c.strip() for c in df_subj.columns]

# 2. Parse Seizures
def parse_time_to_sec(t_str):
    t_str = t_str.strip().replace(":", ".")
    parts = [int(p) for p in t_str.split(".")[:3]]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]

all_seizures = []
patients = sorted(list(set([r.split("/")[0] for r in records])))

for p in patients:
    txt_path = os.path.join(SCRATCH_META_DIR, p, f"Seizures-list-{p}.txt")
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        
    top_reg_start = None
    top_fn = None
    top_m = re.search(r"File\s*name\s*:\s*(\S+).*?Registration\s*start\s*time\s*:\s*([0-9\.: ]+)", content, re.DOTALL | re.IGNORECASE)
    if top_m:
        top_fn = top_m.group(1).strip()
        top_reg_start = parse_time_to_sec(top_m.group(2).replace(" ", ""))

    raw_blocks = re.split(r"(Seizure\s+n\s*[\d\w\(\)\- :]+)", content)
    for k in range(1, len(raw_blocks), 2):
        header_line = raw_blocks[k].strip()
        body = raw_blocks[k+1].strip()
        combined = header_line + "\n" + body
        
        fn_m = re.search(r"File\s*name\s*:\s*(\S+)", combined, re.IGNORECASE)
        if fn_m:
            fn = fn_m.group(1).strip()
        elif top_fn:
            fn = top_fn
        else:
            fn = "UNKNOWN"
            
        fn = fn.replace("PNO", "PN0")
        if fn == "PN11-.edf": fn = "PN11-1.edf"
        elif fn == "PN01.edf": fn = "PN01-1.edf"
        if not fn.endswith(".edf"): fn += ".edf"
        
        rec_id = f"{p}/{fn}"
        
        reg_m = re.search(r"Registration\s*start\s*time\s*:\s*([0-9\.: ]+)", combined, re.IGNORECASE)
        if reg_m:
            reg_start = parse_time_to_sec(reg_m.group(1).replace(" ", ""))
        elif top_reg_start is not None:
            reg_start = top_reg_start
        else:
            reg_start = None
            
        sz_s_m = re.search(r"(?:Seizure\s+[sS]tart\s*time|(?<!Registration\s)[sS]tart\s*time)\s*:\s*([0-9\.: ]+)", combined)
        sz_e_m = re.search(r"(?:Seizure\s+[eE]nd\s*time|(?<!Registration\s)[eE]nd\s*time)\s*:\s*([0-9\.: ]+)", combined)
        
        if sz_s_m and sz_e_m and reg_start is not None:
            m1 = re.search(r"(\d{1,2}[\.:]\d{2}[\.:]\d{2})", sz_s_m.group(1))
            m2 = re.search(r"(\d{1,2}[\.:]\d{2}[\.:]\d{2})", sz_e_m.group(1))
            if m1 and m2:
                s_clock = m1.group(1)
                e_clock = m2.group(1)
                
                # Typo correction for PN00 Seizure 3
                if p == "PN00" and "Seizure n 3" in header_line and e_clock == "19.29.29":
                    e_clock = "18.29.29"
                    
                s_sec = parse_time_to_sec(s_clock)
                e_sec = parse_time_to_sec(e_clock)
                
                onset = (s_sec - reg_start) % 86400
                offset = (e_sec - reg_start) % 86400
                dur = offset - onset
                rec_dur = headers[rec_id]["total_duration_sec"]
                
                # Extract clean seizure number
                sz_num_m = re.search(r"Seizure\s+n\s*(\d+)", header_line, re.IGNORECASE)
                sz_num = int(sz_num_m.group(1)) if sz_num_m else len(all_seizures) + 1
                sz_id = f"{p}_sz{sz_num:02d}"
                
                all_seizures.append({
                    "seizure_id": sz_id,
                    "patient_id": p,
                    "recording_id": rec_id,
                    "file_name": fn,
                    "seizure_raw_title": header_line,
                    "reg_start_clock": reg_start,
                    "sz_start_clock": s_clock,
                    "sz_end_clock": e_clock,
                    "start_sec": float(onset),
                    "end_sec": float(offset),
                    "duration_sec": float(dur),
                    "rec_duration_sec": float(rec_dur),
                    "cohort_split": "CALIBRATION" if p in ["PN00", "PN01", "PN03", "PN05"] else "TEST"
                })

df_events = pd.DataFrame(all_seizures)
events_path = os.path.join(MANIFEST_DIR, "siena_seizure_events.csv")
df_events.to_csv(events_path, index=False)
print(f"Saved {events_path} ({len(df_events)} seizures).")

# 3. Build Recording Manifest
rec_rows = []
for r in records:
    p = r.split("/")[0]
    h = headers[r]
    sz_in_rec = df_events[df_events["recording_id"] == r]
    rec_rows.append({
        "recording_id": r,
        "patient_id": p,
        "file_name": os.path.basename(r),
        "file_size_bytes": sizes.get(r, 0),
        "file_size_mb": round(sizes.get(r, 0) / (1024**2), 2),
        "sampling_rate_hz": h["fs"],
        "num_channels": h["num_channels"],
        "duration_sec": h["total_duration_sec"],
        "duration_hours": round(h["total_duration_sec"] / 3600, 4),
        "seizure_count": len(sz_in_rec),
        "has_seizure": len(sz_in_rec) > 0,
        "cohort_split": "CALIBRATION" if p in ["PN00", "PN01", "PN03", "PN05"] else "TEST"
    })

df_recs = pd.DataFrame(rec_rows)
recs_path = os.path.join(MANIFEST_DIR, "siena_manifest.csv")
df_recs.to_csv(recs_path, index=False)
print(f"Saved {recs_path} ({len(df_recs)} recordings).")

# 4. Build Patient Manifest
pat_rows = []
for _, row in df_subj.iterrows():
    p = row["patient_id"]
    p_recs = df_recs[df_recs["patient_id"] == p]
    p_szs = df_events[df_events["patient_id"] == p]
    pat_rows.append({
        "patient_id": p,
        "age_years": row["age_years"],
        "gender": row["gender"],
        "seizure_type": row["seizure"],
        "localization": row["localization"],
        "lateralization": row["lateralization"],
        "raw_eeg_channels": row["eeg_channel"],
        "total_recordings": len(p_recs),
        "seizure_recordings": int((p_recs["seizure_count"] > 0).sum()),
        "total_seizures": len(p_szs),
        "total_duration_sec": p_recs["duration_sec"].sum(),
        "total_duration_hours": round(p_recs["duration_sec"].sum() / 3600, 4),
        "total_size_mb": round(p_recs["file_size_mb"].sum(), 2),
        "cohort_split": "CALIBRATION" if p in ["PN00", "PN01", "PN03", "PN05"] else "TEST"
    })

df_pats = pd.DataFrame(pat_rows)
pats_path = os.path.join(MANIFEST_DIR, "siena_patient_manifest.csv")
df_pats.to_csv(pats_path, index=False)
print(f"Saved {pats_path} ({len(df_pats)} patients).")
