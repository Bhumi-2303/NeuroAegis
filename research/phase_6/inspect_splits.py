import pandas as pd

df = pd.read_csv("/Volumes/BLACK-BOX/NeuroAegis/research/phase_6/manifests/siena_manifest.csv")
print("Recordings by cohort split:")
for split, g in df.groupby("cohort_split"):
    print(f"\n--- {split} --- ({len(g)} recordings, {g['seizure_count'].sum()} seizures, {g['duration_hours'].sum():.1f} hours, {g['file_size_mb'].sum():.1f} MB = {g['file_size_mb'].sum()/1024:.2f} GB)")
    for _, r in g.sort_values("file_size_mb").iterrows():
        print(f"  {r['recording_id']}: {r['file_size_mb']:.1f} MB, {r['duration_hours']:.2f}h, {r['seizure_count']} sz")
