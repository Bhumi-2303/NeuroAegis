import json

meta_file = "research/experiments/temporal_post_processing/validation_inference_metadata.json"
with open(meta_file, "r") as f:
    meta = json.load(f)

meta["inference_duration_sec"] = "NOT MEASURED"
meta["windows_per_sec"] = "NOT MEASURED"

with open(meta_file, "w") as f:
    json.dump(meta, f, indent=4)
