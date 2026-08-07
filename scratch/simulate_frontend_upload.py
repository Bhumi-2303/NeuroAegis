import requests
import json
import time

import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "data/chbmit_subset/chb01/chb01_01.edf"
print(f"Simulating frontend upload of {filepath}")

# Mimic the FormData sent by the frontend
with open(filepath, "rb") as file_obj:
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/predict/",
        files={"file": (filepath.split('/')[-1], file_obj, "application/octet-stream")},
        data={"sampling_rate": 256.0}
    )

print("Status Code:", response.status_code)
resp_json = response.json()
print("Response:", resp_json)

if "job_id" in resp_json:
    job_id = resp_json["job_id"]
    print(f"\nPolling job {job_id}...")
    for _ in range(30):
        job_res = requests.get(f"http://127.0.0.1:8000/api/v1/jobs/{job_id}")
        job_data = job_res.json()
        if job_data.get("status") == "Completed":
            print("\nJOB COMPLETED:")
            print(json.dumps(job_data, indent=2))
            break
        elif job_data.get("status") == "Failed":
            print("\nJOB FAILED:", job_data)
            break
        time.sleep(2)
