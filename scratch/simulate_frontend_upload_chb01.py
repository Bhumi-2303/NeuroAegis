import requests
import json
import time

filepath = "data/chbmit_subset/chb01/chb01_01.edf"
print(f"Simulating frontend upload of {filepath}")

with open(filepath, "rb") as file_obj:
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/predict/",
        files={"file": (filepath.split('/')[-1], file_obj, "application/octet-stream")},
        data={"sampling_rate": 256.0}
    )

resp_json = response.json()
if "job_id" in resp_json:
    job_id = resp_json["job_id"]
    for _ in range(30):
        job_res = requests.get(f"http://127.0.0.1:8000/api/v1/jobs/{job_id}")
        job_data = job_res.json()
        if job_data.get("status") == "Completed":
            print(f"PROBABILITY: {job_data['result']['probability_seizure']}")
            break
        time.sleep(2)
