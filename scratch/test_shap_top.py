import sys
import json
import requests
import time

def upload_and_check(filepath):
    print(f"\n--- Testing {filepath} ---")
    url = "http://127.0.0.1:8000/api/v1/predict/"
    with open(filepath, 'rb') as f:
        files = {'file': (filepath.split('/')[-1], f, 'application/octet-stream')}
        response = requests.post(url, files=files)
    
    if response.status_code != 200:
        print("Upload failed:", response.text)
        return
        
    data = response.json()
    job_id = data['job_id']
    print(f"Job ID: {job_id}")
    
    # Poll for completion
    for _ in range(30):
        time.sleep(1.0)
        status_resp = requests.get(f"http://127.0.0.1:8000/api/v1/jobs/{job_id}")
        if status_resp.status_code == 200:
            status_data = status_resp.json()
            if status_data['status'] == 'Completed':
                res = status_data['result']
                features = res['shap_explanation']['features']
                sorted_features = sorted(features, key=lambda x: abs(x['value']), reverse=True)
                top = sorted_features[0]
                print(f"Prediction: {res['prediction_label']} (Prob: {res['probability_seizure']:.4f})")
                print(f"Top SHAP feature: {top['featureName']} (SHAP value: {top['value']:.4f}, Raw value: {top['rawValue']:.4f})")
                
                print("Top 3 features:")
                for i in range(min(3, len(sorted_features))):
                    f = sorted_features[i]
                    print(f"  {i+1}. {f['featureName']} (SHAP: {f['value']:.4f}, Raw: {f['rawValue']:.4f})")
                return
            elif status_data['status'] == 'Failed':
                print("Job failed!")
                return
    print("Timeout")

if __name__ == "__main__":
    files = [
        "data/chbmit_subset/chb01/chb01_03.edf",
        "data/chbmit_subset/chb01/chb01_01.edf",
        "data/chbmit_subset/chb01/chb01_01.edf"
    ]
    for f in files:
        upload_and_check(f)
