import requests
import numpy as np

def generate_csv(filename, samples):
    data = np.random.randn(samples, 23)
    np.savetxt(filename, data, delimiter=",")

# Test 1: Exactly 1 hour (921600 samples)
generate_csv("chb_1hr.csv", 921600)
# Test 2: 1 hour + 1 second (921600 + 256 = 921856 samples)
generate_csv("chb_1hr_1sec.csv", 921856)
# Test 3: Exactly 60 seconds (15360 samples)
generate_csv("chb_60s.csv", 15360)

for f in ["chb_60s.csv", "chb_1hr.csv", "chb_1hr_1sec.csv"]:
    print(f"\nUploading {f}...")
    with open(f, "rb") as file_obj:
        response = requests.post(
            "http://127.0.0.1:8000/api/v1/predict/",
            files={"file": (f, file_obj, "text/csv")},
            data={"sampling_rate": 256.0}
        )
    print("Status Code:", response.status_code)
    try:
        print("Response:", response.json())
    except:
        print("Response:", response.text)
