import requests
import numpy as np

# Create a mock Bonn file (4097 samples, 1 channel)
data = np.random.randn(4097)
with open("bonn_mock.txt", "w") as f:
    for val in data:
        f.write(f"{val}\n")

print("Testing Bonn upload WITHOUT sampling rate...")
with open("bonn_mock.txt", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/predict/",
        files={"file": ("bonn_mock.txt", f, "text/plain")}
    )
print("Status Code:", response.status_code)
print("Response:", response.json())

print("\nTesting Bonn upload WITH sampling rate (173.61)...")
with open("bonn_mock.txt", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/predict/",
        files={"file": ("bonn_mock.txt", f, "text/plain")},
        data={"sampling_rate": 173.61}
    )
print("Status Code:", response.status_code)
print("Response:", response.json())
