import requests
import numpy as np

# Create a 2 minute file (256 * 120 = 30720 samples)
data = np.random.randn(30720, 23)
np.savetxt("chb01_03.csv", data, delimiter=",")

print("Uploading chb01_03.csv...")
with open("chb01_03.csv", "rb") as file_obj:
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/predict/",
        files={"file": ("chb01_03.csv", file_obj, "text/csv")},
        data={"sampling_rate": 256.0}
    )
print("Status Code:", response.status_code)
print("Response:", response.json())
