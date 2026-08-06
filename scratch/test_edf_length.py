import mne
import numpy as np
import requests

def create_edf(filename, seconds):
    info = mne.create_info(ch_names=[f"Ch{i}" for i in range(23)], sfreq=256, ch_types='eeg')
    data = np.random.randn(23, int(256 * seconds))
    raw = mne.io.RawArray(data, info)
    mne.export.export_raw(filename, raw, fmt="edf", overwrite=True)

# 1-hour EDF
create_edf("chb_1hr.edf", 3600)
# 1-hour + 1-second EDF
create_edf("chb_1hr_1sec.edf", 3601)

for f in ["chb_1hr.edf", "chb_1hr_1sec.edf"]:
    print(f"\nUploading {f}...")
    with open(f, "rb") as file_obj:
        response = requests.post(
            "http://127.0.0.1:8000/api/v1/predict/",
            files={"file": (f, file_obj, "application/octet-stream")}
        )
    print("Status Code:", response.status_code)
    try:
        print("Response:", response.json())
    except:
        print("Response:", response.text)
