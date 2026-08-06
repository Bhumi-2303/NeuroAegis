import pandas as pd
import numpy as np

# Simulate a 1 hour EDF file DataFrame (921600 samples, 23 channels)
df_1hr = pd.DataFrame(np.zeros((921600, 23)))
df_1hr.columns = [str(i) for i in range(23)]

# Simulate a 1 hour + 1 second EDF file DataFrame
df_1hr_1sec = pd.DataFrame(np.zeros((921856, 23)))
df_1hr_1sec.columns = [str(i) for i in range(23)]

# Load detector
from apps.api.app.services.dataset_detection.detector import dataset_detector

det_ds, conf, rules = dataset_detector.detect(df_1hr, 256.0)
print("1 HOUR FILE:")
print("Dataset:", det_ds)
print("Confidence:", conf)
print("Rules:", rules)

det_ds2, conf2, rules2 = dataset_detector.detect(df_1hr_1sec, 256.0)
print("\n1 HOUR + 1 SEC FILE:")
print("Dataset:", det_ds2)
print("Confidence:", conf2)
print("Rules:", rules2)

