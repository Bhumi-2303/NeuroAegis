import numpy as np
f = np.load("research/experiments/model_c/experiments/L8/val_predictions.npz")
print(f.files)
for k in f.files:
    print(k, f[k].shape, f[k].dtype)
