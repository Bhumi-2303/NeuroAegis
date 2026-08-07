import joblib
import json
import numpy as np

# Load model
model = joblib.load('apps/api/models/chbmit/lightgbm_patient_wise.pkl')
if isinstance(model, dict) and 'model' in model:
    model = model['model']

booster = model.booster_
dump = booster.dump_model()

# Load selected features to get the index of Ch0_line_length
with open('apps/api/models/chbmit/selected_features_patient_wise.json', 'r') as f:
    features = json.load(f)
    
feature_name = "Ch0_line_length"
if feature_name in features:
    feature_idx = features.index(feature_name)
    print(f"Found {feature_name} at index {feature_idx}")
else:
    print(f"{feature_name} not found in features list!")
    exit(1)

# Recursive function to find all split thresholds for a given feature index
def find_splits(tree, feature_idx, thresholds):
    if 'split_feature' in tree:
        if tree['split_feature'] == feature_idx:
            thresholds.append(tree['threshold'])
        if 'left_child' in tree:
            find_splits(tree['left_child'], feature_idx, thresholds)
        if 'right_child' in tree:
            find_splits(tree['right_child'], feature_idx, thresholds)

all_thresholds = []
for tree_info in dump['tree_info']:
    tree = tree_info['tree_structure']
    find_splits(tree, feature_idx, all_thresholds)

if all_thresholds:
    print(f"Total splits on {feature_name}: {len(all_thresholds)}")
    print(f"Min threshold: {min(all_thresholds)}")
    print(f"Max threshold: {max(all_thresholds)}")
    print(f"Mean threshold: {np.mean(all_thresholds)}")
    print(f"90th percentile threshold: {np.percentile(all_thresholds, 90)}")
    
    # Check if 0.0846 is greater than the max threshold
    test_vals = [0.0883, 0.2241, 0.0846]
    for val in test_vals:
        pct_above = sum(1 for t in all_thresholds if val > t) / len(all_thresholds) * 100
        print(f"Value {val} is greater than {pct_above:.1f}% of splits for {feature_name}")
else:
    print(f"No splits found for {feature_name}")
