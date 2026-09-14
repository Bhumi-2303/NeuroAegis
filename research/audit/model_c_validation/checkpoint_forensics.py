import os
import hashlib
import torch
import glob
import time
import json
import subprocess
import sys

# Add root to path so we can import model
sys.path.append(os.path.abspath('.'))
from neuroaegis.models.baselines.model_c import CNN_GNN_GRU

def get_file_info(filepath):
    if not os.path.exists(filepath):
        return None
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    stat = os.stat(filepath)
    return {
        "path": filepath,
        "sha256": sha256.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime": time.ctime(stat.st_mtime)
    }

current_ckpt_path = 'artifacts/checkpoints/frozen_cnn_gnn_gru.pt'
current_ckpt_info = get_file_info(current_ckpt_path)

# Find all checkpoints
search_dirs = ['research', 'models', 'checkpoints', 'saved', 'artifacts']
candidate_checkpoints = []
for d in search_dirs:
    if os.path.isdir(d):
        for root, dirs, files in os.walk(d):
            for file in files:
                if file.endswith('.pt') or file.endswith('.pth'):
                    filepath = os.path.join(root, file)
                    candidate_checkpoints.append(get_file_info(filepath))

# Load checkpoint
try:
    checkpoint_content = torch.load(current_ckpt_path, map_location='cpu', weights_only=False)
except Exception as e:
    checkpoint_content = str(e)

has_optimizer = False
has_epoch = False
is_state_dict_only = False
state_dict = None

if isinstance(checkpoint_content, dict) and 'state_dict' in checkpoint_content:
    state_dict = checkpoint_content['state_dict']
    has_optimizer = 'optimizer' in checkpoint_content or 'optimizer_state_dict' in checkpoint_content
    has_epoch = 'epoch' in checkpoint_content
elif isinstance(checkpoint_content, dict) and any(k.startswith('backbone') for k in checkpoint_content.keys()):
    state_dict = checkpoint_content
    is_state_dict_only = True
else:
    state_dict = checkpoint_content # Try assuming it's a state dict directly

structural_info = {}
total_params = 0
if isinstance(state_dict, dict):
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            shape = list(v.shape)
            dtype = str(v.dtype)
            params = v.numel()
            total_params += params
            structural_info[k] = {"shape": shape, "dtype": dtype, "parameters": params}

# Compare with model implementation
model = CNN_GNN_GRU(
    frozen_backbone_path='artifacts/checkpoints/frozen_cnn_gnn.pt',
    gru_hidden_dim=64,
    gru_layers=1,
    dropout_classifier=0.3
)
model_state_dict = model.state_dict()

missing_keys = set(model_state_dict.keys()) - set(state_dict.keys()) if isinstance(state_dict, dict) else set()
unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys()) if isinstance(state_dict, dict) else set()

if isinstance(state_dict, dict) and not missing_keys and not unexpected_keys:
    match_status = "EXACT MATCH"
elif isinstance(state_dict, dict) and (len(missing_keys) > 0 or len(unexpected_keys) > 0) and len(set(model_state_dict.keys()).intersection(set(state_dict.keys()))) > 0:
    match_status = "PARTIAL MATCH"
else:
    match_status = "MISMATCH"

model_expected_params = sum(p.numel() for p in model.parameters())

results = {
    "current_checkpoint": current_ckpt_info,
    "candidate_checkpoints": candidate_checkpoints,
    "structural_inspection": {
        "is_state_dict_only": is_state_dict_only,
        "has_optimizer": has_optimizer,
        "has_epoch": has_epoch,
        "total_parameters": total_params,
        "expected_parameters": 91858,
        "model_implementation_parameters": model_expected_params,
        "keys": structural_info,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "match_status": match_status
    }
}

with open('research/audit/model_c_validation/checkpoint_forensics_data.json', 'w') as f:
    json.dump(results, f, indent=4)
