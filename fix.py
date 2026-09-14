import json

with open('research/audit/model_c_validation/checkpoint_forensics_data.json') as f:
    data = json.load(f)

import torch
from neuroaegis.models.baselines.model_c import CNN_GNN_GRU

ckpt = torch.load('artifacts/checkpoints/frozen_cnn_gnn_gru.pt', map_location='cpu', weights_only=False)
if 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
    data["structural_inspection"]["has_epoch"] = 'epoch' in ckpt
else:
    state_dict = ckpt

total_params = 0
structural_info = {}
for k, v in state_dict.items():
    if isinstance(v, torch.Tensor):
        shape = list(v.shape)
        dtype = str(v.dtype)
        params = v.numel()
        total_params += params
        structural_info[k] = {"shape": shape, "dtype": dtype, "parameters": params}

model = CNN_GNN_GRU(frozen_backbone_path="artifacts/checkpoints/frozen_cnn_gnn.pt", gru_hidden_dim=64, gru_layers=1, dropout_classifier=0.3)
model_sd = model.state_dict()

missing_keys = set(model_sd.keys()) - set(state_dict.keys())
unexpected_keys = set(state_dict.keys()) - set(model_sd.keys())

data["structural_inspection"]["total_parameters"] = total_params
data["structural_inspection"]["keys"] = structural_info
data["structural_inspection"]["missing_keys"] = list(missing_keys)
data["structural_inspection"]["unexpected_keys"] = list(unexpected_keys)

if not missing_keys and not unexpected_keys:
    data["structural_inspection"]["match_status"] = "EXACT MATCH"
elif missing_keys or unexpected_keys:
    data["structural_inspection"]["match_status"] = "PARTIAL MATCH" if len(set(model_sd.keys()).intersection(set(state_dict.keys()))) > 0 else "MISMATCH"

with open('research/audit/model_c_validation/checkpoint_forensics_data.json', 'w') as f:
    json.dump(data, f, indent=4)
