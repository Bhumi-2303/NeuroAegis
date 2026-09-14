import torch
import hashlib
import json
import os
import sys

# Add root to path so we can import model
sys.path.append(os.path.abspath('.'))
from neuroaegis.models.baselines.model_c import CNN_GNN_GRU

def get_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

ckpt_path = 'artifacts/checkpoints/frozen_cnn_gnn_gru.pt'
expected_hash = '2ec84897c39d31d68cfbf1e5c8708d5073fe19450576bf4f9132929c1832ca'
actual_hash = get_hash(ckpt_path)

hash_match = actual_hash == expected_hash

if not hash_match:
    print(f"HASH MISMATCH. Expected: {expected_hash}, Actual: {actual_hash}")
    sys.exit(1)

model = CNN_GNN_GRU(
    cnn_gnn_checkpoint='research/experiments/cnn_baseline/best_cnn_gnn.pt', # Wait, need to see the init of CNN_GNN_GRU
    gru_hidden_size=64,
    gru_num_layers=1,
    dropout_rate=0.3
)
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

# Param counting
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

trainable_layers = []
for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_layers.append({
            "name": name,
            "shape": list(param.shape),
            "parameters": param.numel()
        })

frozen_backbone = sum(p.numel() for name, p in model.named_parameters() if 'backbone' in name)
trainable_head = sum(p.numel() for name, p in model.named_parameters() if 'backbone' not in name)

audit_data = {
    "checkpoint_path": ckpt_path,
    "expected_hash": expected_hash,
    "actual_hash": actual_hash,
    "hash_match": hash_match,
    "total_parameters": total_params,
    "expected_total_parameters": 91858,
    "total_parameters_match": total_params == 91858,
    "frozen_parameters": frozen_params,
    "expected_frozen_parameters": 52497,
    "frozen_parameters_match": frozen_params == 52497,
    "trainable_parameters": trainable_params,
    "expected_trainable_parameters": 39361,
    "trainable_parameters_match": trainable_params == 39361,
    "frozen_backbone": frozen_backbone,
    "trainable_head": trainable_head,
    "trainable_layers": trainable_layers,
    "gru_config": {
        "input_size": model.gru.input_size if hasattr(model, 'gru') else None,
        "hidden_size": model.gru.hidden_size if hasattr(model, 'gru') else None,
        "num_layers": model.gru.num_layers if hasattr(model, 'gru') else None,
        "bidirectional": model.gru.bidirectional if hasattr(model, 'gru') else None
    }
}

with open('research/audit/model_c_validation/model_architecture_audit.json', 'w') as f:
    json.dump(audit_data, f, indent=4)
    
print("Step 1 done")
