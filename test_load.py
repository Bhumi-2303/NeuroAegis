import torch
c = torch.load('artifacts/checkpoints/frozen_cnn_gnn_gru.pt', map_location='cpu')
print(type(c))
if isinstance(c, dict):
    print(c.keys())
