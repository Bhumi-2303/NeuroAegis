import random
import numpy as np
import torch
import os

def seed_everything(seed: int):
    """
    Called at the top of every training script to ensure reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    print(f"Random seed set to {seed} across Python, NumPy, and Torch/MPS.")
