import torch
from typing import Dict

def get_top_k_indices(x: torch.Tensor, k: int) -> set:
    """Returns set of indices for top-k elements."""
    if k > len(x):
        k = len(x)
    _, idx = torch.topk(x, k)
    return set(idx.tolist())

def jaccard_similarity(x: torch.Tensor, y: torch.Tensor, k: int = 5) -> float:
    """Jaccard similarity on top-k important features."""
    set_x = get_top_k_indices(x, k)
    set_y = get_top_k_indices(y, k)
    if not set_x or not set_y:
        return 0.0
    intersection = len(set_x.intersection(set_y))
    union = len(set_x.union(set_y))
    return intersection / union

def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Spearman rank correlation using pure PyTorch.
    Returns correlation coefficient in [-1, 1].
    """
    # Convert to 1D
    x = x.flatten()
    y = y.flatten()
    
    # Compute ranks
    x_rank = x.argsort().argsort().float()
    y_rank = y.argsort().argsort().float()
    
    x_bar, y_bar = x_rank.mean(), y_rank.mean()
    num = ((x_rank - x_bar) * (y_rank - y_bar)).sum()
    den = torch.sqrt(((x_rank - x_bar)**2).sum() * ((y_rank - y_bar)**2).sum())
    
    if den == 0:
        return 0.0
    return (num / den).item()

def top_k_overlap(x: torch.Tensor, y: torch.Tensor, k: int = 5) -> float:
    """Simple top-k overlap ratio."""
    set_x = get_top_k_indices(x, k)
    set_y = get_top_k_indices(y, k)
    if k == 0:
        return 0.0
    return len(set_x.intersection(set_y)) / float(k)

def calculate_agreement_scores(attr1: Dict[str, torch.Tensor], attr2: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Computes all agreement metrics between two attribution dictionaries
    containing 'time', 'channel', and 'frequency' levels.
    """
    scores = {}
    
    # Channel Agreement
    c1, c2 = attr1["channel"], attr2["channel"]
    k_channel = max(1, len(c1) // 3)
    scores["channel_jaccard"] = jaccard_similarity(c1, c2, k=k_channel)
    scores["channel_spearman"] = spearman_correlation(c1, c2)
    scores["channel_top_k_overlap"] = top_k_overlap(c1, c2, k=k_channel)
    
    # Time-Region Agreement
    t1, t2 = attr1["time"], attr2["time"]
    k_time = max(1, len(t1) // 10) # top 10% time windows
    scores["time_jaccard"] = jaccard_similarity(t1, t2, k=k_time)
    scores["time_spearman"] = spearman_correlation(t1, t2)
    
    # Frequency-Band Agreement
    f1, f2 = attr1["frequency"], attr2["frequency"]
    if f1 is not None and f2 is not None:
        scores["freq_spearman"] = spearman_correlation(f1, f2)
    else:
        scores["freq_spearman"] = None
        
    return scores
