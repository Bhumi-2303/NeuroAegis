import torch
from typing import Dict

def compute_frequency_band_power(x: torch.Tensor, attribution: torch.Tensor, sampling_rate: int = 256) -> Dict[str, float]:
    """
    Decomposes the attributed regions into standard EEG frequency bands.
    Approach: Compute Power Spectral Density (PSD) of the signal weighted by the attribution mask.
    x shape: (1, channels, seq_len)
    attribution shape: (1, channels, seq_len)
    """
    # Weight the raw signal by the absolute attribution
    weighted_x = x * torch.abs(attribution)
    
    # Compute PSD via FFT
    fft_vals = torch.fft.rfft(weighted_x, dim=-1)
    psd = torch.abs(fft_vals) ** 2
    
    # Average across channels and batch
    psd = psd.mean(dim=(0, 1))
    
    # Frequencies
    freqs = torch.fft.rfftfreq(x.size(-1), 1 / sampling_rate)
    
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 40.0) # Up to 40Hz per preprocessing config
    }
    
    band_powers = {}
    total_power = psd.sum().item() + 1e-8
    
    for band_name, (low, high) in bands.items():
        idx = (freqs >= low) & (freqs <= high)
        power = psd[idx].sum().item()
        # Normalized power for the band
        band_powers[band_name] = power / total_power
        
    return band_powers

def extract_levels(attribution: torch.Tensor, x: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """
    Extracts time, channel, and frequency level attributions.
    """
    # 1. Time (aggregate across channels)
    # Shape: (1, seq_len) -> squeeze to (seq_len)
    time_attr = torch.abs(attribution).mean(dim=1).squeeze(0)
    
    # 2. Channel (aggregate across time)
    # Shape: (1, channels) -> squeeze to (channels)
    channel_attr = torch.abs(attribution).mean(dim=2).squeeze(0)
    
    # 3. Frequency Band
    freq_attr = None
    if x is not None:
        band_powers = compute_frequency_band_power(x, attribution)
        freq_attr = torch.tensor(list(band_powers.values()), device=attribution.device)
        
    return {
        "time": time_attr,
        "channel": channel_attr,
        "frequency": freq_attr
    }
