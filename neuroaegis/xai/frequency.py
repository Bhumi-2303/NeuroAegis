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
    # Check if attribution is 2D (batch, time) or 3D (batch, channels, time)
    if attribution.dim() == 2:
        # If it's only temporal attention, channel attribution is uniform
        batch_size, seq_len = attribution.shape
        num_channels = x.size(1) if x is not None else 1
        
        # 1. Time
        time_attr = torch.abs(attribution).squeeze(0)
        
        # 2. Channel
        channel_attr = torch.ones(num_channels, device=attribution.device) / num_channels
        
        # Expand for frequency
        attribution_expanded = attribution.unsqueeze(1).expand(-1, num_channels, -1)
    else:
        # 3D
        time_attr = torch.abs(attribution).mean(dim=1).squeeze(0)
        channel_attr = torch.abs(attribution).mean(dim=2).squeeze(0)
        attribution_expanded = attribution
        
    # Interpolate time_attr to match original x length if x is provided
    if x is not None and time_attr.size(-1) != x.size(-1):
        # time_attr shape is currently (seq_len)
        # interpolate needs (batch, channels, seq_len) -> (1, 1, seq_len)
        t_reshaped = time_attr.unsqueeze(0).unsqueeze(0)
        t_interp = torch.nn.functional.interpolate(t_reshaped, size=x.size(-1), mode='linear', align_corners=False)
        time_attr = t_interp.squeeze(0).squeeze(0)
    
    # 3. Frequency Band
    freq_attr = None
    if x is not None:
        # Interpolate attribution_expanded to match x's seq_len if reduced by CNN pooling
        if attribution_expanded.size(-1) != x.size(-1):
            attr_resized = torch.nn.functional.interpolate(
                attribution_expanded.unsqueeze(0), # (1, batch, channels, time)
                size=(attribution_expanded.size(1), x.size(-1)), 
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            attr_resized = attribution_expanded
            
        band_powers = compute_frequency_band_power(x, attr_resized)
        freq_attr = torch.tensor(list(band_powers.values()), device=attribution.device)
        
    return {
        "time": time_attr,
        "channel": channel_attr,
        "frequency": freq_attr
    }
