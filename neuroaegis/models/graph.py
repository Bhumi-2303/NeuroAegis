import torch
import torch.nn.functional as F

def compute_pearson_correlation(x: torch.Tensor) -> torch.Tensor:
    """
    Computes Pearson correlation adjacency over channels.
    x shape: (batch, channels, seq_len)
    Returns adjacency of shape (batch, channels, channels)
    """
    # Center the sequence along time dimension
    x_mean = x.mean(dim=-1, keepdim=True)
    x_centered = x - x_mean
    
    # Compute covariance
    cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (x.size(-1) - 1 + 1e-8)
    
    # Compute std deviation
    std = torch.std(x, dim=-1, keepdim=True) + 1e-8
    std_matrix = torch.bmm(std, std.transpose(1, 2))
    
    # Correlation = Covariance / (std1 * std2)
    corr = cov / std_matrix
    
    # Ensure numerical stability
    return torch.clamp(corr, -1.0, 1.0)

def compute_plv(x: torch.Tensor) -> torch.Tensor:
    """
    Computes Phase-Locking-Value (PLV) adjacency.
    For simplicity in PyTorch without full Hilbert transform, 
    we approximate PLV or use a pseudo-phase approach via simple differentiation or analytic signal mock.
    In a true production setting, this would use torch.fft to get the analytic signal.
    x shape: (batch, channels, seq_len)
    Returns: (batch, channels, channels)
    """
    # Real PLV requires analytic signal. We implement it via FFT.
    # 1. FFT
    X = torch.fft.fft(x, dim=-1)
    
    # 2. Analytic signal (zero out negative frequencies)
    n = x.size(-1)
    # create step function: 1 at DC and Nyquist, 2 at positive freqs, 0 at negative freqs
    h = torch.zeros(n, device=x.device, dtype=X.dtype)
    if n > 0:
        h[0] = 1
        if n % 2 == 0:
            h[1:n//2] = 2
            h[n//2] = 1
        else:
            h[1:(n+1)//2] = 2
            
    # Apply analytic signal filter
    X_analytic = X * h.view(1, 1, -1)
    
    # 3. IFFT to get complex time-domain signal
    x_analytic = torch.fft.ifft(X_analytic, dim=-1)
    
    # 4. Extract instantaneous phase
    phase = torch.angle(x_analytic)
    
    # 5. Compute PLV between channels: | (1/T) * sum_t exp(i * (phase_c1(t) - phase_c2(t))) |
    # We can rewrite this using matrix multiplication over time
    # exp(i * phase)
    exp_phase = torch.exp(1j * phase) # shape: (batch, channels, seq_len)
    
    # dot product over time
    # plv(c1, c2) = | sum(exp_phase_c1 * conj(exp_phase_c2)) | / seq_len
    # exp_phase @ exp_phase.conj().T
    plv = torch.bmm(exp_phase, exp_phase.conj().transpose(1, 2)) / x.size(-1)
    
    return torch.abs(plv)

def build_graph_adjacency(x: torch.Tensor, method: str = "correlation") -> torch.Tensor:
    """
    Factory for graph adjacency matrix construction.
    """
    if method == "correlation":
        return compute_pearson_correlation(x)
    elif method == "plv":
        return compute_plv(x)
    else:
        raise ValueError(f"Unknown graph construction method: {method}")
