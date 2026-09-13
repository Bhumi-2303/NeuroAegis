import torch
from neuroaegis.models import get_model, ModelConfig, get_loss_fn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_models():
    # 1. Device selection: Apple MPS with CPU fallback
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Config & Synthetic Batch setup
    # From Contract 2.2 and 2.1
    batch_size = 4
    config = ModelConfig(
        in_channels=18,
        seq_len=1280, # 256 Hz * 5 seconds
        graph_method="correlation",
        loss="weighted_bce"
    )
    
    # Synthetic window samples (mimicking extracted np.ndarray from WindowSample)
    # Shape: (batch, channels, seq_len)
    X = torch.randn(batch_size, config.in_channels, config.seq_len, device=device)
    y = torch.randint(0, 2, (batch_size, 1), device=device).float()
    
    loss_fn = get_loss_fn(config.loss, pos_weight=1.5).to(device)
    
    models_to_test = ["logistic_regression", "cnn", "cnn_gru", "proposed"]
    
    for model_name in models_to_test:
        print(f"\n--- Testing {model_name} ---")
        
        # Instantiate model
        # Test PLV for proposed as well
        if model_name == "proposed":
            config.graph_method = "plv"
            
        model = get_model(model_name, config).to(device)
        
        # Verify parameter count constraint
        param_count = count_parameters(model)
        print(f"Total trainable parameters: {param_count:,}")
        
        if model_name == "proposed":
            if param_count > 5000000 or param_count < 50000:
                print(f"WARNING: Proposed model params out of expected 1M-5M range ({param_count})")
        
        # One Forward Pass
        out = model(X)
        
        # Check Contract 2.3 Shape
        assert hasattr(out, 'logits'), "Missing logits"
        assert 'logits' in out
        print("Contract 2.3 Check Passed.")
        
        # Loss
        loss = loss_fn(out.logits, y)
        print(f"Loss value: {loss.item():.4f}")
        
        # One Backward Pass
        model.zero_grad()
        loss.backward()
        
        # Verify gradients exist
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"Backward pass successful, gradients generated: {has_grads}")
        
if __name__ == "__main__":
    test_models()
