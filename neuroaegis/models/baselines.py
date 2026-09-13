import torch
import torch.nn as nn
from neuroaegis.models.schemas import ModelConfig, ModelOutput

class BaselineLogisticRegression(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Flattens (batch, channels, seq_len) into a single vector per example
        input_dim = config.in_channels * config.seq_len
        self.linear = nn.Linear(input_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # x shape: (batch, channels, seq_len)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        logits = self.linear(x_flat)
        return ModelOutput(logits=logits)

class DummyRandomForest(nn.Module):
    """
    Dummy wrapper representing Random Forest. 
    In PyTorch, RF is non-differentiable, but included to conform to the factory interface.
    Would typically wrap sklearn logic for streaming or batched inference.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> ModelOutput:
        batch_size = x.size(0)
        # Just return zeros to adhere to schema
        logits = torch.zeros((batch_size, 1), device=x.device) + self.dummy_param
        return ModelOutput(logits=logits)

class Baseline1DCNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Simple 1D CNN over time sequence
        self.conv = nn.Sequential(
            nn.Conv1d(config.in_channels, config.cnn_hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(config.cnn_hidden_dim, config.cnn_hidden_dim * 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(config.cnn_hidden_dim * 2, config.num_classes)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # x shape: (batch, channels, seq_len)
        feats = self.conv(x) # (batch, cnn_hidden_dim*2, 1)
        feats = feats.squeeze(-1) # (batch, cnn_hidden_dim*2)
        logits = self.fc(feats)
        return ModelOutput(logits=logits, embeddings=feats)

class BaselineCNNGRU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(config.in_channels, config.cnn_hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.gru = nn.GRU(
            input_size=config.cnn_hidden_dim, 
            hidden_size=config.gru_hidden_dim, 
            batch_first=True
        )
        self.fc = nn.Linear(config.gru_hidden_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # x shape: (batch, channels, seq_len)
        cnn_out = self.conv(x) # (batch, cnn_hidden_dim, reduced_seq_len)
        # Permute for GRU: (batch, seq, features)
        cnn_out = cnn_out.transpose(1, 2)
        gru_out, hidden = self.gru(cnn_out) # gru_out: (batch, seq, gru_hidden_dim)
        
        # Take the last hidden state for prediction
        final_state = gru_out[:, -1, :] # (batch, gru_hidden_dim)
        logits = self.fc(final_state)
        return ModelOutput(logits=logits, embeddings=final_state)
