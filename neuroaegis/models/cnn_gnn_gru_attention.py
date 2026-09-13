import torch
import torch.nn as nn
import torch.nn.functional as F
from neuroaegis.models.schemas import ModelConfig, ModelOutput
from neuroaegis.models.graph import build_graph_adjacency

class SpatialTemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        # x shape: (batch, seq_len, hidden_dim)
        attn_scores = self.attn(x) # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * x, dim=1) # (batch, hidden_dim)
        return context, attn_weights.squeeze(-1)

class CNN_GNN_GRU_Attention(nn.Module):
    """
    Proposed architecture matching Section 8 of the spec.
    Data flow: CNN -> GNN -> GRU -> Attention
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 1. Independent CNN over time for each channel
        # We use in_channels=1 because we fold the EEG channels into the batch dim
        self.cnn = nn.Sequential(
            nn.Conv1d(1, config.cnn_hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(config.cnn_hidden_dim, config.cnn_hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 2. GNN layer (Graph Convolution over channels)
        # H' = A * H * W
        self.gnn_weight = nn.Parameter(torch.randn(config.cnn_hidden_dim, config.gnn_hidden_dim) / config.cnn_hidden_dim**0.5)
        self.gnn_bias = nn.Parameter(torch.zeros(config.gnn_hidden_dim))
        
        # 3. Temporal GRU
        gru_input_dim = config.in_channels * config.gnn_hidden_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=config.gru_hidden_dim,
            batch_first=True
        )
        
        # 4. Attention
        self.attention = SpatialTemporalAttention(config.gru_hidden_dim)
        
        # 5. Classifier
        self.fc = nn.Linear(config.gru_hidden_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # x shape: (batch, channels, seq_len)
        batch_size, num_channels, seq_len = x.size()
        
        # --- 0. Graph Construction ---
        # Compute adjacency early before reshaping
        adj = build_graph_adjacency(x, method=self.config.graph_method) # (batch, channels, channels)
        
        # --- 1. CNN ---
        # Fold channels into batch dimension: (batch * channels, 1, seq_len)
        x_cnn = x.view(batch_size * num_channels, 1, seq_len)
        cnn_out = self.cnn(x_cnn) # (batch * channels, cnn_hidden, reduced_seq_len)
        
        _, cnn_hidden, reduced_seq_len = cnn_out.size()
        # Reshape back: (batch, channels, cnn_hidden, reduced_seq_len)
        cnn_out = cnn_out.view(batch_size, num_channels, cnn_hidden, reduced_seq_len)
        
        # Transpose for GNN: (batch, reduced_seq_len, channels, cnn_hidden)
        cnn_out = cnn_out.permute(0, 3, 1, 2)
        
        # --- 2. GNN ---
        # We can apply GNN across channels for all time steps using torch.matmul
        # H: (batch, reduced_seq_len, channels, cnn_hidden)
        # A: (batch, channels, channels)
        
        # Add self-loops to adjacency and normalize (simple degree norm approximation)
        I = torch.eye(num_channels, device=x.device).unsqueeze(0)
        adj_tilde = adj + I
        rowsum = adj_tilde.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        adj_norm = adj_tilde / rowsum
        
        # A_norm @ H : (batch, channels, channels) @ (batch, reduced_seq_len, channels, cnn_hidden)
        # To do this efficiently, we can reshape H: (batch, channels, reduced_seq_len * cnn_hidden)
        H = cnn_out.permute(0, 2, 1, 3).reshape(batch_size, num_channels, -1)
        G_out = torch.bmm(adj_norm, H) # (batch, channels, reduced_seq_len * cnn_hidden)
        
        # Reshape back to (batch, reduced_seq_len, channels, cnn_hidden)
        G_out = G_out.view(batch_size, num_channels, reduced_seq_len, cnn_hidden).permute(0, 2, 1, 3)
        
        # Apply weights: H_new = G_out @ W + b
        # (batch, reduced_seq_len, channels, cnn_hidden) @ (cnn_hidden, gnn_hidden)
        gnn_out = torch.matmul(G_out, self.gnn_weight) + self.gnn_bias
        gnn_out = F.relu(gnn_out) # (batch, reduced_seq_len, channels, gnn_hidden)
        
        # --- 3. GRU ---
        # Flatten spatial dimensions: (batch, reduced_seq_len, channels * gnn_hidden)
        gru_in = gnn_out.reshape(batch_size, reduced_seq_len, num_channels * self.config.gnn_hidden_dim)
        gru_out, _ = self.gru(gru_in) # (batch, reduced_seq_len, gru_hidden)
        
        # --- 4. Attention ---
        context, attn_weights = self.attention(gru_out) # context: (batch, gru_hidden)
        
        # --- 5. Output ---
        logits = self.fc(context)
        
        return ModelOutput(
            logits=logits,
            attention_weights=attn_weights,
            graph_adjacency=adj,
            embeddings=context
        )
