from typing import Optional
import torch

class ModelOutput(dict):
    """
    Contract 2.3: Standardized output dictionary for all models.
    Every model must return this dict shape, with unused keys as None.
    """
    def __init__(
        self, 
        logits: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        graph_adjacency: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None
    ):
        super().__init__(
            logits=logits,
            attention_weights=attention_weights,
            graph_adjacency=graph_adjacency,
            embeddings=embeddings
        )
        # Ensure we can access them via attribute for convenience if needed
        self.logits = logits
        self.attention_weights = attention_weights
        self.graph_adjacency = graph_adjacency
        self.embeddings = embeddings

class ModelConfig:
    def __init__(
        self,
        in_channels: int = 18,
        seq_len: int = 1280, # 256 Hz * 5 seconds = 1280
        graph_method: str = "correlation", # "correlation" | "plv"
        loss: str = "weighted_bce", # "weighted_bce" | "focal"
        cnn_hidden_dim: int = 64,
        gnn_hidden_dim: int = 128,
        gru_hidden_dim: int = 256,
        num_classes: int = 1
    ):
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.graph_method = graph_method
        self.loss = loss
        self.cnn_hidden_dim = cnn_hidden_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.num_classes = num_classes
