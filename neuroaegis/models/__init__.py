from .schemas import ModelConfig, ModelOutput
from .baselines import BaselineLogisticRegression, DummyRandomForest, Baseline1DCNN, BaselineCNNGRU
from .cnn_gnn_gru_attention import CNN_GNN_GRU_Attention
from .losses import get_loss_fn

def get_model(model_name: str, config: ModelConfig):
    if model_name == "logistic_regression":
        return BaselineLogisticRegression(config)
    elif model_name == "random_forest":
        return DummyRandomForest(config)
    elif model_name == "cnn":
        return Baseline1DCNN(config)
    elif model_name == "cnn_gru":
        return BaselineCNNGRU(config)
    elif model_name == "proposed":
        return CNN_GNN_GRU_Attention(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
