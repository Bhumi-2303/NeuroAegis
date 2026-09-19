import torch
import torch.nn as nn
from typing import Any, Optional, Dict
import warnings
from neuroaegis.guardrails.checks import label_attention_visualization

class IntegratedGradients:
    """
    Architecture-agnostic Integrated Gradients.
    Computes attributions with respect to the input sequence x.
    """
    def __init__(self, model: nn.Module, steps: int = 50):
        self.model = model
        self.steps = steps

    def attribute(self, x: torch.Tensor, target_class: int = 0) -> torch.Tensor:
        # x shape: (1, channels, seq_len)
        self.model.eval()
        baseline = torch.zeros_like(x)
        
        alphas = torch.linspace(0.0, 1.0, steps=self.steps, device=x.device).view(-1, 1, 1, 1)
        # Scaled inputs: (steps, batch, channels, seq_len)
        scaled_inputs = baseline.unsqueeze(0) + alphas * (x - baseline).unsqueeze(0)
        
        # We process step by step to save memory, or all at once if batch fits.
        # Given memory constraints (1-5M params, batch 4-8), we process sequentially.
        accumulated_grads = torch.zeros_like(x)
        
        try:
            for step_idx in range(self.steps):
                inp = scaled_inputs[step_idx].requires_grad_(True)
                out = self.model(inp)
                logits = out.logits
                
                # Select target class
                if logits.size(-1) > 1:
                    score = logits[0, target_class]
                else:
                    score = logits[0, 0] # Binary classification
                    if target_class == 0:
                        score = -score # flip gradient direction for class 0 if using a single logit
                
                score.backward()
                if inp.grad is not None:
                    accumulated_grads += inp.grad.detach()
                self.model.zero_grad(set_to_none=True)
                del inp, out, logits, score
                
            avg_grads = accumulated_grads / self.steps
            ig_attributions = (x - baseline) * avg_grads
            return ig_attributions
        finally:
            self.model.zero_grad(set_to_none=True)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()


class AttentionExtractor:
    """
    Extracts attention weights from models that expose it in the Contract 2.3 dict.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        
    def attribute(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
            
        attn = out.attention_weights
        if attn is None:
            warnings.warn("Model did not return attention_weights.")
            return None
            
        # Attention is inherently unvalidated per the clinical-agreement spec
        # Here we just return the raw weights. The harness handles the auto-labeling.
        return attn

class SHAPWrapper:
    """
    Optional SHAP wrapper for tree-based baselines or NN embeddings.
    """
    def __init__(self, model: Any, model_type: str = "deep", background_data: Optional[torch.Tensor] = None):
        self.model = model
        self.model_type = model_type
        self.explainer = None
        
        try:
            import shap
            self.shap = shap
        except ImportError:
            self.shap = None
            warnings.warn("SHAP library not installed. SHAP wrapper will return None.")
            
        if self.shap and background_data is not None:
            if self.model_type == "tree":
                self.explainer = self.shap.TreeExplainer(self.model)
            else:
                # Wrap model to only return logits
                class LogitsOnly(nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self.m = m
                    def forward(self, x):
                        return self.m(x).logits
                        
                self.explainer = self.shap.DeepExplainer(LogitsOnly(self.model), background_data)
                
    def attribute(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.shap or not self.explainer:
            return None
            
        if self.model_type == "tree":
            # For sklearn models wrapped in PyTorch or direct sklearn
            shap_values = self.explainer.shap_values(x.cpu().numpy())
            return torch.tensor(shap_values, device=x.device)
        else:
            shap_values = self.explainer.shap_values(x)
            # shap_values might be a list depending on num_classes
            if isinstance(shap_values, list):
                return shap_values[1] # target class 1
            return shap_values
