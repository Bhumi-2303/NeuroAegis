import torch
import torch.nn as nn
from neuroaegis.models import ModelOutput
from neuroaegis.xai import IntegratedGradients, AttentionExtractor, SHAPWrapper, build_comparison_harness

class StubXAIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(18 * 128, 1) # Dummy for grads
        
    def forward(self, x):
        # x: (batch, channels, time) -> (1, 18, 128)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        logits = self.fc(x_flat)
        
        # Random attention weights: (batch, channels, time)
        attn = torch.rand_like(x)
        
        return ModelOutput(
            logits=logits,
            attention_weights=attn
        )

def run_xai_pipeline():
    print("Initializing Stub Model...")
    model = StubXAIModel()
    
    # 1 Seizure Event Window
    # Shape: (batch, channels, seq_len)
    # Using smaller seq_len for fast test (128)
    x_event = torch.randn(1, 18, 128)
    
    print("Computing Integrated Gradients...")
    ig = IntegratedGradients(model, steps=10)
    ig_attr = ig.attribute(x_event, target_class=0)
    
    print("Extracting Attention...")
    attn_extractor = AttentionExtractor(model)
    attention_attr = attn_extractor.attribute(x_event)
    
    print("Computing SHAP Wrapper (Mock)...")
    # Because we don't have SHAP installed reliably or it's slow, we simulate SHAP
    # by instantiating the wrapper but injecting a mock if it fails.
    shap_wrapper = SHAPWrapper(model, model_type="deep")
    shap_attr = shap_wrapper.attribute(x_event)
    
    if shap_attr is None:
        print(" -> SHAP library not available, injecting random dummy SHAP values for comparison.")
        shap_attr = torch.randn_like(x_event)
        
    print("Building Comparison Harness...")
    explanation_map = build_comparison_harness(
        event_id="chb01_seizure_001",
        x=x_event,
        ig_attr=ig_attr,
        attention_attr=attention_attr,
        shap_attr=shap_attr
    )
    
    print("\n--- XAI Explanation Map ---")
    print(f"Event: {explanation_map.event_id}")
    explanation_map.display_warnings()
    
    print("\nPairwise Agreement Scores:")
    for pair, scores in explanation_map.agreement_scores.items():
        print(f"[{pair}]")
        for metric, val in scores.items():
            if val is not None:
                print(f"  - {metric}: {val:.4f}")
            else:
                print(f"  - {metric}: None")
                
if __name__ == "__main__":
    run_xai_pipeline()
