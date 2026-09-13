import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass
from neuroaegis.xai.frequency import extract_levels
from neuroaegis.xai.agreement import calculate_agreement_scores
from neuroaegis.guardrails.checks import label_attention_visualization

@dataclass
class ExplanationMap:
    event_id: str
    time_levels: Dict[str, torch.Tensor]
    channel_levels: Dict[str, torch.Tensor]
    frequency_levels: Dict[str, torch.Tensor]
    agreement_scores: Dict[str, Dict[str, float]]
    
    def display_warnings(self):
        """Auto-attaches warnings for unvalidated methods."""
        if "attention" in self.time_levels:
            print("WARNING [Attention]: unvalidated as a faithful explanation — see clinician-agreement study")

def build_comparison_harness(
    event_id: str,
    x: torch.Tensor,
    ig_attr: torch.Tensor,
    attention_attr: Optional[torch.Tensor] = None,
    shap_attr: Optional[torch.Tensor] = None
) -> ExplanationMap:
    """
    Produces a single 'explanation map' and pairwise agreement scores.
    """
    # 1. Extract levels for each available method
    methods_attr = {}
    methods_attr["ig"] = extract_levels(ig_attr, x)
    
    if attention_attr is not None:
        # Match dimensions if necessary (attention is usually time/channel based)
        methods_attr["attention"] = extract_levels(attention_attr, x)
        
    if shap_attr is not None:
        methods_attr["shap"] = extract_levels(shap_attr, x)
        
    # 2. Build Explanation Map components
    time_levels = {m: attr["time"] for m, attr in methods_attr.items()}
    channel_levels = {m: attr["channel"] for m, attr in methods_attr.items()}
    frequency_levels = {m: attr["frequency"] for m, attr in methods_attr.items()}
    
    # 3. Pairwise Agreement Scores
    agreement_scores = {}
    method_names = list(methods_attr.keys())
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            m1, m2 = method_names[i], method_names[j]
            pair_name = f"{m1}_vs_{m2}"
            scores = calculate_agreement_scores(methods_attr[m1], methods_attr[m2])
            agreement_scores[pair_name] = scores
            
    # Compile the final object
    explanation = ExplanationMap(
        event_id=event_id,
        time_levels=time_levels,
        channel_levels=channel_levels,
        frequency_levels=frequency_levels,
        agreement_scores=agreement_scores
    )
    
    return explanation
