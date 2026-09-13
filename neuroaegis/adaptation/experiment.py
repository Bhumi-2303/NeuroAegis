import torch
import torch.nn as nn
from typing import List, Any
from .variants import VariantA_DirectTransfer, VariantB_Normalization, VariantC_Calibration, VariantD_Combined
from neuroaegis.guardrails.checks import assert_no_patient_leakage

def build_adapted_model(
    model: nn.Module, 
    variant: str, 
    calibration_patients: List[str], 
    test_patients: List[str], 
    data_loader_factory: Any
) -> nn.Module:
    """
    Builds the adapted model based on the variant (A, B, C, D).
    Enforces that calibration_patients and test_patients never overlap.
    """
    # Guardrail: Never tune/calibrate on the test evaluation set!
    assert_no_patient_leakage(set(calibration_patients), set(test_patients))
    
    # We only build the calibration loader if adaptation is active (B, C, D)
    if variant != "A":
        calibration_loader = data_loader_factory(calibration_patients, "calibration")
    else:
        calibration_loader = None
        
    if variant == "A":
        return VariantA_DirectTransfer(model)
    elif variant == "B":
        return VariantB_Normalization(model, calibration_loader)
    elif variant == "C":
        return VariantC_Calibration(model, calibration_loader)
    elif variant == "D":
        return VariantD_Combined(model, calibration_loader)
    else:
        raise ValueError(f"Unknown adaptation variant: {variant}")
