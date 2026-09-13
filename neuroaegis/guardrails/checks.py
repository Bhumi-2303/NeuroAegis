import hashlib
import json
import os
from typing import Set, Dict, Any, List

def assert_separate_namespaces(datasets: List[str]) -> None:
    """
    Enforces separate result namespaces (results/chbmit_siena/ vs results/bonn/).
    Raises ValueError if a script tries to merge them into one metrics table.
    """
    datasets_lower = [d.lower() for d in datasets]
    has_bonn = any('bonn' in d for d in datasets_lower)
    has_siena_or_chbmit = any('siena' in d or 'chbmit' in d for d in datasets_lower)
    
    if has_bonn and has_siena_or_chbmit:
        raise ValueError(
            "Namespace violation: Cannot merge Bonn and CHB-MIT/Siena datasets "
            "into the same metrics table."
        )

def assert_no_patient_leakage(train_ids: Set[str], test_ids: Set[str]) -> None:
    """
    Ensures that there is no patient leakage between train and test splits.
    Every training script must call this before .fit().
    """
    leakage = set(train_ids).intersection(set(test_ids))
    if leakage:
        raise ValueError(f"Patient leakage detected! Patients in both train and test: {leakage}")

def check_frozen(model_path: str, registry_path: str = None) -> bool:
    """
    Checks if a model checkpoint hash matches a 'frozen' registry entry.
    Refuses to run Siena eval if not registered.
    """
    if registry_path is None:
        # Default to FROZEN_REGISTRY.md in the current directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        registry_path = os.path.join(base_dir, "FROZEN_REGISTRY.md")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} not found.")
        
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Registry {registry_path} not found.")

    # Compute SHA256 of the model
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    model_hash = sha256_hash.hexdigest()

    with open(registry_path, "r") as f:
        registry_content = f.read()

    if model_hash not in registry_content:
        raise ValueError(
            f"Model hash {model_hash} not found in {registry_path}. "
            "Model must be frozen and registered before evaluation on Siena/clinician data."
        )
    return True

def format_metrics_report(metrics: Dict[str, float]) -> str:
    """
    Enforces a fixed print/report order for metrics.
    Accuracy is placed last and labeled as reference only.
    """
    required_metrics = ['sensitivity', 'event_sensitivity', 'fa_per_24h', 'auprc', 'accuracy']
    for req in required_metrics:
        if req not in metrics:
            raise ValueError(f"Missing required metric for reporting: {req}")

    report_lines = [
        f"Sensitivity: {metrics['sensitivity']:.4f}",
        f"Event-Sensitivity: {metrics['event_sensitivity']:.4f}",
        f"FA/24h: {metrics['fa_per_24h']:.4f}",
        f"AUPRC: {metrics['auprc']:.4f}"
    ]

    # Add any other metrics not in the strict ordering
    for k, v in metrics.items():
        if k not in required_metrics:
            if isinstance(v, float):
                report_lines.append(f"{k}: {v:.4f}")
            else:
                report_lines.append(f"{k}: {v}")

    # Accuracy must be last and explicitly labeled
    report_lines.append(f"Accuracy: {metrics['accuracy']:.4f} (reference only, not clinically meaningful)")
    
    return "\n".join(report_lines)

def false_alarms_per_24h(false_positives: int, total_duration_hours: float) -> float:
    """
    The only public FA function.
    Raw FP counts are a private intermediate and never surfaced in reports.
    """
    if total_duration_hours <= 0:
        raise ValueError("Total duration hours must be greater than zero.")
    return (false_positives / total_duration_hours) * 24.0

def label_attention_visualization(fig_or_ax: Any) -> None:
    """
    Auto-labels every attention visualization as unvalidated.
    Operates on a mocked matplotlib Figure or Axes object.
    """
    warning_text = "unvalidated — see IG/clinician agreement"
    
    if hasattr(fig_or_ax, 'set_title'):
        # Usually an Axes object
        current_title = fig_or_ax.get_title() if hasattr(fig_or_ax, 'get_title') else ""
        new_title = f"{current_title} | {warning_text}" if current_title else warning_text
        fig_or_ax.set_title(new_title)
    elif hasattr(fig_or_ax, 'suptitle'):
        # Usually a Figure object
        fig_or_ax.suptitle(warning_text)
    else:
        raise TypeError("fig_or_ax must be a matplotlib Figure or Axes object with set_title or suptitle.")

def load_frozen_postprocessing_config(config_path: str, expected_hash: str) -> Dict[str, Any]:
    """
    Loads detection-event parameters from a version-locked config.
    Ensures the config hash matches the expected frozen state.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config path {config_path} not found.")

    with open(config_path, "rb") as f:
        content_bytes = f.read()

    file_hash = hashlib.sha256(content_bytes).hexdigest()
    if file_hash != expected_hash:
        raise ValueError(
            f"Post-processing config hash mismatch. "
            f"Expected {expected_hash}, got {file_hash}. "
            "Config must be version-locked before Siena is touched."
        )

    return json.loads(content_bytes.decode('utf-8'))

class Normalizer:
    """
    Computes normalization statistics strictly per-fold.
    """
    def __init__(self, fold_id: str):
        self.fold_id = fold_id
        self.is_fitted = False

    def fit(self, fold_id: str, data: Any) -> 'Normalizer':
        if self.fold_id != fold_id:
            raise ValueError(
                f"Normalizer initialized for fold '{self.fold_id}', "
                f"but fit called on data for fold '{fold_id}'."
            )
        self.is_fitted = True
        return self
