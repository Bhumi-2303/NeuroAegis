import os
import tempfile
import hashlib
import json
import pytest
from unittest.mock import MagicMock

from neuroaegis.guardrails.checks import (
    assert_separate_namespaces,
    assert_no_patient_leakage,
    check_frozen,
    format_metrics_report,
    false_alarms_per_24h,
    label_attention_visualization,
    load_frozen_postprocessing_config,
    Normalizer,
)

def test_assert_separate_namespaces():
    # Valid cases
    assert_separate_namespaces(["results/chbmit_siena/metrics.csv", "results/other/metrics.csv"])
    assert_separate_namespaces(["results/bonn/metrics.csv"])
    
    # Invalid case: combining Bonn and CHBMIT/Siena
    with pytest.raises(ValueError, match="Namespace violation"):
        assert_separate_namespaces(["results/chbmit_siena/", "results/bonn/"])

def test_assert_no_patient_leakage():
    # Valid
    assert_no_patient_leakage({"pt1", "pt2"}, {"pt3", "pt4"})
    
    # Invalid
    with pytest.raises(ValueError, match="Patient leakage detected"):
        assert_no_patient_leakage({"pt1", "pt2"}, {"pt2", "pt3"})

def test_check_frozen():
    # We will create a temporary model file and a mock registry
    with tempfile.TemporaryDirectory() as tmpdir:
        model_content = b"mock model checkpoint data"
        model_hash = hashlib.sha256(model_content).hexdigest()
        
        model_path = os.path.join(tmpdir, "model.pt")
        with open(model_path, "wb") as f:
            f.write(model_content)
            
        registry_path = os.path.join(tmpdir, "FROZEN_REGISTRY.md")
        with open(registry_path, "w") as f:
            f.write(f"| mock_model | {model_hash} | 2026-09-13 |\n")
            
        # Should pass
        assert check_frozen(model_path, registry_path) == True
        
        # Now an unregistered model
        unregistered_path = os.path.join(tmpdir, "unregistered.pt")
        with open(unregistered_path, "wb") as f:
            f.write(b"something else entirely")
            
        with pytest.raises(ValueError, match="not found in"):
            check_frozen(unregistered_path, registry_path)

def test_format_metrics_report():
    metrics = {
        "sensitivity": 0.95,
        "event_sensitivity": 0.90,
        "fa_per_24h": 0.5,
        "auprc": 0.85,
        "accuracy": 0.98,
        "f1_score": 0.92
    }
    
    report = format_metrics_report(metrics)
    lines = report.split("\n")
    
    assert lines[0].startswith("Sensitivity")
    assert lines[1].startswith("Event-Sensitivity")
    assert lines[2].startswith("FA/24h")
    assert lines[3].startswith("AUPRC")
    
    # f1_score should be in the middle somewhere
    assert any("f1_score" in line for line in lines[4:-1])
    
    # accuracy MUST be last and contain the warning
    assert lines[-1].startswith("Accuracy")
    assert "(reference only, not clinically meaningful)" in lines[-1]
    
    # Missing required metric
    with pytest.raises(ValueError, match="Missing required metric"):
        format_metrics_report({"sensitivity": 0.95})

def test_false_alarms_per_24h():
    assert false_alarms_per_24h(5, 48.0) == 2.5
    assert false_alarms_per_24h(10, 12.0) == 20.0
    
    with pytest.raises(ValueError, match="greater than zero"):
        false_alarms_per_24h(5, 0)

def test_label_attention_visualization():
    mock_ax = MagicMock()
    mock_ax.get_title.return_value = "Original Title"
    
    label_attention_visualization(mock_ax)
    mock_ax.set_title.assert_called_with("Original Title | unvalidated — see IG/clinician agreement")
    
    mock_fig = MagicMock()
    del mock_fig.set_title # Remove set_title so it falls back to suptitle
    
    label_attention_visualization(mock_fig)
    mock_fig.suptitle.assert_called_with("unvalidated — see IG/clinician agreement")

def test_load_frozen_postprocessing_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        config_data = {"min_duration": 5, "merge_window": 10}
        content = json.dumps(config_data).encode('utf-8')
        
        with open(config_path, "wb") as f:
            f.write(content)
            
        expected_hash = hashlib.sha256(content).hexdigest()
        
        # Valid load
        loaded = load_frozen_postprocessing_config(config_path, expected_hash)
        assert loaded == config_data
        
        # Hash mismatch (tuned after seeing results)
        with pytest.raises(ValueError, match="hash mismatch"):
            load_frozen_postprocessing_config(config_path, "bad_hash_123")

def test_normalizer():
    norm = Normalizer("fold_1")
    
    # Valid fit
    norm.fit("fold_1", [1, 2, 3])
    assert norm.is_fitted
    
    # Invalid fit (e.g. forgot per-fold and passed a different fold)
    with pytest.raises(ValueError, match="Normalizer initialized for fold 'fold_1'"):
        norm.fit("fold_2", [4, 5, 6])
