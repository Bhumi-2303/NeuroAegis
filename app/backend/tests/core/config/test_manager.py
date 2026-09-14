
import pytest
import yaml
from apps.api.app.core.config.manager import ConfigurationManager
from pydantic import ValidationError


@pytest.fixture
def valid_yaml_content():
    return {
        "id": "bonn_seizure_pipeline",
        "task": "seizure_detection",
        "dataset": "bonn",
        "model": {
            "plugin_id": "lightgbm_bonn",
            "params": {"batch_size": 32}
        },
        "feature_extractors": [
            {
                "plugin_id": "time_domain_features"
            }
        ],
        "explainer": {
            "plugin_id": "shap_tree"
        }
    }

@pytest.fixture
def temp_config_dir(tmp_path, valid_yaml_content):
    config_dir = tmp_path / "pipelines"
    config_dir.mkdir()
    
    file_path = config_dir / "bonn.yaml"
    with file_path.open("w") as f:
        yaml.dump(valid_yaml_content, f)
        
    return config_dir

def test_configuration_manager_load_valid(temp_config_dir):
    manager = ConfigurationManager(str(temp_config_dir))
    manager.load_all()
    
    assert "bonn_seizure_pipeline" in manager.list_pipelines()
    
    pipeline = manager.get_pipeline("bonn_seizure_pipeline")
    assert pipeline.id == "bonn_seizure_pipeline"
    assert pipeline.model.plugin_id == "lightgbm_bonn"
    assert pipeline.model.params["batch_size"] == 32
    assert len(pipeline.feature_extractors) == 1

def test_configuration_manager_invalid_schema(tmp_path):
    config_dir = tmp_path / "pipelines"
    config_dir.mkdir()
    
    # Missing model
    invalid_content = {
        "id": "bad_pipeline",
        "task": "seizure_detection",
        "dataset": "bonn",
        "feature_extractors": [{"plugin_id": "foo"}]
    }
    
    with (config_dir / "bad.yaml").open("w") as f:
        yaml.dump(invalid_content, f)
        
    manager = ConfigurationManager(str(config_dir))
    
    with pytest.raises(ValidationError):
        manager.load_all()

def test_configuration_manager_missing_directory():
    manager = ConfigurationManager("/path/does/not/exist")
    with pytest.raises(FileNotFoundError):
        manager.load_all()

def test_configuration_manager_not_found(temp_config_dir):
    manager = ConfigurationManager(str(temp_config_dir))
    manager.load_all()
    
    with pytest.raises(KeyError):
        manager.get_pipeline("non_existent_pipeline")
