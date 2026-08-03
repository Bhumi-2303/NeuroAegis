import pytest
from apps.api.app.core.registry.base_registry import BaseRegistry
from apps.api.app.core.registry.hub import RegistryHub


@pytest.fixture(autouse=True)
def reset_hub():
    """Ensure a clean singleton state before each test."""
    RegistryHub.reset()
    yield
    RegistryHub.reset()

def test_registry_hub_is_singleton():
    hub1 = RegistryHub()
    hub2 = RegistryHub()
    assert hub1 is hub2

def test_registry_hub_initializes_registries():
    hub = RegistryHub()
    
    # Ensure all expected registries are present and are instances of BaseRegistry
    assert isinstance(hub.models, BaseRegistry)
    assert isinstance(hub.features, BaseRegistry)
    assert isinstance(hub.explainers, BaseRegistry)
    assert isinstance(hub.datasets, BaseRegistry)
    
    # Check that they have the correct names
    assert hub.models.registry_name == "ModelRegistry"
    assert hub.features.registry_name == "FeatureRegistry"
    assert hub.explainers.registry_name == "ExplainerRegistry"
    assert hub.datasets.registry_name == "DatasetRegistry"
