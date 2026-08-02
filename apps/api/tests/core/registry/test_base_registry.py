import pytest
import threading
from typing import Dict, Any
from apps.api.app.core.exceptions import PluginNotFoundError, PluginRegistrationError
from apps.api.app.core.registry.base_registry import BaseRegistry
from apps.api.app.domain.protocols.plugin_protocol import PluginProtocol

class DummyPlugin(PluginProtocol):
    def get_metadata(self) -> Dict[str, Any]:
        return {"name": "dummy"}

def test_registry_registration_and_retrieval():
    registry = BaseRegistry[DummyPlugin]("TestRegistry")
    plugin = DummyPlugin()
    
    registry.register("dummy_1", plugin)
    assert registry.has("dummy_1")
    assert registry.get("dummy_1") is plugin

def test_registry_duplicate_registration_raises():
    registry = BaseRegistry[DummyPlugin]("TestRegistry")
    plugin = DummyPlugin()
    registry.register("dummy_1", plugin)
    
    with pytest.raises(PluginRegistrationError):
        registry.register("dummy_1", plugin)

def test_registry_get_missing_raises():
    registry = BaseRegistry[DummyPlugin]("TestRegistry")
    with pytest.raises(PluginNotFoundError):
        registry.get("missing_plugin")

def test_registry_unregister():
    registry = BaseRegistry[DummyPlugin]("TestRegistry")
    plugin = DummyPlugin()
    
    registry.register("dummy_1", plugin)
    registry.unregister("dummy_1")
    assert not registry.has("dummy_1")
    
    with pytest.raises(PluginNotFoundError):
        registry.unregister("dummy_1")

def test_registry_concurrent_access():
    """Verify that multiple threads can register plugins safely without race conditions."""
    registry = BaseRegistry[DummyPlugin]("TestRegistry")
    num_threads = 100
    
    def register_task(index: int):
        plugin = DummyPlugin()
        registry.register(f"plugin_{index}", plugin)

    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=register_task, args=(i,))
        threads.append(thread)
        thread.start()
        
    for thread in threads:
        thread.join()
        
    assert len(registry.list_all()) == num_threads
    for i in range(num_threads):
        assert registry.has(f"plugin_{i}")
