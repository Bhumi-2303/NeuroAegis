from typing import TypeVar, Generic, Dict, List
from threading import RLock
from apps.api.app.core.exceptions import PluginNotFoundError, PluginRegistrationError
from apps.api.app.domain.protocols.plugin_protocol import PluginProtocol

T = TypeVar('T', bound=PluginProtocol)

class BaseRegistry(Generic[T]):
    """
    Thread-safe generic registry for managing plugins.
    Ensures that plugin instances can be registered, retrieved, and unregistered concurrently.
    """
    def __init__(self, registry_name: str = "BaseRegistry"):
        self.registry_name = registry_name
        self._plugins: Dict[str, T] = {}
        self._lock = RLock()
        
    def register(self, plugin_id: str, plugin_instance: T) -> None:
        """
        Register a new plugin instance.
        
        Args:
            plugin_id: Unique identifier for the plugin.
            plugin_instance: The instantiated plugin object.
            
        Raises:
            PluginRegistrationError: If a plugin with the given ID already exists.
        """
        with self._lock:
            if plugin_id in self._plugins:
                raise PluginRegistrationError(
                    f"Cannot register plugin '{plugin_id}' in {self.registry_name}: ID already exists."
                )
            self._plugins[plugin_id] = plugin_instance
            
    def get(self, plugin_id: str) -> T:
        """
        Retrieve a registered plugin by ID.
        
        Args:
            plugin_id: The unique identifier.
            
        Returns:
            T: The plugin instance.
            
        Raises:
            PluginNotFoundError: If the plugin is not found.
        """
        with self._lock:
            if plugin_id not in self._plugins:
                raise PluginNotFoundError(plugin_id=plugin_id, registry_name=self.registry_name)
            return self._plugins[plugin_id]
            
    def unregister(self, plugin_id: str) -> None:
        """
        Unregister a plugin by ID.
        
        Args:
            plugin_id: The unique identifier.
            
        Raises:
            PluginNotFoundError: If the plugin is not found.
        """
        with self._lock:
            if plugin_id not in self._plugins:
                raise PluginNotFoundError(plugin_id=plugin_id, registry_name=self.registry_name)
            del self._plugins[plugin_id]
            
    def has(self, plugin_id: str) -> bool:
        """Check if a plugin exists in the registry."""
        with self._lock:
            return plugin_id in self._plugins

    def list_all(self) -> List[str]:
        """List all registered plugin IDs."""
        with self._lock:
            return list(self._plugins.keys())
