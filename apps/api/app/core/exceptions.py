class NeuroAegisError(Exception):
    """Base exception for all NeuroAegis platform errors."""
    pass

class PluginError(NeuroAegisError):
    """Base exception for plugin-related errors."""
    pass

class PluginNotFoundError(PluginError):
    """Raised when a requested plugin cannot be found in the registry."""
    def __init__(self, plugin_id: str, registry_name: str = "Registry"):
        super().__init__(f"Plugin '{plugin_id}' not found in {registry_name}.")
        self.plugin_id = plugin_id
        self.registry_name = registry_name

class PluginRegistrationError(PluginError):
    """Raised when a plugin fails to register (e.g., due to duplicate key)."""
    pass
