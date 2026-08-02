from typing import Protocol, runtime_checkable, Dict, Any

@runtime_checkable
class PluginProtocol(Protocol):
    """
    Base protocol for all NeuroAegis V2 plugins.
    Ensures every plugin exposes standard metadata for the dynamic registry.
    """
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieve metadata about the plugin (e.g., version, author, compatibility).
        
        Returns:
            Dict[str, Any]: A dictionary containing plugin specifications.
        """
        ...
