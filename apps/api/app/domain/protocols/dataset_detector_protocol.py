from typing import Protocol, runtime_checkable, Dict, Any, Optional
from .plugin_protocol import PluginProtocol

@runtime_checkable
class DatasetDetectorProtocol(PluginProtocol, Protocol):
    """
    Protocol for dataset detection plugins.
    Analyzes raw file headers/snippets to identify the dataset format.
    """
    
    def detect(self, file_snippet: bytes) -> Optional[Dict[str, Any]]:
        """
        Identify if a file snippet matches this dataset format.
        
        Args:
            file_snippet (bytes): The first few kilobytes of the file.
            
        Returns:
            Optional[Dict[str, Any]]: Detection confidence and metadata, or None if no match.
        """
        ...
