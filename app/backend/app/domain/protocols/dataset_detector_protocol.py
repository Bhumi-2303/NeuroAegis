from __future__ import annotations
from typing import Any, Protocol, runtime_checkable

from .plugin_protocol import PluginProtocol


@runtime_checkable
class DatasetDetectorProtocol(PluginProtocol, Protocol):
    """
    Protocol for dataset detection plugins.
    Analyzes raw file headers/snippets to identify the dataset format.
    """
    
    def detect(self, file_snippet: bytes) -> dict[str, Any] | None:
        """
        Identify if a file snippet matches this dataset format.
        
        Args:
            file_snippet (bytes): The first few kilobytes of the file.
            
        Returns:
            Optional[Dict[str, Any]]: Detection confidence and metadata, or None if no match.
        """
        ...
