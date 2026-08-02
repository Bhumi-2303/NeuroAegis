import yaml
from pathlib import Path
from typing import Dict, List
from .schemas import PipelineConfig

class ConfigurationManager:
    """
    Manages loading and validation of pipeline configurations from YAML files.
    """
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, PipelineConfig] = {}
        
    def load_all(self) -> None:
        """Scan the config directory and load all .yaml/.yml files."""
        if not self.config_dir.exists() or not self.config_dir.is_dir():
            raise FileNotFoundError(f"Config directory {self.config_dir} does not exist.")
            
        for ext in ("*.yaml", "*.yml"):
            for file_path in self.config_dir.glob(ext):
                self._load_file(file_path)
                
    def _load_file(self, file_path: Path) -> None:
        """Parse and validate a single YAML config file."""
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
            if not isinstance(data, dict):
                raise ValueError(f"Invalid YAML format in {file_path.name}. Expected a dictionary.")
            
            config = PipelineConfig(**data)
            self._configs[config.id] = config
            
    def get_pipeline(self, pipeline_id: str) -> PipelineConfig:
        """Retrieve a loaded pipeline configuration by ID."""
        if pipeline_id not in self._configs:
            raise KeyError(f"Pipeline config '{pipeline_id}' not found.")
        return self._configs[pipeline_id]
        
    def list_pipelines(self) -> List[str]:
        """List all loaded pipeline IDs."""
        return list(self._configs.keys())
