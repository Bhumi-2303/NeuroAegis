import torch
import hashlib
import json
import os
from datetime import datetime
from neuroaegis.tracking.config import ExperimentConfig

class CheckpointManager:
    def __init__(self, save_dir: str = "checkpoints", registry_path: str = "neuroaegis/guardrails/FROZEN_REGISTRY.json"):
        self.save_dir = save_dir
        self.registry_path = registry_path
        os.makedirs(self.save_dir, exist_ok=True)
        
    def save_checkpoint(self, model: torch.nn.Module, epoch: int, config: ExperimentConfig) -> str:
        """
        Saves a checkpoint, hashes it, includes hash in filename, and updates registry.
        """
        # Save temporary file to compute hash
        tmp_path = os.path.join(self.save_dir, f"tmp_{config.experiment_name}_epoch_{epoch}.pt")
        torch.save(model.state_dict(), tmp_path)
        
        # Compute SHA256
        sha256_hash = hashlib.sha256()
        with open(tmp_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        model_hash = sha256_hash.hexdigest()
        
        # Final filename with hash
        final_filename = f"{config.experiment_name}_epoch_{epoch}_{model_hash[:12]}.pt"
        final_path = os.path.join(self.save_dir, final_filename)
        
        # Rename tmp to final
        os.rename(tmp_path, final_path)
        
        # Update registry mapping
        self._update_registry(model_hash, config, final_path)
        
        print(f"Checkpoint saved: {final_path}")
        return final_path
        
    def _update_registry(self, model_hash: str, config: ExperimentConfig, filepath: str):
        registry = {}
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                try:
                    registry = json.load(f)
                except json.JSONDecodeError:
                    pass
                    
        # Map hash -> config -> frozen (bool) -> frozen_at timestamp
        registry[model_hash] = {
            "filepath": filepath,
            "frozen": False, # Defaults to false, manual flip to freeze for clinical eval
            "frozen_at": None,
            "config": config.model_dump()
        }
        
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=4)
