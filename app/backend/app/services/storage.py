from __future__ import annotations
import json
import logging
import os
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings

logger = logging.getLogger("neuroaegis.storage")

# Safe job identifier pattern (UUID or alphanumeric with hyphens/underscores)
SAFE_JOB_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")


class StorageError(Exception):
    """Base exception for staged storage operations."""
    pass


class StorageNotFoundError(StorageError):
    """Raised when a requested staged window does not exist."""
    pass


class StorageSecurityError(StorageError):
    """Raised when an unsafe path or path traversal attempt is detected."""
    pass


class StorageBackend(ABC):
    """Abstract interface for staging and retrieving prediction window packages."""

    @abstractmethod
    def save_window(
        self,
        job_id: str,
        eeg_data: np.ndarray,
        channel_names: list[str],
        fs: float,
        dataset_name: str,
        eeg_visualization: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Atomically save an inference window package to persistent storage.
        Returns the resolved file path/reference string.
        """
        pass

    @abstractmethod
    def load_window(
        self,
        staged_reference: str,
    ) -> tuple[np.ndarray, list[str], float, str, dict[str, Any] | None, dict[str, Any] | None]:
        """
        Load a staged window package.
        Returns: (eeg_data, channel_names, fs, dataset_name, eeg_visualization, metadata)
        """
        pass

    @abstractmethod
    def delete_window(self, staged_reference: str) -> bool:
        """
        Remove a staged window package.
        Returns True if deleted, False if not found.
        """
        pass


class LocalStorageBackend(StorageBackend):
    """
    Local filesystem storage backend for the shared Docker volume.
    Enforces atomic creation (temp file -> fsync -> rename) and path traversal security.
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self.storage_dir = Path(storage_dir or settings.STORAGE_DIR).resolve()
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(f"Could not initialize storage directory {self.storage_dir}: {exc}")

    def _resolve_and_verify_path(self, staged_reference: str) -> Path:
        """
        Verify that the staged reference resolves safely within self.storage_dir.
        Guards against directory traversal (../) and user-supplied malicious paths.
        """
        if not staged_reference or not isinstance(staged_reference, str):
            raise StorageError("Invalid staged reference: must be a non-empty string")

        ref_path = Path(staged_reference)
        if ref_path.is_absolute():
            resolved = ref_path.resolve()
        else:
            resolved = (self.storage_dir / ref_path).resolve()

        # Strict containment check
        try:
            resolved.relative_to(self.storage_dir)
        except ValueError as exc:
            raise StorageSecurityError(
                f"Path traversal detected: reference '{staged_reference}' escapes storage directory '{self.storage_dir}'"
            ) from exc

        return resolved

    def save_window(
        self,
        job_id: str,
        eeg_data: np.ndarray,
        channel_names: list[str],
        fs: float,
        dataset_name: str,
        eeg_visualization: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Atomically write canonical .npz package.
        Step 1: Write to unique temp file in storage directory.
        Step 2: Flush and fsync buffer to storage device.
        Step 3: Atomic rename (os.replace) to final target path.
        """
        if not job_id or not isinstance(job_id, str):
            raise StorageError("job_id must be a non-empty string identifier")

        if not SAFE_JOB_ID_REGEX.match(job_id):
            raise StorageSecurityError(f"Unsafe job_id format: '{job_id}' contains invalid characters")

        if not isinstance(eeg_data, np.ndarray):
            raise StorageError(f"eeg_data must be a NumPy array, got {type(eeg_data).__name__}")

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.storage_dir / f"{job_id}.npz"

        # Unique temporary file within the exact same filesystem volume for atomic rename
        temp_file = tempfile.NamedTemporaryFile(
            dir=str(self.storage_dir),
            prefix=f".tmp_{job_id}_",
            suffix=".npz",
            delete=False,
        )
        temp_path = Path(temp_file.name)

        try:
            # Canonical Staged Prediction Payload
            np.savez(
                temp_file,
                eeg_data=eeg_data.astype(np.float32),
                channel_names=np.array(channel_names, dtype=str),
                fs=np.float64(fs),
                dataset_name=str(dataset_name),
                eeg_visualization_json=json.dumps(eeg_visualization) if eeg_visualization else "",
                metadata_json=json.dumps(metadata) if metadata else "{}",
            )
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()

            # Atomic swap
            os.replace(str(temp_path), str(target_path))
            logger.info(f"Atomically staged prediction window for job {job_id} at {target_path}")
            return str(target_path)

        except Exception as exc:
            temp_file.close()
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            logger.error(f"Failed to atomically write staged window for {job_id}: {exc}", exc_info=True)
            raise StorageError(f"Failed to save staged window package for {job_id}: {exc}") from exc

    def load_window(
        self,
        staged_reference: str,
    ) -> tuple[np.ndarray, list[str], float, str, dict[str, Any] | None, dict[str, Any] | None]:
        """
        Safely load canonical staged window package.
        """
        resolved_path = self._resolve_and_verify_path(staged_reference)

        if not resolved_path.exists():
            raise StorageNotFoundError(f"Staged window package not found at: {resolved_path}")

        try:
            with np.load(str(resolved_path), allow_pickle=False) as npz:
                if "eeg_data" not in npz:
                    raise StorageError("Invalid staged package: missing 'eeg_data' key")

                eeg_data = npz["eeg_data"]
                channel_names = [str(ch) for ch in npz["channel_names"]] if "channel_names" in npz else []
                fs = float(npz["fs"]) if "fs" in npz else 256.0
                dataset_name = str(npz["dataset_name"]) if "dataset_name" in npz else "bonn"

                eeg_vis = None
                if "eeg_visualization_json" in npz:
                    vis_raw = str(npz["eeg_visualization_json"]).strip()
                    if vis_raw:
                        try:
                            eeg_vis = json.loads(vis_raw)
                        except Exception:
                            logger.warning(f"Could not parse eeg_visualization_json from {resolved_path}")

                metadata = None
                if "metadata_json" in npz:
                    meta_raw = str(npz["metadata_json"]).strip()
                    if meta_raw:
                        try:
                            metadata = json.loads(meta_raw)
                        except Exception:
                            logger.warning(f"Could not parse metadata_json from {resolved_path}")

                return eeg_data, channel_names, fs, dataset_name, eeg_vis, metadata

        except StorageError:
            raise
        except Exception as exc:
            logger.error(f"Failed to load staged package from {resolved_path}: {exc}", exc_info=True)
            raise StorageError(f"Failed to load or parse staged package: {exc}") from exc

    def delete_window(self, staged_reference: str) -> bool:
        """
        Safely delete a staged window package.
        """
        try:
            resolved_path = self._resolve_and_verify_path(staged_reference)
        except StorageSecurityError:
            raise
        except Exception:
            return False

        if resolved_path.exists():
            try:
                resolved_path.unlink()
                logger.info(f"Cleaned up staged window: {resolved_path}")
                return True
            except Exception as exc:
                logger.warning(f"Failed to delete staged window {resolved_path}: {exc}")
                return False
        return False


# Default storage backend singleton
storage_backend = LocalStorageBackend()
