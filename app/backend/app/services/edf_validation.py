from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Literal

import mne
import yaml
from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.services.dataset_detection.detector import DatasetDetector, dataset_detector


DatasetId = Literal["bonn", "chbmit", "siena", "unknown"]
_NON_EEG_CHANNEL_NAME = re.compile(
    r"^(?:ECG|EKG|EOG|EMG|RESP|TRIG|STATUS|STI|EXG)(?:[-_ 0-9].*)?$",
    re.IGNORECASE,
)


class EdfValidationResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    validation_status: Literal["valid", "invalid"] = Field(alias="validationStatus")
    file_name: str = Field(alias="fileName")
    file_size_bytes: int = Field(alias="fileSizeBytes")
    sampling_rate: float | None = Field(default=None, alias="samplingRate")
    duration_seconds: float | None = Field(default=None, alias="durationSeconds")
    total_channels: int = Field(default=0, alias="totalChannels")
    eeg_channels: int = Field(default=0, alias="eegChannels")
    channel_names: list[str] = Field(default_factory=list, alias="channelNames")
    excluded_channels: list[str] = Field(default_factory=list, alias="excludedChannels")
    dataset: DatasetId = "unknown"
    detection_confidence: float = Field(default=0.0, alias="detectionConfidence")
    matched_rules: list[str] = Field(default_factory=list, alias="matchedRules")
    errors: list[str] = Field(default_factory=list)

    def response(self) -> dict:
        return self.model_dump(by_alias=True)


class DatasetSizePolicy(BaseModel):
    min_allowed_bytes: int | None = None
    max_allowed_bytes: int | None = None


class UploadValidationError(ValueError):
    def __init__(self, message: str, *, file_size_bytes: int = 0):
        super().__init__(message)
        self.file_size_bytes = file_size_bytes


def usable_eeg_channel_indices(raw) -> list[int]:
    """Return only EEG channels that are usable by the prediction pipeline."""
    channel_types = raw.get_channel_types(unique=False)
    return [
        index
        for index, (name, channel_type) in enumerate(zip(raw.ch_names, channel_types))
        if channel_type == "eeg" and not _NON_EEG_CHANNEL_NAME.match(name)
    ]


def sanitize_upload_filename(filename: str | None) -> str:
    """Validate the client name and return a safe display name."""
    raw_name = filename or ""
    normalized = unicodedata.normalize("NFKC", raw_name)
    if not normalized or "/" in normalized or "\\" in normalized:
        raise UploadValidationError("Filename must not contain path separators")
    if any(part == ".." for part in normalized.split(".")) or normalized in {".", ".."}:
        raise UploadValidationError("Path traversal is not allowed in filenames")
    if Path(normalized).suffix.lower() != ".edf":
        raise UploadValidationError("Only .edf files are supported")

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized).strip(".")
    if not safe_name or Path(safe_name).suffix.lower() != ".edf":
        raise UploadValidationError("Filename is not a valid EDF filename")
    return safe_name


async def save_upload_to_temp(
    upload: UploadFile,
    *,
    temp_root: str | Path,
    max_upload_size: int,
) -> tuple[Path, str, int]:
    """Stream an upload to a generated temp path without trusting its name."""
    safe_name = sanitize_upload_filename(upload.filename)
    root = Path(temp_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix="neuroaegis_", suffix=".edf", dir=root)
    path = Path(raw_path)
    total = 0
    try:
        with os.fdopen(fd, "wb") as destination:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_upload_size:
                    raise UploadValidationError(
                        "File exceeds the configured upload size limit",
                        file_size_bytes=total,
                    )
                destination.write(chunk)
        if total == 0:
            raise UploadValidationError("EDF file is empty", file_size_bytes=0)
        return path, safe_name, total
    except Exception:
        path.unlink(missing_ok=True)
        raise


def cleanup_temp_upload(path: Path) -> None:
    """Remove only the generated upload file and its generated temp directory."""
    path.unlink(missing_ok=True)
    parent = path.parent
    if parent.name.startswith("neuroaegis_edf_"):
        shutil.rmtree(parent, ignore_errors=True)


def _load_size_policies() -> dict[str, DatasetSizePolicy]:
    config_path = Path(settings.BASE_DIR) / "config" / "dataset_validation.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            values = yaml.safe_load(config_file) or {}
        return {
            dataset_id: DatasetSizePolicy(**policy)
            for dataset_id, policy in values.get("datasets", {}).items()
        }
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return {}


class EDFValidationService:
    """Validate EDF metadata without preloading signal samples."""

    def __init__(
        self,
        detector: DatasetDetector | None = None,
        *,
        max_upload_size: int | None = None,
        size_policies: dict[str, DatasetSizePolicy] | None = None,
    ):
        self.detector = detector or dataset_detector
        self.max_upload_size = max_upload_size if max_upload_size is not None else settings.MAX_EEG_UPLOAD_BYTES
        self.size_policies = size_policies if size_policies is not None else _load_size_policies()

    def validate_file(
        self,
        path: str | Path,
        *,
        file_name: str,
        file_size_bytes: int | None = None,
    ) -> EdfValidationResult:
        file_path = Path(path)
        actual_size = file_size_bytes
        if actual_size is None and file_path.exists():
            actual_size = file_path.stat().st_size
        actual_size = int(actual_size or 0)
        result = EdfValidationResult(
            validationStatus="invalid",
            fileName=file_name,
            fileSizeBytes=actual_size,
        )

        try:
            safe_name = sanitize_upload_filename(file_name)
        except UploadValidationError as exc:
            result.errors.append(str(exc))
            return result

        if not file_path.exists() or not file_path.is_file():
            result.errors.append("Uploaded file could not be found")
            return result
        if actual_size <= 0:
            result.errors.append("EDF file is empty")
            return result
        if actual_size > self.max_upload_size:
            result.errors.append("File exceeds the configured upload size limit")
            return result

        raw = None
        try:
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose="ERROR")
            sampling_rate = float(raw.info.get("sfreq", float("nan")))
            total_channels = len(raw.ch_names)
            duration_seconds = float(raw.n_times / sampling_rate) if sampling_rate > 0 else float("nan")
            channel_types = raw.get_channel_types(unique=False)
            excluded_channels = [
                name
                for name, channel_type in zip(raw.ch_names, channel_types)
                if channel_type != "eeg" or _NON_EEG_CHANNEL_NAME.match(name)
            ]
            eeg_channel_indices = usable_eeg_channel_indices(raw)
            eeg_channels = [raw.ch_names[index] for index in eeg_channel_indices]

            result.sampling_rate = sampling_rate if math.isfinite(sampling_rate) and sampling_rate > 0 else None
            result.duration_seconds = duration_seconds if math.isfinite(duration_seconds) and duration_seconds > 0 else None
            result.total_channels = total_channels
            result.eeg_channels = len(eeg_channels)
            result.channel_names = list(raw.ch_names)
            result.excluded_channels = excluded_channels

            if result.sampling_rate is None:
                result.errors.append("EDF metadata contains an invalid sampling rate")
            if result.duration_seconds is None:
                result.errors.append("EDF metadata contains an invalid recording duration")
            if total_channels <= 0:
                result.errors.append("EDF contains no channels")
            if not eeg_channels:
                result.errors.append("EDF contains no usable EEG channels")

            if not result.errors:
                dataset, confidence, matched_rules = self.detector.detect_recording(
                    file_name=safe_name,
                    total_channels=total_channels,
                    sampling_rate=result.sampling_rate or 0.0,
                    duration_seconds=result.duration_seconds or 0.0,
                )
                result.dataset = dataset if dataset in {"bonn", "chbmit", "siena"} else "unknown"
                result.detection_confidence = confidence
                result.matched_rules = matched_rules
                policy = self.size_policies.get(result.dataset)
                if policy:
                    if policy.min_allowed_bytes is not None and actual_size < policy.min_allowed_bytes:
                        result.errors.append("File is below the configured dataset size policy")
                    if policy.max_allowed_bytes is not None and actual_size > policy.max_allowed_bytes:
                        result.errors.append("File exceeds the configured dataset size policy")
        except Exception:
            result.errors.append("File is not a readable EDF recording")
        finally:
            if raw is not None:
                raw.close()

        if not result.errors:
            result.validation_status = "valid"
        return result


edf_validation_service = EDFValidationService()
