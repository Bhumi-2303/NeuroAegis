from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import welch

from app.core.config import settings
from app.services.edf_validation import usable_eeg_channel_indices


MAX_VISUALIZATION_POINTS = 2000
REFERENCE_SECONDS = 10.0
ACTIVITY_CHUNK_SECONDS = 30.0


@dataclass(frozen=True)
class SeizureInterval:
    start_seconds: float
    end_seconds: float
    source: str = "dataset_annotation"

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def response(self) -> dict[str, Any]:
        return {
            "startSeconds": self.start_seconds,
            "endSeconds": self.end_seconds,
            "durationSeconds": self.duration_seconds,
            "source": self.source,
        }


@dataclass(frozen=True)
class AnnotationResult:
    intervals: tuple[SeizureInterval, ...]
    source: str | None
    available: bool


def _project_root() -> Path:
    # settings.BASE_DIR is app/backend in the supported local layout.
    return Path(settings.BASE_DIR).resolve().parents[1]


def _finite_interval(start: float, end: float, duration_seconds: float | None = None) -> SeizureInterval | None:
    if not math.isfinite(start) or not math.isfinite(end) or end <= start or start < 0:
        return None
    if duration_seconds is not None and end > duration_seconds + 1e-6:
        return None
    return SeizureInterval(float(start), float(end))


def parse_chbmit_summary(text: str, file_name: str, *, duration_seconds: float | None = None) -> AnnotationResult:
    """Parse one recording section from the official CHB-MIT summary format."""
    wanted = Path(file_name).name.casefold()
    sections = re.split(r"(?=^File Name:\s*)", text, flags=re.IGNORECASE | re.MULTILINE)
    for section in sections:
        file_match = re.search(r"^File Name:\s*(\S+)", section, flags=re.IGNORECASE | re.MULTILINE)
        if not file_match or Path(file_match.group(1)).name.casefold() != wanted:
            continue

        starts = re.findall(
            r"Seizure(?:\s+\d+)?\s+Start Time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds?",
            section,
            flags=re.IGNORECASE,
        )
        ends = re.findall(
            r"Seizure(?:\s+\d+)?\s+End Time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds?",
            section,
            flags=re.IGNORECASE,
        )
        declared_match = re.search(r"Number of Seizures in File:\s*(\d+)", section, flags=re.IGNORECASE)
        declared_count = int(declared_match.group(1)) if declared_match else None
        intervals = tuple(
            interval
            for start, end in zip(starts, ends)
            if (interval := _finite_interval(float(start), float(end), duration_seconds)) is not None
        )
        source_is_valid = not (declared_count and not intervals)
        return AnnotationResult(intervals, "chbmit_summary", source_is_valid)

    return AnnotationResult((), None, False)


def _clock_seconds(value: str) -> float | None:
    normalized = value.strip().replace(".", ":")
    parts = normalized.split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (float(part) for part in parts)
    except ValueError:
        return None
    if not (0 <= minutes < 60 and 0 <= seconds < 60):
        return None
    return hours * 3600 + minutes * 60 + seconds


def _relative_clock_seconds(registration: str, event: str) -> float | None:
    registration_seconds = _clock_seconds(registration)
    event_seconds = _clock_seconds(event)
    if registration_seconds is None or event_seconds is None:
        return None
    relative = event_seconds - registration_seconds
    if relative < 0:
        relative += 24 * 3600
    return relative


def parse_siena_seizure_list(text: str, file_name: str, *, duration_seconds: float | None = None) -> AnnotationResult:
    """Parse a Siena seizure list using registration-relative clock times."""
    wanted = Path(file_name).name.casefold()
    intervals: list[SeizureInterval] = []
    matched = False
    sections = re.split(r"(?=^Seizure\s+n\b)", text, flags=re.IGNORECASE | re.MULTILINE)
    for section in sections:
        file_match = re.search(r"^File name:\s*(\S+\.edf)", section, flags=re.IGNORECASE | re.MULTILINE)
        if not file_match or Path(file_match.group(1)).name.casefold() != wanted:
            continue
        matched = True
        registration_match = re.search(
            r"Registration start time:\s*([0-9:.]+)", section, flags=re.IGNORECASE
        )
        start_match = re.search(r"Seizure start time:\s*([0-9:.]+)", section, flags=re.IGNORECASE)
        end_match = re.search(r"Seizure end time:\s*([0-9:.]+)", section, flags=re.IGNORECASE)
        if not registration_match or not start_match or not end_match:
            continue
        start = _relative_clock_seconds(registration_match.group(1), start_match.group(1))
        end = _relative_clock_seconds(registration_match.group(1), end_match.group(1))
        if start is None or end is None:
            continue
        interval = _finite_interval(start, end, duration_seconds)
        if interval is not None:
            intervals.append(interval)

    return AnnotationResult(tuple(intervals), "siena_seizure_list" if matched else None, matched and bool(intervals))


def _read_siena_manifest(root: Path, file_name: str, duration_seconds: float | None) -> AnnotationResult:
    manifest = root / "research" / "experiments" / "siena" / "manifests" / "siena_seizure_events.csv"
    if not manifest.exists():
        return AnnotationResult((), None, False)
    wanted = Path(file_name).name.casefold()
    intervals: list[SeizureInterval] = []
    matched = False
    try:
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if Path(row.get("file_name", "")).name.casefold() != wanted:
                    continue
                matched = True
                interval = _finite_interval(
                    float(row["start_sec"]),
                    float(row["end_sec"]),
                    duration_seconds,
                )
                if interval is not None:
                    intervals.append(interval)
    except (OSError, KeyError, TypeError, ValueError):
        return AnnotationResult((), None, False)
    return AnnotationResult(tuple(intervals), "siena_manifest" if matched else None, matched and bool(intervals))


def resolve_dataset_annotations(
    dataset: str,
    file_name: str,
    *,
    duration_seconds: float | None = None,
    project_root: Path | None = None,
) -> AnnotationResult:
    """Resolve annotations from dataset-provided/local annotation metadata only."""
    root = project_root or _project_root()
    basename = Path(file_name).name

    if dataset == "chbmit":
        patient_match = re.search(r"(chb\d+)", basename, flags=re.IGNORECASE)
        if not patient_match:
            return AnnotationResult((), None, False)
        patient = patient_match.group(1).lower()
        candidates = (
            root / "CHB-MIT Dataset" / patient / f"{patient}-summary.txt",
            root / "data" / "chbmit_subset" / patient / f"{patient}-summary.txt",
        )
        for summary_path in candidates:
            if summary_path.exists():
                try:
                    return parse_chbmit_summary(
                        summary_path.read_text(encoding="utf-8"),
                        basename,
                        duration_seconds=duration_seconds,
                    )
                except OSError:
                    continue
        return AnnotationResult((), None, False)

    if dataset == "siena":
        patient_match = re.search(r"(PN\d+)", basename, flags=re.IGNORECASE)
        if not patient_match:
            return AnnotationResult((), None, False)
        patient = patient_match.group(1).upper()
        list_path = root / "data" / "siena-scalp-eeg" / patient / f"Seizures-list-{patient}.txt"
        if list_path.exists():
            try:
                result = parse_siena_seizure_list(
                    list_path.read_text(encoding="utf-8"),
                    basename,
                    duration_seconds=duration_seconds,
                )
                if result.available and result.intervals:
                    return result
                # The raw list can contain a clock typo or a record whose local
                # EDF is shorter than the listed registration. Use a validated
                # local manifest only when it still fits this EDF.
                fallback = _read_siena_manifest(root, basename, duration_seconds)
                if fallback.intervals:
                    return fallback
                return result
            except OSError:
                pass
        return _read_siena_manifest(root, basename, duration_seconds)

    return AnnotationResult((), None, False)


def _sample_indices(total_samples: int, max_points: int) -> np.ndarray:
    if total_samples <= 0:
        return np.empty(0, dtype=int)
    target = min(total_samples, max_points)
    if target == total_samples:
        return np.arange(total_samples, dtype=int)
    return np.unique(np.linspace(0, total_samples - 1, target).round().astype(int))


def read_sparse_samples(raw: Any, picks: list[int], indices: np.ndarray) -> np.ndarray:
    """Read selected samples in small contiguous blocks, never the full recording."""
    if not picks or indices.size == 0:
        return np.empty((len(picks), 0), dtype=float)
    result = np.empty((len(picks), indices.size), dtype=float)
    for offset in range(0, indices.size, 64):
        selected = indices[offset : offset + 64]
        start = int(selected[0])
        stop = int(selected[-1]) + 1
        block = raw.get_data(picks=picks, start=start, stop=stop)
        result[:, offset : offset + selected.size] = block[:, selected - start]
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _reference_bounds(
    duration_seconds: float,
    intervals: Iterable[SeizureInterval],
    *,
    reference_seconds: float = REFERENCE_SECONDS,
) -> tuple[float, float] | None:
    reference_length = min(duration_seconds, reference_seconds)
    if reference_length <= 0:
        return None
    sorted_intervals = sorted(intervals, key=lambda item: item.start_seconds)
    boundaries: list[tuple[float, float]] = []
    cursor = 0.0
    for interval in sorted_intervals:
        boundaries.append((cursor, interval.start_seconds))
        cursor = max(cursor, interval.end_seconds)
    boundaries.append((cursor, duration_seconds))
    for start, end in boundaries:
        if end - start >= reference_length:
            return start, start + reference_length
    return None


def _spectral_metrics(signal: np.ndarray, sampling_rate: float) -> dict[str, float]:
    values = np.asarray(signal, dtype=float)
    if values.size == 0:
        return {"rms": 0.0, "lineLength": 0.0, "variance": 0.0, "bandPower": 0.0, "spectralEntropy": 0.0}
    rms = float(np.sqrt(np.mean(np.square(values))))
    line_length = float(np.mean(np.abs(np.diff(values)))) if values.size > 1 else 0.0
    variance = float(np.var(values))
    nperseg = min(values.size, max(8, int(sampling_rate * 2)))
    frequencies, power = welch(values, fs=sampling_rate, nperseg=nperseg)
    band_mask = (frequencies >= 0.5) & (frequencies <= min(45.0, sampling_rate / 2.0))
    band_power = float(np.trapz(power[band_mask], frequencies[band_mask])) if np.any(band_mask) else 0.0
    total_power = float(np.sum(power))
    probabilities = power / total_power if total_power > 0 else np.zeros_like(power)
    entropy = float(-np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0])))
    return {
        "rms": rms,
        "lineLength": line_length,
        "variance": variance,
        "bandPower": band_power,
        "spectralEntropy": entropy,
    }


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return _spectral_metrics(np.empty(0), 1.0)
    return {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}


def _metrics_for_interval(
    raw: Any,
    picks: list[int],
    start_seconds: float,
    end_seconds: float,
    sampling_rate: float,
) -> list[dict[str, float]]:
    start_sample = max(0, int(round(start_seconds * sampling_rate)))
    end_sample = min(raw.n_times, max(start_sample + 1, int(round(end_seconds * sampling_rate))))
    chunk_samples = max(1, int(round(ACTIVITY_CHUNK_SECONDS * sampling_rate)))
    chunks: list[list[dict[str, float]]] = [[] for _ in picks]
    for chunk_start in range(start_sample, end_sample, chunk_samples):
        chunk_end = min(end_sample, chunk_start + chunk_samples)
        data = raw.get_data(picks=picks, start=chunk_start, stop=chunk_end)
        for channel_index, channel_data in enumerate(data):
            chunks[channel_index].append(_spectral_metrics(np.nan_to_num(channel_data), sampling_rate))
    return [_mean_metrics(channel_chunks) for channel_chunks in chunks]


def _channel_activity(
    raw: Any,
    picks: list[int],
    intervals: tuple[SeizureInterval, ...],
    reference_bounds: tuple[float, float],
    sampling_rate: float,
) -> list[dict[str, Any]]:
    reference_start, reference_end = reference_bounds
    response: list[dict[str, Any]] = []
    baseline_metrics = _metrics_for_interval(raw, picks, reference_start, reference_end, sampling_rate)
    seizure_metrics = [
        _metrics_for_interval(raw, picks, interval.start_seconds, interval.end_seconds, sampling_rate)
        for interval in intervals
    ]
    for channel_index, pick in enumerate(picks):
        baseline = baseline_metrics[channel_index]
        seizure = _mean_metrics([metrics[channel_index] for metrics in seizure_metrics])
        ratios = []
        for key in ("rms", "lineLength", "variance", "bandPower"):
            denominator = abs(baseline[key])
            if denominator > 1e-12:
                ratios.append(seizure[key] / denominator)
        score = float(np.mean(ratios)) if ratios else 0.0
        response.append(
            {
                "channelName": raw.ch_names[pick],
                "activityScore": score,
                "baselineScore": 1.0,
                "relativeChange": score - 1.0,
                "metrics": {
                    **seizure,
                    "baselineRms": baseline["rms"],
                    "baselineLineLength": baseline["lineLength"],
                    "baselineVariance": baseline["variance"],
                    "baselineBandPower": baseline["bandPower"],
                    "baselineSpectralEntropy": baseline["spectralEntropy"],
                    "referenceStartSeconds": reference_start,
                    "referenceEndSeconds": reference_end,
                },
            }
        )
    return response


def _channel_payload(raw: Any, picks: list[int], samples: np.ndarray, sampling_rate: float) -> list[dict[str, Any]]:
    return [
        {
            "id": f"eeg-{index + 1}",
            "name": raw.ch_names[pick],
            # MNE exposes EDF physical samples in volts; the existing UI labels
            # its traces in microvolts. Model inputs remain in their original units.
            "samples": (samples[index] * 1_000_000.0).tolist(),
            "unit": "uV",
            "samplingRate": float(sampling_rate),
        }
        for index, pick in enumerate(picks)
    ]


def build_eeg_visualization_from_raw(
    raw: Any,
    *,
    file_name: str,
    file_size_bytes: int,
    dataset: str,
    max_points: int = MAX_VISUALIZATION_POINTS,
) -> dict[str, Any]:
    """Build the normalized real-EEG contract from a lazy MNE Raw object."""
    sampling_rate = float(raw.info["sfreq"])
    duration_seconds = float(raw.n_times / sampling_rate)
    picks = usable_eeg_channel_indices(raw)
    indices = _sample_indices(raw.n_times, max_points)
    samples = read_sparse_samples(raw, picks, indices)
    annotations = resolve_dataset_annotations(
        dataset,
        file_name,
        duration_seconds=duration_seconds,
    )
    intervals = annotations.intervals
    reference_bounds = (
        _reference_bounds(duration_seconds, intervals)
        if annotations.available
        else None
    )
    reference_payload: dict[str, Any] | None = None
    if reference_bounds is not None:
        reference_start, reference_end = reference_bounds
        reference_indices = _sample_indices(
            max(1, int(round((reference_end - reference_start) * sampling_rate))),
            min(max_points, 1500),
        )
        reference_indices = reference_indices + int(round(reference_start * sampling_rate))
        reference_samples = read_sparse_samples(raw, picks, reference_indices)
        reference_payload = {
            "available": True,
            "startSeconds": reference_start,
            "endSeconds": reference_end,
            "durationSeconds": reference_end - reference_start,
            "originalSampleCount": int(reference_indices.size),
            "visualizationSampleCount": int(reference_indices.size),
            "channels": _channel_payload(raw, picks, reference_samples, sampling_rate),
        }

    channel_types = raw.get_channel_types(unique=False)
    excluded_details = []
    for index, (name, channel_type) in enumerate(zip(raw.ch_names, channel_types)):
        if index in picks:
            continue
        reason = "non_eeg_channel" if channel_type != "eeg" else "excluded_by_channel_name"
        excluded_details.append({"name": name, "reason": reason})

    patient_match = re.search(r"(chb\d+|PN\d+)", Path(file_name).stem, flags=re.IGNORECASE)
    patient_identifier = patient_match.group(1).upper() if patient_match else None
    activity = (
        _channel_activity(raw, picks, intervals, reference_bounds, sampling_rate)
        if intervals and reference_bounds
        else []
    )
    return {
        "dataset": dataset,
        "fileName": Path(file_name).name,
        "fileSizeBytes": int(file_size_bytes),
        "patientIdentifier": patient_identifier,
        "samplingRate": sampling_rate,
        "durationSeconds": duration_seconds,
        "totalChannels": len(raw.ch_names),
        "eegChannelCount": len(picks),
        "channels": _channel_payload(raw, picks, samples, sampling_rate),
        "visualizationSampleCount": int(indices.size),
        "originalSampleCount": int(raw.n_times),
        "timeStartSeconds": 0.0,
        "timeEndSeconds": max(0.0, (raw.n_times - 1) / sampling_rate),
        "seizures": [interval.response() for interval in intervals],
        "hasSeizureAnnotations": bool(intervals),
        "annotationStatus": "available" if annotations.available else "unavailable",
        "annotationSource": annotations.source,
        "referenceAvailable": reference_payload is not None,
        "reference": reference_payload,
        "channelActivity": activity,
        "channelActivityAvailable": bool(activity),
        "channelActivityNote": (
            "Signal-derived channel activity. This score is calculated from EEG signal characteristics during the annotated seizure interval. It is not a channel-level model prediction."
            if activity
            else "No seizure annotation available for channel activity analysis."
        ),
        "excludedChannels": [item["name"] for item in excluded_details],
        "excludedChannelDetails": excluded_details,
    }


def build_window_visualization(eeg_data: Any, channel_names: list[str], fs: float, max_points: int = 1500) -> dict[str, Any]:
    """Compatibility helper for callers that only have the model window."""
    data = np.asarray(eeg_data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        raise ValueError(f"EEG visualization expects 2D data, got shape {data.shape}")
    indices = _sample_indices(data.shape[1], max_points)
    return {
        "dataset": "unknown",
        "fileName": "",
        "fileSizeBytes": 0,
        "patientIdentifier": None,
        "samplingRate": float(fs),
        "durationSeconds": float(data.shape[1] / fs) if fs else 0.0,
        "totalChannels": int(data.shape[0]),
        "eegChannelCount": int(data.shape[0]),
        "channels": [
            {
                "id": f"eeg-{index + 1}",
                "name": channel_names[index] if index < len(channel_names) else f"CH{index + 1}",
                "samples": (np.nan_to_num(data[index, indices]) * 1_000_000.0).tolist(),
                "unit": "uV",
                "samplingRate": float(fs),
            }
            for index in range(data.shape[0])
        ],
        "visualizationSampleCount": int(indices.size),
        "originalSampleCount": int(data.shape[1]),
        "timeStartSeconds": 0.0,
        "timeEndSeconds": float((data.shape[1] - 1) / fs) if fs and data.shape[1] else 0.0,
        "seizures": [],
        "hasSeizureAnnotations": False,
        "annotationStatus": "unavailable",
        "annotationSource": None,
        "referenceAvailable": False,
        "reference": None,
        "channelActivity": [],
        "channelActivityAvailable": False,
        "channelActivityNote": "No seizure annotation available for channel activity analysis.",
        "excludedChannels": [],
        "excludedChannelDetails": [],
    }
