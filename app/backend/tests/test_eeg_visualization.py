from __future__ import annotations

from pathlib import Path

import mne
import pytest

from app.services import eeg_visualization as visualization
from app.services.eeg_visualization import (
    AnnotationResult,
    SeizureInterval,
    build_eeg_visualization_from_raw,
    parse_chbmit_summary,
    parse_siena_seizure_list,
)
from tests.edf_fixture import write_synthetic_edf


def _open_fixture(tmp_path: Path, channels: list[str], *, duration: int = 12):
    path = write_synthetic_edf(
        tmp_path / "unknown_recording.edf",
        channel_names=channels,
        duration_seconds=duration,
    )
    return path, mne.io.read_raw_edf(path, preload=False, verbose="ERROR")


@pytest.mark.parametrize("channel_count", [1, 16, 23, 24, 26])
def test_visualization_preserves_dynamic_channel_count(tmp_path: Path, channel_count: int):
    names = [f"EEG-{index + 1}" for index in range(channel_count)]
    path, raw = _open_fixture(tmp_path, names)
    try:
        result = build_eeg_visualization_from_raw(
            raw,
            file_name=path.name,
            file_size_bytes=path.stat().st_size,
            dataset="unknown",
        )
    finally:
        raw.close()

    assert result["totalChannels"] == channel_count
    assert result["eegChannelCount"] == channel_count
    assert [channel["name"] for channel in result["channels"]] == names


def test_visualization_is_bounded_and_preserves_original_sample_count(tmp_path: Path):
    path, raw = _open_fixture(tmp_path, ["Fp1", "F3"], duration=12)
    try:
        result = build_eeg_visualization_from_raw(
            raw,
            file_name=path.name,
            file_size_bytes=path.stat().st_size,
            dataset="unknown",
            max_points=1500,
        )
    finally:
        raw.close()

    assert result["originalSampleCount"] == 12 * 256
    assert result["visualizationSampleCount"] == 1500
    assert all(len(channel["samples"]) == 1500 for channel in result["channels"])


def test_channel_names_and_actual_samples_are_preserved(tmp_path: Path):
    path, raw = _open_fixture(tmp_path, ["Fp1", "F3"], duration=1)
    try:
        expected = raw.get_data(picks=[0, 1], start=0, stop=raw.n_times)
        result = build_eeg_visualization_from_raw(
            raw,
            file_name=path.name,
            file_size_bytes=path.stat().st_size,
            dataset="unknown",
        )
    finally:
        raw.close()

    assert [channel["name"] for channel in result["channels"]] == ["Fp1", "F3"]
    assert result["channels"][0]["samples"] == pytest.approx((expected[0] * 1_000_000.0).tolist())


def test_non_eeg_channels_are_excluded_with_reason(tmp_path: Path):
    path, raw = _open_fixture(tmp_path, ["Fp1", "EKG 1", "EOG"], duration=1)
    try:
        result = build_eeg_visualization_from_raw(
            raw,
            file_name=path.name,
            file_size_bytes=path.stat().st_size,
            dataset="unknown",
        )
    finally:
        raw.close()

    assert result["eegChannelCount"] == 1
    assert result["excludedChannels"] == ["EKG 1", "EOG"]
    assert {item["name"] for item in result["excludedChannelDetails"]} == {"EKG 1", "EOG"}


def test_chbmit_annotation_parser_supports_zero_and_multiple_intervals():
    text = """
File Name: chb01_01.edf
Number of Seizures in File: 0

File Name: chb01_02.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 10 seconds
Seizure 1 End Time: 20 seconds
Seizure 2 Start Time: 40 seconds
Seizure 2 End Time: 45 seconds
"""

    zero = parse_chbmit_summary(text, "chb01_01.edf", duration_seconds=60)
    multiple = parse_chbmit_summary(text, "chb01_02.edf", duration_seconds=60)

    assert zero.available is True
    assert zero.intervals == ()
    assert [(item.start_seconds, item.end_seconds) for item in multiple.intervals] == [(10.0, 20.0), (40.0, 45.0)]


def test_siena_annotation_parser_uses_registration_relative_times():
    text = """
Seizure n 1
File name: PN00-2.edf
Registration start time: 02.18.17
Seizure start time: 02.38.37
Seizure end time: 02.39.31
"""

    result = parse_siena_seizure_list(text, "PN00-2.edf", duration_seconds=2302)

    assert result.available is True
    assert [(item.start_seconds, item.end_seconds) for item in result.intervals] == [(1220.0, 1274.0)]


def test_reference_segment_is_genuine_and_activity_is_signal_derived(tmp_path: Path, monkeypatch):
    path, raw = _open_fixture(tmp_path, ["Fp1", "F3"], duration=30)
    monkeypatch.setattr(
        visualization,
        "resolve_dataset_annotations",
        lambda *args, **kwargs: AnnotationResult((SeizureInterval(20.0, 21.0),), "test_annotation", True),
    )
    try:
        result = build_eeg_visualization_from_raw(
            raw,
            file_name=path.name,
            file_size_bytes=path.stat().st_size,
            dataset="unknown",
        )
    finally:
        raw.close()

    assert result["hasSeizureAnnotations"] is True
    assert result["seizures"] == [{
        "startSeconds": 20.0,
        "endSeconds": 21.0,
        "durationSeconds": 1.0,
        "source": "dataset_annotation",
    }]
    assert result["referenceAvailable"] is True
    assert result["reference"]["startSeconds"] == 0.0
    assert result["reference"]["endSeconds"] == 10.0
    assert result["channelActivityAvailable"] is True
    assert len(result["channelActivity"]) == 2
    assert "not a channel-level model prediction" in result["channelActivityNote"]


def test_no_annotation_has_no_reference_or_channel_activity(tmp_path: Path):
    path, raw = _open_fixture(tmp_path, ["Fp1"], duration=12)
    try:
        result = build_eeg_visualization_from_raw(
            raw,
            file_name=path.name,
            file_size_bytes=path.stat().st_size,
            dataset="unknown",
        )
    finally:
        raw.close()

    assert result["hasSeizureAnnotations"] is False
    assert result["referenceAvailable"] is False
    assert result["channelActivity"] == []
    assert result["channelActivityNote"] == "No seizure annotation available for channel activity analysis."


def test_reference_is_unavailable_when_seizure_covers_recording(tmp_path: Path, monkeypatch):
    path, raw = _open_fixture(tmp_path, ["Fp1"], duration=12)
    monkeypatch.setattr(
        visualization,
        "resolve_dataset_annotations",
        lambda *args, **kwargs: AnnotationResult((SeizureInterval(0.0, 12.0),), "test_annotation", True),
    )
    try:
        result = build_eeg_visualization_from_raw(
            raw,
            file_name=path.name,
            file_size_bytes=path.stat().st_size,
            dataset="unknown",
        )
    finally:
        raw.close()

    assert result["referenceAvailable"] is False
    assert result["channelActivityAvailable"] is False
