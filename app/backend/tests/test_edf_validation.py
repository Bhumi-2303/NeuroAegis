from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import mne
import pytest
from fastapi import UploadFile

from app.services.edf_validation import (
    DatasetSizePolicy,
    EDFValidationService,
    EdfValidationResult,
    cleanup_temp_upload,
    sanitize_upload_filename,
    save_upload_to_temp,
)
from tests.edf_fixture import write_synthetic_edf


@pytest.fixture
def validator() -> EDFValidationService:
    return EDFValidationService(max_upload_size=10_000_000)


def test_valid_edf_returns_structured_metadata(tmp_path: Path, validator: EDFValidationService):
    path = write_synthetic_edf(tmp_path / "chb01_01.edf", channel_names=[f"Ch{i}" for i in range(23)])

    result = validator.validate_file(path, file_name=path.name)

    assert result.validation_status == "valid"
    assert result.dataset == "chbmit"
    assert result.sampling_rate == 256.0
    assert result.duration_seconds == 1.0
    assert result.total_channels == 23
    assert result.eeg_channels == 23
    assert result.errors == []


def test_validation_reads_only_edf_metadata(tmp_path: Path, validator: EDFValidationService, monkeypatch):
    path = write_synthetic_edf(tmp_path / "chb01_01.edf", channel_names=[f"Ch{i}" for i in range(23)])
    real_reader = mne.io.read_raw_edf
    read_options = {}

    def read_with_capture(*args, **kwargs):
        read_options.update(kwargs)
        return real_reader(*args, **kwargs)

    monkeypatch.setattr(mne.io, "read_raw_edf", read_with_capture)
    result = validator.validate_file(path, file_name=path.name)

    assert result.validation_status == "valid"
    assert read_options["preload"] is False


@pytest.mark.parametrize("file_name", ["record.csv", "record.txt", "record.json", "record.mat", "record.zip"])
def test_non_edf_extensions_are_rejected(file_name: str):
    with pytest.raises(ValueError, match=r"Only \.edf"):
        sanitize_upload_filename(file_name)


def test_fake_edf_extension_is_rejected(tmp_path: Path, validator: EDFValidationService):
    path = tmp_path / "fake.edf"
    path.write_bytes(b"not an EDF")

    result = validator.validate_file(path, file_name=path.name)

    assert result.validation_status == "invalid"
    assert result.errors == ["File is not a readable EDF recording"]


def test_empty_edf_is_rejected(tmp_path: Path, validator: EDFValidationService):
    path = tmp_path / "empty.edf"
    path.touch()

    result = validator.validate_file(path, file_name=path.name)

    assert result.validation_status == "invalid"
    assert "EDF file is empty" in result.errors


def test_corrupted_edf_is_rejected(tmp_path: Path, validator: EDFValidationService):
    path = tmp_path / "corrupt.edf"
    path.write_bytes(b"0       ".ljust(512, b"x"))

    result = validator.validate_file(path, file_name=path.name)

    assert result.validation_status == "invalid"
    assert "readable EDF" in result.errors[0]


def test_path_traversal_filename_is_rejected():
    with pytest.raises(ValueError, match="path separators"):
        sanitize_upload_filename("../../file.edf")
    with pytest.raises(ValueError, match="path separators"):
        sanitize_upload_filename(r"..\file.edf")


def test_valid_chbmit_edf_is_detected(tmp_path: Path, validator: EDFValidationService):
    path = write_synthetic_edf(tmp_path / "chb05_02.edf", channel_names=[f"Ch{i}" for i in range(23)])

    result = validator.validate_file(path, file_name=path.name)

    assert result.dataset == "chbmit"
    assert result.validation_status == "valid"


def test_valid_siena_edf_is_detected(tmp_path: Path, validator: EDFValidationService):
    path = write_synthetic_edf(
        tmp_path / "PN00-1.edf",
        channel_names=[f"EEG{i}" for i in range(35)],
        sampling_rate=512,
    )

    result = validator.validate_file(path, file_name=path.name)

    assert result.dataset == "siena"
    assert result.validation_status == "valid"


def test_valid_unknown_edf_is_not_assigned_to_a_dataset(tmp_path: Path, validator: EDFValidationService):
    path = write_synthetic_edf(
        tmp_path / "unknown_recording.edf",
        channel_names=["A", "B"],
        sampling_rate=100,
    )

    result = validator.validate_file(path, file_name=path.name)

    assert result.validation_status == "valid"
    assert result.dataset == "unknown"


def test_dataset_size_policy_rejects_below_and_above_and_accepts_valid_size(tmp_path: Path):
    path = write_synthetic_edf(tmp_path / "chb01_01.edf", channel_names=[f"Ch{i}" for i in range(23)])
    file_size = path.stat().st_size

    below = EDFValidationService(
        max_upload_size=10_000_000,
        size_policies={"chbmit": DatasetSizePolicy(min_allowed_bytes=file_size + 1)},
    ).validate_file(path, file_name=path.name)
    above = EDFValidationService(
        max_upload_size=10_000_000,
        size_policies={"chbmit": DatasetSizePolicy(max_allowed_bytes=file_size - 1)},
    ).validate_file(path, file_name=path.name)
    valid = EDFValidationService(
        max_upload_size=10_000_000,
        size_policies={"chbmit": DatasetSizePolicy(min_allowed_bytes=file_size, max_allowed_bytes=file_size)},
    ).validate_file(path, file_name=path.name)

    assert any("below the configured dataset size policy" in error for error in below.errors)
    assert any("exceeds the configured dataset size policy" in error for error in above.errors)
    assert valid.validation_status == "valid"


def test_transport_limit_accepts_below_and_exact_and_rejects_above(tmp_path: Path):
    path = write_synthetic_edf(tmp_path / "boundary.edf", channel_names=["Ch0"])
    limit = EDFValidationService().max_upload_size

    below = EDFValidationService().validate_file(
        path,
        file_name=path.name,
        file_size_bytes=limit - 1,
    )
    exact = EDFValidationService().validate_file(
        path,
        file_name=path.name,
        file_size_bytes=limit,
    )
    above = EDFValidationService().validate_file(
        path,
        file_name=path.name,
        file_size_bytes=limit + 1,
    )

    assert below.validation_status == "valid"
    assert exact.validation_status == "valid"
    assert above.validation_status == "invalid"
    assert above.errors == ["File exceeds the configured upload size limit"]


def test_corrupted_large_edf_is_rejected_after_size_check(tmp_path: Path):
    path = tmp_path / "corrupt_large.edf"
    limit = EDFValidationService().max_upload_size
    with path.open("wb") as file:
        file.write(b"not an EDF")
        file.truncate(limit - 1)

    result = EDFValidationService().validate_file(path, file_name=path.name)

    assert result.validation_status == "invalid"
    assert result.file_size_bytes == limit - 1
    assert result.errors == ["File is not a readable EDF recording"]


@pytest.mark.skipif(
    not Path("/Volumes/BLACK-BOX/NeuroAegis/CHB-MIT Dataset/chb04/chb04_27.edf").exists(),
    reason="local large CHB-MIT fixture is unavailable",
)
def test_local_large_chbmit_edf_is_validated_without_preloading():
    path = Path("/Volumes/BLACK-BOX/NeuroAegis/CHB-MIT Dataset/chb04/chb04_27.edf")

    result = EDFValidationService().validate_file(path, file_name=path.name)

    assert result.file_size_bytes == 177_285_376
    assert result.file_size_bytes > 100 * 1024 * 1024
    assert result.validation_status == "valid"
    assert result.dataset == "chbmit"


def test_above_transport_size_limit_is_rejected(tmp_path: Path):
    path = write_synthetic_edf(tmp_path / "chb01_01.edf", channel_names=["Ch0"])

    result = EDFValidationService(max_upload_size=path.stat().st_size - 1).validate_file(
        path,
        file_name=path.name,
    )

    assert result.validation_status == "invalid"
    assert "upload size limit" in result.errors[0]


def test_no_eeg_channels_is_rejected(tmp_path: Path, validator: EDFValidationService):
    path = write_synthetic_edf(tmp_path / "non_eeg.edf", channel_names=["ECG", "EOG"])

    result = validator.validate_file(path, file_name=path.name)

    assert result.validation_status == "invalid"
    assert any("no usable EEG channels" in error for error in result.errors)


def test_invalid_metadata_is_rejected(tmp_path: Path, validator: EDFValidationService, monkeypatch):
    path = write_synthetic_edf(tmp_path / "metadata.edf", channel_names=["Ch0"])

    class InvalidRaw:
        info = {"sfreq": float("nan")}
        ch_names = ["Ch0"]
        n_times = 1

        def get_channel_types(self, unique=False):
            return ["eeg"]

        def close(self):
            pass

    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *args, **kwargs: InvalidRaw())
    result = validator.validate_file(path, file_name=path.name)

    assert result.validation_status == "invalid"
    assert any("invalid sampling rate" in error for error in result.errors)


@pytest.mark.asyncio
async def test_temp_upload_cleanup_after_failure():
    with TemporaryDirectory(prefix="neuroaegis_edf_test_") as temp_dir:
        upload = UploadFile(filename="valid.edf", file=BytesIO(b"edf"))
        with pytest.raises(ValueError, match="upload size limit"):
            await save_upload_to_temp(upload, temp_root=temp_dir, max_upload_size=2)
        assert list(Path(temp_dir).iterdir()) == []


def test_cleanup_removes_generated_upload(tmp_path: Path):
    generated_dir = tmp_path / "neuroaegis_edf_generated"
    generated_dir.mkdir()
    upload_path = generated_dir / "generated.edf"
    upload_path.write_bytes(b"edf")

    cleanup_temp_upload(upload_path)

    assert not upload_path.exists()
