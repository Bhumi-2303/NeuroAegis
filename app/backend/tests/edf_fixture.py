from __future__ import annotations

import struct
from pathlib import Path


def _field(value: str, width: int) -> bytes:
    return value.encode("ascii")[:width].ljust(width, b" ")


def write_synthetic_edf(
    path: Path,
    *,
    channel_names: list[str],
    sampling_rate: int = 256,
    duration_seconds: int = 1,
) -> Path:
    """Write a minimal valid EDF fixture; it contains no real EEG recording."""
    samples_per_record = sampling_rate * duration_seconds
    number_of_signals = len(channel_names)
    header_bytes = 256 + (256 * number_of_signals)
    header = b"".join(
        [
            _field("0", 8),
            _field("X X X X", 80),
            _field("NeuroAegis synthetic fixture", 80),
            _field("01.01.24", 8),
            _field("00.00.00", 8),
            _field(str(header_bytes), 8),
            _field("", 44),
            _field("1", 8),
            _field(str(duration_seconds), 8),
            _field(str(number_of_signals), 4),
            b"".join(_field(name, 16) for name in channel_names),
            b"".join(_field("", 80) for _ in channel_names),
            b"".join(_field("uV", 8) for _ in channel_names),
            b"".join(_field("-32768", 8) for _ in channel_names),
            b"".join(_field("32767", 8) for _ in channel_names),
            b"".join(_field("-32768", 8) for _ in channel_names),
            b"".join(_field("32767", 8) for _ in channel_names),
            b"".join(_field("", 80) for _ in channel_names),
            b"".join(_field(str(samples_per_record), 8) for _ in channel_names),
            b"".join(_field("", 32) for _ in channel_names),
        ]
    )
    assert len(header) == header_bytes
    signal_record = b"".join(
        struct.pack("<h", (sample + channel_index) % 100)
        for sample in range(samples_per_record)
        for channel_index in range(number_of_signals)
    )
    path.write_bytes(header + signal_record)
    return path
