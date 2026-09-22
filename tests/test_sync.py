"""Synchronization / framing recovery tests."""

from __future__ import annotations

import numpy as np
import pytest

from wavebridge.bridge import decode_field, encode_field
from wavebridge.sync import (
    SyncError,
    extract_packet,
    frame_to_waveform,
    transmit_framed,
)


def test_identity_framed_roundtrip():
    values = np.array([0.1, -0.2, 0.5, -0.9, 0.0], dtype=np.float32)
    packet = encode_field(values, metadata={"k": "v"})
    wave = frame_to_waveform(packet)
    recovered = extract_packet(wave)
    arr, meta = decode_field(recovered)
    assert float(np.max(np.abs(arr - values))) < 1e-3
    # metadata is not serialized in the binary frame (by design for v0.1)
    assert recovered.weight_count == values.size


@pytest.mark.parametrize("shift", [-1000, -500, -73, 37, 800])
def test_random_shift_recovery(shift: int):
    rng = np.random.default_rng(abs(shift) + 1)
    values = rng.standard_normal(64).astype(np.float32)
    packet = encode_field(values)

    if shift < 0:
        wave = transmit_framed(packet, leading_silence=-shift)
    else:
        wave = transmit_framed(packet, trailing_silence=shift)
        # also prepend a little silence so the packet is not at sample 0
        wave = np.concatenate(
            [np.zeros(50, dtype=np.float32), wave]
        )

    recovered = extract_packet(wave)
    arr, _ = decode_field(recovered)
    scale = float(np.max(np.abs(values))) or 1.0
    rel = float(np.max(np.abs(arr - values)) / scale)
    assert rel < 2e-4


def test_missing_sync_raises():
    noise = np.random.default_rng(0).standard_normal(2000).astype(np.float32) * 0.01
    with pytest.raises(SyncError, match="sync"):
        extract_packet(noise)


def test_crc_failure_is_explicit():
    values = np.ones(16, dtype=np.float32)
    packet = encode_field(values)
    wave = frame_to_waveform(packet)
    # Corrupt a mid-region sample strongly
    mid = wave.size // 2
    wave = wave.copy()
    wave[mid : mid + 8] = 0.99
    with pytest.raises((SyncError, ValueError)):
        extract_packet(wave)
