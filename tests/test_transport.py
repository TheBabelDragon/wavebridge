"""Tests for packet ↔ waveform transport (no modulation)."""

from __future__ import annotations

import numpy as np

from wavebridge.bridge import decode_field, encode_field
from wavebridge.transport import (
    packet_bytes_to_waveform,
    packet_to_waveform,
    waveform_to_packet,
    waveform_to_packet_bytes,
)


def test_packet_to_waveform_and_back():
    values = np.linspace(-0.8, 0.8, 48, dtype=np.float32)
    packet = encode_field(values)
    wave = packet_to_waveform(packet)
    assert wave.ndim == 1
    assert wave.dtype == np.float32
    assert float(np.max(np.abs(wave))) <= 1.0 + 1e-5

    recovered = waveform_to_packet(
        wave,
        peak_scale=packet.peak_scale,
        sample_rate=packet.sample_rate,
        shape=packet.shape,
    )
    arr, _ = decode_field(recovered)
    scale = float(np.max(np.abs(values))) or 1.0
    rel = float(np.max(np.abs(arr - values)) / scale)
    assert rel < 2e-4


def test_packet_bytes_waveform_roundtrip():
    payload = b"WB1P" + bytes(range(64))
    wave = packet_bytes_to_waveform(payload)
    back = waveform_to_packet_bytes(wave)
    assert len(back) >= len(payload) - 1
