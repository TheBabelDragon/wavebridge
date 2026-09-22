"""Tests for the transport-neutral FieldPacket boundary."""

from __future__ import annotations

import numpy as np
import pytest

from wavebridge.bridge import FieldPacket, decode_field, encode_field


def test_encode_decode_roundtrip_1d():
    values = np.array([0.1, -0.5, 0.9, 0.0, -0.25], dtype=np.float32)
    packet = encode_field(values, metadata={"tag": "t1"})
    assert isinstance(packet, FieldPacket)
    assert packet.weight_count == 5
    assert packet.payload_length == 5 * 2
    assert packet.dtype in ("float32", "float64")
    recovered, meta = decode_field(packet)
    assert recovered.shape == values.shape
    # PCM16 quantization error is bounded
    err = float(np.max(np.abs(recovered - values)))
    assert err < 1e-3
    assert meta.get("tag") == "t1"
    assert "peak_scale" in meta


def test_encode_decode_roundtrip_2d():
    rng = np.random.default_rng(0)
    values = rng.standard_normal((4, 8)).astype(np.float32)
    packet = encode_field(values)
    recovered, _ = decode_field(packet)
    assert recovered.shape == values.shape
    # Relative scale preserved after peak restore
    scale = float(np.max(np.abs(values))) or 1.0
    rel = float(np.max(np.abs(recovered - values)) / scale)
    assert rel < 2e-4


def test_packet_to_bytes_and_back():
    values = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    packet = encode_field(values, metadata={"src": "unit"})
    blob = packet.to_bytes()
    restored = FieldPacket.from_bytes(blob)
    assert restored.weight_count == packet.weight_count
    assert restored.peak_scale == pytest.approx(packet.peak_scale)
    assert restored.crc32 == packet.crc32
    arr, meta = decode_field(restored)
    assert arr.shape == values.shape
    assert float(np.max(np.abs(arr - values))) < 1e-3


def test_crc_detects_corruption():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    packet = encode_field(values)
    blob = bytearray(packet.to_bytes())
    # Flip a payload byte
    blob[_HEADER_SIZE_SAFE(blob)] ^= 0xFF
    with pytest.raises(ValueError, match="crc32"):
        FieldPacket.from_bytes(bytes(blob))


def _HEADER_SIZE_SAFE(blob: bytearray) -> int:
    # Skip magic + a few header fields into payload region
    return 20


def test_empty_array():
    values = np.array([], dtype=np.float32)
    packet = encode_field(values)
    recovered, _ = decode_field(packet)
    assert recovered.size == 0


def test_metadata_preserved():
    meta_in = {"experiment": "A", "body": "optical"}
    packet = encode_field(np.ones(8, dtype=np.float32), metadata=meta_in)
    _, meta_out = decode_field(packet)
    assert meta_out["experiment"] == "A"
    assert meta_out["body"] == "optical"
