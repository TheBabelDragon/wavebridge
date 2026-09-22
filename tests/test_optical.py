"""Optical channel + body simulator tests (no hardware)."""

from __future__ import annotations

import numpy as np
import pytest

from wavebridge.bridge import encode_field, decode_field
from wavebridge.hardware.optical import (
    OpticalBodySimulator,
    OpticalPathSpec,
    SerialOpticalChannel,
    SimulatedOpticalChannel,
)
from wavebridge.physical import packet_roundtrip


def test_simulated_optical_packet_roundtrip():
    values = np.linspace(-0.8, 0.8, 64, dtype=np.float32)
    packet = encode_field(values)
    ch = SimulatedOpticalChannel(
        path=OpticalPathSpec(gain=0.95, noise_rms=0.0003, seed=1)
    )
    obs = packet_roundtrip(packet, ch, leading_silence=80, trailing_silence=40)
    scale = float(np.max(np.abs(values))) or 1.0
    rel = float(np.max(np.abs(obs.values - values)) / scale)
    assert rel < 0.08


def test_serial_optical_requires_backend():
    ch = SerialOpticalChannel()
    with pytest.raises(NotImplementedError, match="backend"):
        ch.transmit(np.zeros(16, dtype=np.float32))


def test_optical_body_excite_shape():
    body = OpticalBodySimulator(n_lasers=12, n_detectors=20, seed=7)
    rec = body.excite(3, drive_level=0.8)
    assert rec.laser_id == 3
    assert rec.detector_response.shape == (20,)
    assert float(np.min(rec.detector_response)) >= 0.0
    assert float(np.max(rec.detector_response)) <= 1.0


def test_optical_body_transfer_matrix():
    body = OpticalBodySimulator(n_lasers=8, n_detectors=16, seed=0, noise_rms=0.0)
    T = body.estimate_transfer_matrix(drive_level=1.0)
    assert T.shape == (16, 8)
    assert float(np.max(np.abs(T[:, 0] - T[:, 1]))) > 0.01


def test_optical_body_mixed_excitation():
    body = OpticalBodySimulator(n_lasers=6, n_detectors=10, seed=2)
    drives = [0.0] * 6
    drives[1] = 0.9
    drives[4] = 0.5
    rec = body.excite_many(drives)
    assert rec.detector_response.shape == (10,)
    assert "drives" in rec.extras


def test_optical_replaces_simulated_in_same_api():
    """MetaField can swap SimulatedChannel → SimulatedOpticalChannel unchanged."""
    values = np.array([0.2, -0.4, 0.6], dtype=np.float32)
    packet = encode_field(values)
    ch = SimulatedOpticalChannel(path=OpticalPathSpec(noise_rms=0.0, gain=1.0))
    obs = packet_roundtrip(packet, ch)
    assert float(np.max(np.abs(obs.values - values))) < 1e-3
