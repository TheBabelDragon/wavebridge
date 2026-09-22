"""Calibration → ChannelProfile tests."""

from __future__ import annotations

import numpy as np

from wavebridge.calibration import calibrate_from_probe, probe_waveform
from wavebridge.channel import ChannelSpec, simulate_channel
from wavebridge.channel_profile import ChannelProfile


def test_calibrate_identity():
    tx = probe_waveform(1024, kind="chirp")
    profile = calibrate_from_probe(tx, tx, name="id")
    assert abs(profile.gain - 1.0) < 0.05
    assert abs(profile.dc_offset) < 0.05
    assert profile.noise_rms < 0.02


def test_calibrate_gain_and_dc():
    tx = probe_waveform(2048, kind="tone")
    rx = 0.7 * tx + 0.05
    profile = calibrate_from_probe(tx, rx, name="g")
    assert abs(profile.gain - 0.7) < 0.05
    assert abs(profile.dc_offset - 0.05) < 0.05


def test_profile_to_channel_spec():
    p = ChannelProfile(name="x", gain=0.9, noise_rms=0.001, dc_offset=0.01)
    spec = p.to_channel_spec()
    assert isinstance(spec, ChannelSpec)
    assert abs(spec.gain - 0.9) < 1e-9


def test_calibrate_through_simulated_channel():
    tx = probe_waveform(1500, kind="chirp")
    spec = ChannelSpec(gain=0.85, dc_offset=0.02, noise_rms=0.0, seed=0)
    rx = simulate_channel(tx, sample_rate=44100, spec=spec)
    profile = calibrate_from_probe(tx, rx, name="sim")
    assert abs(profile.gain - 0.85) < 0.08
    assert abs(profile.dc_offset - 0.02) < 0.08
