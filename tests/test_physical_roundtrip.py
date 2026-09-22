"""
True physical packet path:

  FieldPacket → waveform → SimulatedChannel → waveform → packet → Observation

Must either decode successfully or fail explicitly (CRC/sync).
"""

from __future__ import annotations

import numpy as np
import pytest

from wavebridge.bridge import encode_field
from wavebridge.channel import ChannelSpec
from wavebridge.physical import (
    SimulatedChannel,
    packet_roundtrip,
    receive_packet,
    send_packet,
)
from wavebridge.sync import SyncError


def test_clean_simulated_roundtrip():
    values = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    packet = encode_field(values, metadata={"run": "clean"})
    channel = SimulatedChannel(spec=ChannelSpec(gain=1.0, noise_rms=0.0))

    obs = packet_roundtrip(packet, channel, leading_silence=200)
    assert obs.source == "physical"
    assert obs.channel == "SimulatedChannel"
    scale = float(np.max(np.abs(values))) or 1.0
    rel = float(np.max(np.abs(obs.values - values)) / scale)
    assert rel < 2e-4
    assert obs.values.shape == values.shape


def test_noisy_channel_still_recovers_or_fails_explicitly():
    rng = np.random.default_rng(9)
    values = rng.standard_normal(80).astype(np.float32)
    packet = encode_field(values)
    # Mild noise — framing should survive
    channel = SimulatedChannel(
        spec=ChannelSpec(gain=0.98, noise_rms=0.0005, seed=3)
    )
    obs = packet_roundtrip(
        packet,
        channel,
        leading_silence=150,
        trailing_silence=100,
    )
    scale = float(np.max(np.abs(values))) or 1.0
    rel = float(np.max(np.abs(obs.values - values)) / scale)
    assert rel < 0.05


def test_destroyed_waveform_fails_explicitly():
    values = np.array([0.3, -0.3, 0.6], dtype=np.float32)
    packet = encode_field(values)
    channel = SimulatedChannel(spec=ChannelSpec())
    send_packet(packet, channel, leading_silence=10)
    # Overwrite buffer with pure noise after transmit
    channel._buffer = (
        np.random.default_rng(1).standard_normal(5000).astype(np.float32) * 0.5
    )
    with pytest.raises((SyncError, ValueError, RuntimeError)):
        receive_packet(channel)


def test_shifted_packet_through_channel():
    values = np.array([0.0, 0.25, -0.5, 0.75, -1.0], dtype=np.float32)
    packet = encode_field(values)
    channel = SimulatedChannel(spec=ChannelSpec(gain=1.0, noise_rms=0.0))
    obs = packet_roundtrip(
        packet,
        channel,
        leading_silence=800,
        trailing_silence=37,
    )
    assert float(np.max(np.abs(obs.values - values))) < 1e-3
