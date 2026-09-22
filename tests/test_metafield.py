"""Tests for the MetaField-facing WaveBridge API."""

from __future__ import annotations

import numpy as np

from wavebridge.bridge import FieldPacket
from wavebridge.metafield import decode_field_observation, encode_field_state
from wavebridge.observation import Observation


def test_encode_field_state_returns_packet():
    values = np.array([0.2, -0.4, 0.8], dtype=np.float32)
    packet = encode_field_state(values, source="metafield", metadata={"run": 1})
    assert isinstance(packet, FieldPacket)
    assert packet.metadata.get("source") == "metafield"
    assert packet.metadata.get("run") == 1
    assert packet.weight_count == 3


def test_decode_field_observation_from_array():
    values = np.linspace(-0.5, 0.5, 16, dtype=np.float32)
    obs = decode_field_observation(
        values, source="optical", channel="bpw34", metadata={"gain": 1.0}
    )
    assert isinstance(obs, Observation)
    assert obs.source == "optical"
    assert obs.channel == "bpw34"
    assert obs.metadata.get("gain") == 1.0
    np.testing.assert_allclose(obs.values, values)


def test_decode_field_observation_from_packet():
    values = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    packet = encode_field_state(values, source="metafield")
    obs = decode_field_observation(packet, source="optical")
    assert isinstance(obs, Observation)
    assert obs.source == "optical"
    assert obs.values.shape == values.shape
    err = float(np.max(np.abs(obs.values - values)))
    assert err < 1e-3


def test_roundtrip_via_public_api():
    rng = np.random.default_rng(7)
    state = rng.standard_normal(32).astype(np.float32)
    packet = encode_field_state(state, source="metafield")
    obs = decode_field_observation(packet, source="sim")
    scale = float(np.max(np.abs(state))) or 1.0
    rel = float(np.max(np.abs(obs.values - state)) / scale)
    assert rel < 2e-4
