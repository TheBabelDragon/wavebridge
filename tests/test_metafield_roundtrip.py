"""
End-to-end synthetic roundtrip:

  state → encode → simulated channel → decode → recovered state
"""

from __future__ import annotations

import numpy as np

from wavebridge.bridge import decode_field, encode_field
from wavebridge.channel import ChannelSpec, simulate_channel
from wavebridge.codec import decode_pcm16, encode_pcm16
from wavebridge.metafield import decode_field_observation, encode_field_state


def test_state_encode_channel_decode_roundtrip():
    rng = np.random.default_rng(42)
    state = rng.standard_normal(128).astype(np.float32)

    # 1. Encode to transport-neutral packet
    packet = encode_field_state(state, source="metafield")

    # 2. Materialize as normalized samples (what a physical channel sees)
    pcm = np.frombuffer(packet.payload, dtype="<i2")
    samples = decode_pcm16(pcm)

    # 3. Simulated physical channel (mild noise + gain)
    channel = ChannelSpec(gain=0.98, noise_rms=0.002, seed=1)
    captured = simulate_channel(samples, sample_rate=packet.sample_rate, spec=channel)

    # 4. Re-quantize (ADC model) and rebuild packet-like payload
    captured_pcm = encode_pcm16(captured)
    recovered_packet = encode_field(
        captured * packet.peak_scale,  # restore scale for encode path
        metadata={"source": "optical"},
        sample_rate=packet.sample_rate,
    )
    # Alternative path: decode the original packet after channel on samples
    # Here we use the public observation path for clarity.
    obs = decode_field_observation(
        captured * packet.peak_scale,
        source="optical",
        metadata={"channel": "sim"},
    )

    recovered = obs.values
    assert recovered.shape == state.shape

    scale = float(np.max(np.abs(state))) or 1.0
    rel_err = float(np.max(np.abs(recovered - state)) / scale)
    # With mild channel noise we still stay well under 5 % relative peak error
    assert rel_err < 0.05


def test_packet_survives_identity_channel():
    state = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    packet = encode_field(state)
    recovered, meta = decode_field(packet)
    err = float(np.max(np.abs(recovered - state)))
    assert err < 1e-3
    assert meta.get("peak_scale") is not None
