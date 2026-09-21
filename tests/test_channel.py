import numpy as np
import pytest
from wavebridge.channel import (
    ChannelSpec,
    channel_metrics,
    simulate_channel,
)
def test_identity_channel_is_exact():
    x = np.linspace(-0.9, 0.9, 1000, dtype=np.float32)
    y = simulate_channel(
        x,
        spec=ChannelSpec(),
    )
    np.testing.assert_array_equal(x, y)
def test_gain_offset_and_clipping_follow_contract():
    x = np.array(
        [-1.0, 0.0, 1.0],
        dtype=np.float32,
    )
    spec = ChannelSpec(
        gain=2.0,
        dc_offset=0.25,
    )
    y = simulate_channel(x, spec=spec)
    np.testing.assert_allclose(
        y,
        [-1.0, 0.25, 1.0],
        atol=1e-7,
    )
def test_seeded_noise_is_reproducible():
    x = np.zeros(256, dtype=np.float32)
    spec = ChannelSpec(
        noise_rms=0.1,
        seed=123,
    )
    a = simulate_channel(x, spec=spec)
    b = simulate_channel(x, spec=spec)
    np.testing.assert_array_equal(a, b)
def test_bandwidth_limit_changes_fast_signal():
    x = np.zeros(100, dtype=np.float32)
    x[0] = 1.0
    y = simulate_channel(
        x,
        sample_rate=44100,
        spec=ChannelSpec(
            bandwidth_hz=1000.0,
        ),
    )
    assert 0.0 < y[0] < 1.0
    assert y[1] < y[0]
def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        simulate_channel(
            np.zeros(4),
            spec=ChannelSpec(noise_rms=-1.0),
        )
    with pytest.raises(ValueError):
        simulate_channel(
            np.zeros(4),
            spec=ChannelSpec(bandwidth_hz=0.0),
        )
def test_metrics_are_deterministic():
    x = np.array(
        [0.0, 1.0, -1.0],
        dtype=np.float32,
    )
    y = np.array(
        [0.0, 0.5, -0.5],
        dtype=np.float32,
    )
    metrics = channel_metrics(x, y)
    assert metrics["samples"] == 3
    assert metrics["mae"] == pytest.approx(1.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(
        np.sqrt(1.0 / 6.0)
    )
    assert metrics["max_error"] == pytest.approx(0.5)
