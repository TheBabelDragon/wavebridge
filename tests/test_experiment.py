import numpy as np
import pytest

from wavebridge.channel import ChannelSpec
from wavebridge.experiment import (
    SurvivalReport,
    run_weight_survival,
    sweep_channel_survival,
)


def _make_weights(n=4096, seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.25, n).astype(np.float32)


def test_identity_channel_preserves_weights_within_quantization():
    weights = _make_weights()
    report = run_weight_survival(
        weights,
        channel=ChannelSpec(),          # perfect channel
        requantize=True,
    )
    assert isinstance(report, SurvivalReport)
    assert report.n_weights == 4096
    # Waveform domain should be near-perfect after requantization
    assert report.waveform["max_error"] < 2.0 / 32768.0
    # Weight domain limited only by 16-bit quantization + peak scaling
    assert report.weights["max_error"] < 1e-3


def test_severe_noise_increases_weight_error():
    weights = _make_weights()
    clean = run_weight_survival(
        weights,
        channel=ChannelSpec(noise_rms=0.0),
    )
    noisy = run_weight_survival(
        weights,
        channel=ChannelSpec(noise_rms=0.05, seed=1),
    )
    assert noisy.weights["rmse"] > clean.weights["rmse"]
    assert noisy.weights["max_error"] > clean.weights["max_error"]


def test_clipping_reduces_dynamic_range():
    weights = _make_weights()
    mild = run_weight_survival(
        weights,
        channel=ChannelSpec(clip_min=-1.0, clip_max=1.0),
    )
    hard = run_weight_survival(
        weights,
        channel=ChannelSpec(clip_min=-0.5, clip_max=0.5),
    )
    assert hard.weights["mae"] > mild.weights["mae"]


def test_bandwidth_limit_affects_high_frequency_content():
    # Impulse-like weights (high-frequency content)
    weights = np.zeros(256, dtype=np.float32)
    weights[0] = 1.0
    weights[128] = -0.8
    full = run_weight_survival(
        weights,
        channel=ChannelSpec(bandwidth_hz=None),
    )
    limited = run_weight_survival(
        weights,
        channel=ChannelSpec(bandwidth_hz=500.0),
    )
    assert limited.weights["rmse"] > full.weights["rmse"]


def test_sweep_produces_reports():
    weights = _make_weights(n=512)
    reports = sweep_channel_survival(
        weights,
        gains=(1.0,),
        bandwidths_hz=(None, 2000.0),
        noise_rms_values=(0.0, 0.01),
        clip_levels=(1.0,),
    )
    assert len(reports) == 4
    assert all(isinstance(r, SurvivalReport) for r in reports)


def test_empty_weights():
    report = run_weight_survival(np.array([], dtype=np.float32))
    assert report.n_weights == 0
    assert report.waveform["samples"] == 0
    assert report.weights["samples"] == 0
