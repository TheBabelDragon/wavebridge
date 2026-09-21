"""WaveBridge weight-survival experiments (pure numerical domain)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .channel import ChannelSpec, channel_metrics, simulate_channel
from .codec import (
    decode_pcm16,
    encode_pcm16,
    flatten_weights,
    normalize_peak,
)


@dataclass(frozen=True)
class SurvivalReport:
    """Results of a single weight → channel → reconstruct run."""

    n_weights: int
    peak: float
    waveform: dict          # normalized-sample metrics
    weights: dict           # original-scale weight metrics
    channel: ChannelSpec


def run_weight_survival(
    weights,
    *,
    sample_rate: int = 44100,
    channel: ChannelSpec | None = None,
    requantize: bool = True,
) -> SurvivalReport:
    """
    Pure-numerical survival experiment:

        weights
          → flatten
          → peak normalize
          → PCM16
          → normalized samples
          → simulate_channel()
          → optional PCM16 re-quantization
          → decode
          → reconstruct original weight scale
          → metrics

    Returns both waveform-domain and weight-domain losses.
    """
    channel = channel or ChannelSpec()
    flat = flatten_weights(weights)
    if flat.size == 0:
        empty = {"samples": 0, "mae": 0.0, "rmse": 0.0, "max_error": 0.0}
        return SurvivalReport(
            n_weights=0,
            peak=1.0,
            waveform=empty,
            weights=empty,
            channel=channel,
        )

    # Encode side
    normalized, peak = normalize_peak(flat)
    pcm = encode_pcm16(normalized)

    # Channel operates in normalized sample domain
    samples = decode_pcm16(pcm)          # exact inverse of encode (quantization already applied)
    captured = simulate_channel(
        samples,
        sample_rate=sample_rate,
        spec=channel,
    )

    # Optional second quantization (models ADC / re-digitization)
    if requantize:
        captured = decode_pcm16(encode_pcm16(captured))

    # Waveform-domain metrics (normalized samples)
    waveform_metrics = channel_metrics(samples, captured)

    # Reconstruct original weight scale and compute weight-domain metrics
    reconstructed = captured * peak
    weight_metrics = channel_metrics(flat, reconstructed)

    return SurvivalReport(
        n_weights=int(flat.size),
        peak=float(peak),
        waveform=waveform_metrics,
        weights=weight_metrics,
        channel=channel,
    )


def sweep_channel_survival(
    weights,
    *,
    gains: Sequence[float] = (0.5, 0.75, 1.0, 1.25),
    bandwidths_hz: Sequence[float | None] = (
        500.0, 1000.0, 2000.0, 5000.0, 10000.0, None,
    ),
    noise_rms_values: Sequence[float] = (0.0, 0.001, 0.005, 0.01, 0.05),
    clip_levels: Sequence[float] = (1.0, 0.8, 0.5),
    sample_rate: int = 44100,
    requantize: bool = True,
    seed: int = 0,
) -> list[SurvivalReport]:
    """
    Produce a channel-survival map by sweeping common physical degradations.

    Each combination yields an independent SurvivalReport.
    """
    reports: list[SurvivalReport] = []
    for gain in gains:
        for bw in bandwidths_hz:
            for noise in noise_rms_values:
                for clip in clip_levels:
                    spec = ChannelSpec(
                        gain=float(gain),
                        dc_offset=0.0,
                        bandwidth_hz=bw,
                        noise_rms=float(noise),
                        clip_min=-float(clip),
                        clip_max=float(clip),
                        seed=seed,
                    )
                    reports.append(
                        run_weight_survival(
                            weights,
                            sample_rate=sample_rate,
                            channel=spec,
                            requantize=requantize,
                        )
                    )
    return reports
