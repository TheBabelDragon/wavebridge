"""
Channel calibration from known probe waveforms.

Measure/model:
  gain, DC offset, noise RMS, clipping, latency, dynamic range

Result is a ChannelProfile that can drive SimulatedChannel or document
a real audio/optical path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .channel_profile import ChannelProfile


def _align_latency(tx: np.ndarray, rx: np.ndarray, max_lag: int = 4096) -> Tuple[int, np.ndarray]:
    """
    Estimate integer sample latency via cross-correlation.
    Returns (lag, rx_aligned) where positive lag means rx is delayed.
    """
    tx = np.asarray(tx, dtype=np.float32).reshape(-1)
    rx = np.asarray(rx, dtype=np.float32).reshape(-1)
    n = min(tx.size, rx.size)
    tx, rx = tx[:n], rx[:n]
    max_lag = min(max_lag, n // 2)
    best_lag = 0
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = tx[: n - lag], rx[lag:n]
        else:
            a, b = tx[-lag:n], rx[: n + lag]
        if a.size < 8:
            continue
        a0, b0 = a - a.mean(), b - b.mean()
        denom = (np.linalg.norm(a0) * np.linalg.norm(b0)) + 1e-12
        score = float(np.dot(a0, b0) / denom)
        if score > best:
            best = score
            best_lag = lag
    if best_lag >= 0:
        aligned = rx[best_lag : best_lag + tx.size]
        if aligned.size < tx.size:
            aligned = np.pad(aligned, (0, tx.size - aligned.size))
    else:
        aligned = np.pad(rx, (-best_lag, 0))[: tx.size]
    return best_lag, aligned.astype(np.float32)


def calibrate_from_probe(
    transmitted: np.ndarray,
    received: np.ndarray,
    *,
    sample_rate: int = 44100,
    name: str = "calibrated",
) -> ChannelProfile:
    """
    Estimate ChannelProfile from a known transmit waveform and capture.

    Uses correlation for latency, least-squares for gain/DC, residual for noise.
    """
    tx = np.asarray(transmitted, dtype=np.float32).reshape(-1)
    rx = np.asarray(received, dtype=np.float32).reshape(-1)
    if tx.size == 0 or rx.size == 0:
        return ChannelProfile(name=name, sample_rate=sample_rate)

    lag, rx_a = _align_latency(tx, rx)
    n = min(tx.size, rx_a.size)
    tx_n, rx_n = tx[:n], rx_a[:n]

    A = np.column_stack([tx_n, np.ones(n, dtype=np.float32)])
    try:
        coef, _, _, _ = np.linalg.lstsq(A, rx_n, rcond=None)
        gain = float(coef[0])
        dc = float(coef[1])
    except Exception:
        gain, dc = 1.0, 0.0

    residual = rx_n - (gain * tx_n + dc)
    noise_rms = float(np.std(residual)) if residual.size else 0.0

    peak_tx = float(np.max(np.abs(tx_n))) or 1.0
    peak_rx = float(np.max(np.abs(rx_n))) or 1e-12
    sig = float(np.std(gain * tx_n)) or 1e-12
    dr = 20.0 * float(np.log10(max(sig / (noise_rms + 1e-12), 1e-12)))

    return ChannelProfile(
        name=name,
        gain=gain,
        dc_offset=dc,
        noise_rms=noise_rms,
        latency_samples=int(lag),
        dynamic_range_db=dr,
        sample_rate=int(sample_rate),
        clip_min=float(np.min(rx_n)),
        clip_max=float(np.max(rx_n)),
        extras={"peak_tx": peak_tx, "peak_rx": peak_rx},
    )


def probe_waveform(n: int = 2048, *, kind: str = "chirp") -> np.ndarray:
    """Generate a calibration probe in the normalized sample domain."""
    t = np.linspace(0.0, 1.0, int(n), dtype=np.float32)
    if kind == "tone":
        return (0.5 * np.sin(2.0 * np.pi * 17 * t)).astype(np.float32)
    if kind == "impulse":
        x = np.zeros(int(n), dtype=np.float32)
        x[n // 10] = 0.9
        return x
    phase = 2.0 * np.pi * (5.0 * t + 40.0 * t * t)
    return (0.5 * np.sin(phase)).astype(np.float32)
