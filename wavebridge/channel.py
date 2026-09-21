from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class ChannelSpec:
    """Deterministic normalized-sample physical-channel model."""
    gain: float = 1.0
    dc_offset: float = 0.0
    bandwidth_hz: float | None = None
    noise_rms: float = 0.0
    clip_min: float = -1.0
    clip_max: float = 1.0
    seed: int = 0
def _lowpass_first_order(values, sample_rate, cutoff_hz):
    """Apply a causal first-order low-pass filter without SciPy."""
    if cutoff_hz is None:
        return values.copy()
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if cutoff_hz <= 0:
        raise ValueError("bandwidth_hz must be positive or None")
    alpha = 1.0 - np.exp(
        -2.0 * np.pi * cutoff_hz / sample_rate
    )
    output = np.empty_like(values, dtype=np.float32)
    state = 0.0
    for index, value in enumerate(values):
        state += alpha * (float(value) - state)
        output[index] = state
    return output
def simulate_channel(values, sample_rate=44100, spec=None):
    """
    Simulate:
        gain
          -> DC offset
          -> bandwidth limit
          -> noise
          -> clipping
    Inputs and outputs use the normalized WaveBridge sample domain.
    """
    spec = spec or ChannelSpec()
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if spec.clip_min >= spec.clip_max:
        raise ValueError("clip_min must be less than clip_max")
    if spec.noise_rms < 0:
        raise ValueError("noise_rms must be non-negative")
    signal = values * np.float32(spec.gain)
    signal = signal + np.float32(spec.dc_offset)
    signal = _lowpass_first_order(
        signal,
        sample_rate,
        spec.bandwidth_hz,
    )
    if spec.noise_rms:
        rng = np.random.default_rng(spec.seed)
        noise = rng.normal(
            0.0,
            spec.noise_rms,
            size=signal.size,
        ).astype(np.float32)
        signal = signal + noise
    return np.clip(
        signal,
        spec.clip_min,
        spec.clip_max,
    ).astype(np.float32)
def channel_metrics(original, captured):
    """Return sample-domain reconstruction metrics."""
    original = np.asarray(
        original,
        dtype=np.float32,
    ).reshape(-1)
    captured = np.asarray(
        captured,
        dtype=np.float32,
    ).reshape(-1)
    if original.shape != captured.shape:
        raise ValueError(
            "original and captured must have the same shape"
        )
    if original.size == 0:
        return {
            "samples": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "max_error": 0.0,
        }
    error = captured - original
    return {
        "samples": int(original.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "max_error": float(np.max(np.abs(error))),
    }
