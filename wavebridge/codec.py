import numpy as np
def flatten_weights(array):
    return np.asarray(array, dtype=np.float32).reshape(-1)
def normalize_peak(values):
    values = np.asarray(values, dtype=np.float32)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if peak == 0.0:
        return values.copy(), 1.0
    return np.clip(values / peak, -1.0, 1.0), peak
def encode_pcm16(values):
    values = np.asarray(values, dtype=np.float32)
    return np.round(
        np.clip(values, -1.0, 1.0) * 32767.0
    ).astype("<i2")
def decode_pcm16(samples):
    return np.asarray(samples, dtype="<i2").astype(
        np.float32
    ) / 32767.0
def quantization_error(original, recovered):
    a = np.asarray(original, dtype=np.float32)
    b = np.asarray(recovered, dtype=np.float32)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))
