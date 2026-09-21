import numpy as np
from wavebridge.codec import (
    normalize_peak,
    encode_pcm16,
    decode_pcm16,
    quantization_error,
)
def test_peak_normalization():
    x = np.array([-2.0, 0.0, 1.0, 2.0], dtype=np.float32)
    y, peak = normalize_peak(x)
    assert peak == 2.0
    assert np.max(np.abs(y)) <= 1.0
    assert y[0] == -1.0
    assert y[-1] == 1.0
def test_pcm_roundtrip_is_bounded():
    x = np.linspace(-1.0, 1.0, 10000, dtype=np.float32)
    pcm = encode_pcm16(x)
    recovered = decode_pcm16(pcm)
    assert recovered.shape == x.shape
    assert quantization_error(x, recovered) < 1.0 / 30000.0
