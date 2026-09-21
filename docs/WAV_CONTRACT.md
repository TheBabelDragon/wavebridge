# WaveBridge WAV Contract

## v0.1

WaveBridge uses:

- WAV container
- PCM encoding
- signed 16-bit samples
- mono
- 44,100 samples/sec
- one numerical weight per sample

The first implementation performs peak normalization.

For weights `w`:

    p = max(abs(w))
    x = clip(w / p, -1, 1)
    pcm = round(x * 32767)

The normalization peak MUST be retained as metadata by higher-level
model exporters because the WAV alone does not preserve the original
floating-point scale.

No carrier, Manchester encoding, framing, or modulation is part of
the v0.1 weight representation.

Those belong to later physical-link layers.
