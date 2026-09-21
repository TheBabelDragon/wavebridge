# WaveBridge Channel Simulator
The simulator models the first physical-link effects before hardware is connected.
## Processing order
1. gain
2. DC offset
3. bandwidth limit
4. additive noise
5. clipping
The model deliberately contains no carrier, modulation scheme, or neural-network
assumptions.
## Sample domain
The simulator operates on normalized floating-point samples in the `[-1, 1]`
domain used by the WB1 PCM codec.
## Bandwidth
`bandwidth_hz` selects a causal first-order low-pass filter.
The implementation uses NumPy only and therefore requires no SciPy dependency.
## Reproducibility
Noise uses `numpy.random.default_rng(seed)`.
Identical input, parameters, and seed produce identical captures.
## Metrics
`channel_metrics()` reports:
- sample count
- mean absolute error
- root mean square error
- maximum absolute error
These are sample-domain diagnostics. They do not yet claim equivalent neural-network
performance.
