# WaveBridge Weight Survival Experiment

This is the first experiment that turns WaveBridge from a representation
into a measurable physical-link model.

## Pipeline (pure numerical domain)

```
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
```

No WAV files are involved. The physical simulator is exercised in the
normalized sample domain so that container effects remain orthogonal.

## Two distinct loss domains

1. **Waveform-domain loss**
   - normalized-sample MAE
   - normalized-sample RMSE
   - maximum absolute error
   - Measures fidelity of the physical waveform itself.

2. **Weight-domain loss**
   - Reconstruct using the original normalization peak
   - MAE / RMSE / max error against the original weights
   - Measures how much of the numerical weight information survives.

These are reported separately so that “waveform fidelity” and
“weight fidelity” are never conflated.

## Channel survival map

`sweep_channel_survival()` produces a map over:

| Parameter     | Values                          |
|---------------|---------------------------------|
| gain          | 0.5, 0.75, 1.0, 1.25           |
| bandwidth_hz  | 500, 1k, 2k, 5k, 10k, None     |
| noise_rms     | 0, 0.001, 0.005, 0.01, 0.05    |
| clip          | ±1.0, ±0.8, ±0.5               |

Each combination yields an independent `SurvivalReport`.

## Experimental ladder

```
WB1  →  channel physics  →  weight survival  →  WAV round-trip  →  hardware
```

This document covers the second and third steps.
