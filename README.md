# WaveBridge

**Neural weights \u2194 physical WAV bridge**

WaveBridge v0.1 is the deliberately boring first contract:

```
.npy / .npz
      \u2193
float32 weights
      \u2193
normalize
      \u2193
signed int16 PCM
      \u2193
44.1 kHz mono WAV
      \u2193
SIGNAL GENERATOR
```

Then the reverse path:

```
SIGNAL GENERATOR
      \u2193
optical hardware
      \u2193
BPW34 measurements
      \u2193
WAV
      \u2193
wavebridge wav-to-weights
      \u2193
.npy
```

No modulation, no protocol, no UTF-8 instruction layer yet.  
First prove that actual model weight data can make the round trip through the audio/optical interface.  
Then we can add the UTF-8/kernel envelope as v0.2.

## Install

```bash
pip install -r requirements-wavebridge.txt
```

## Usage

### Weights \u2192 WAV

```bash
python -m wavebridge weights-to-wav \
    checkpoints/best_model.npy \
    artifacts/weights.wav \
    --sample-rate 44100
```

### WAV \u2192 Weights

```bash
python -m wavebridge wav-to-weights \
    artifacts/weights.wav \
    recovered_weights.npy
```

## Design Notes (v0.1)

- Mono, 16-bit PCM, 44.1 kHz by default
- Peak-normalized before quantization
- Supports both `.npy` and `.npz` inputs
- Round-trip fidelity is limited only by 16-bit quantization noise (~3e-5 max abs error on normalized weights)

## License

MIT (or whatever you prefer \u2014 open to discussion)
