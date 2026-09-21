# WaveBridge

Neural weights as physical waveforms.

WaveBridge defines a deterministic bridge between numerical model
weights and ordinary PCM WAV files suitable for signal generators,
audio interfaces, analog amplifiers, optical transmitters, and
photodiode receivers.

## Core path

weights
  -> normalize
  -> quantize
  -> PCM WAV
  -> signal generator
  -> physical channel
  -> recorded WAV
  -> decode
  -> reconstructed values

## Initial contract

- mono
- signed 16-bit PCM
- 44.1 kHz
- deterministic normalization
- explicit metadata
- no modulation in v0.1
- no neural-network framework dependency

The waveform is the payload.
The physical channel is the bridge.
