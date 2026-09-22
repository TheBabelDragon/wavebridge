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

## MetaField integration (transport-neutral boundary)

WaveBridge now exposes a logical packet layer that sits *above*
PCM/WAV and *below* any physical channel:

```
field state
  → encode_field / encode_field_state
  → FieldPacket  (dtype, shape, peak_scale, payload, crc32, …)
  → PCM WAV | PAM | laser | optical body | ADC
  → decode_field / decode_field_observation
  → recovered values + Observation
```

### Public API

```python
from wavebridge import encode_field, decode_field, FieldPacket, Observation
from wavebridge import encode_field_state, decode_field_observation

packet = encode_field_state(values, source="metafield")
values, meta = decode_field(packet)

obs = decode_field_observation(packet, source="optical")
# obs.values, obs.source, obs.channel, obs.metadata
```

Design rules:

- WaveBridge knows nothing about MetaField internals.
- The packet is transport-neutral (not modulation).
- Physical channels (SimulatedChannel, AudioChannel, OpticalChannel, …)
  implement `transmit(waveform)` / `receive() → waveform` later;
  they are not required for the first integration commit.

See `tests/test_metafield_roundtrip.py` for a pure-numerical closed loop.
