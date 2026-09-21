#!/usr/bin/env python3
import argparse
import struct
import wave
from pathlib import Path
import numpy as np
def weights_to_wav(src, dst, sample_rate=44100):
    data = np.load(src)
    if isinstance(data, np.lib.npyio.NpzFile):
        arrays = [data[k].astype(np.float32).ravel() for k in data.files]
        weights = np.concatenate(arrays)
    else:
        weights = data.astype(np.float32).ravel()
    peak = float(np.max(np.abs(weights)))
    if peak == 0:
        peak = 1.0
    pcm = np.clip(weights / peak, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    print(f"Wrote {dst}")
    print(f"samples={len(pcm16)}")
    print(f"sample_rate={sample_rate}")
    print(f"duration={len(pcm16) / sample_rate:.6f}s")
    print(f"normalization_peak={peak}")
def wav_to_weights(src, dst):
    with wave.open(str(src), "rb") as w:
        if w.getnchannels() != 1:
            raise ValueError("WaveBridge currently requires mono WAV")
        if w.getsampwidth() != 2:
            raise ValueError("WaveBridge currently requires 16-bit PCM WAV")
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        weights = samples.astype(np.float32) / 32767.0
    np.save(dst, weights)
    print(f"Wrote {dst}")
    print(f"samples={len(weights)}")
    print(f"sample_rate={w.getframerate()}")
def main():
    p = argparse.ArgumentParser(prog="wavebridge")
    sub = p.add_subparsers(dest="command", required=True)
    enc = sub.add_parser("weights-to-wav")
    enc.add_argument("input")
    enc.add_argument("output")
    enc.add_argument("--sample-rate", type=int, default=44100)
    dec = sub.add_parser("wav-to-weights")
    dec.add_argument("input")
    dec.add_argument("output")
    args = p.parse_args()
    if args.command == "weights-to-wav":
        weights_to_wav(
            Path(args.input),
            Path(args.output),
            args.sample_rate,
        )
    elif args.command == "wav-to-weights":
        wav_to_weights(
            Path(args.input),
            Path(args.output),
        )
if __name__ == "__main__":
    main()
