import wave
from pathlib import Path
import numpy as np
def write_wav(path, pcm16, sample_rate=44100):
    pcm16 = np.asarray(pcm16, dtype="<i2")
    with wave.open(str(Path(path)), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
def read_wav(path):
    with wave.open(str(Path(path)), "rb") as w:
        if w.getnchannels() != 1:
            raise ValueError("WaveBridge requires mono WAV")
        if w.getsampwidth() != 2:
            raise ValueError("WaveBridge requires 16-bit PCM WAV")
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    return rate, np.frombuffer(frames, dtype="<i2").copy()
