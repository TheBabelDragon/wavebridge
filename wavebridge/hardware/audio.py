"""
Audio hardware / file bridge.

Primary path (always available, no extra deps):

    packet → framed waveform → write WAV → (physical cable / DAW / cable)
           → read WAV → extract packet

Live path (optional, requires sounddevice):

    AudioLoopbackChannel  — play + record on the host audio interface

Neither path imports MetaField.  Both implement PhysicalChannel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..codec import decode_pcm16, encode_pcm16
from ..physical import PhysicalChannel
from ..wav import read_wav, write_wav


class AudioFileChannel(PhysicalChannel):
    """
    File-based audio channel.

    transmit(waveform) writes mono 16-bit PCM WAV.
    receive() reads the same (or a different) path back as normalized samples.

    Typical offline experiment:

        ch = AudioFileChannel("tx.wav", rx_path="rx.wav")
        ch.transmit(wave)
        # ... user copies/plays/records externally into rx.wav ...
        captured = ch.receive()
    """

    def __init__(
        self,
        tx_path: Union[str, Path],
        *,
        rx_path: Optional[Union[str, Path]] = None,
        sample_rate: int = 44100,
    ) -> None:
        self.tx_path = Path(tx_path)
        self.rx_path = Path(rx_path) if rx_path is not None else self.tx_path
        self.sample_rate = int(sample_rate)
        self._last_tx: Optional[np.ndarray] = None

    def transmit(self, waveform: np.ndarray) -> None:
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        self._last_tx = samples.copy()
        pcm = encode_pcm16(samples)
        self.tx_path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(self.tx_path, pcm, sample_rate=self.sample_rate)

    def receive(self) -> np.ndarray:
        if not self.rx_path.exists():
            raise FileNotFoundError(
                f"AudioFileChannel: rx path does not exist: {self.rx_path}"
            )
        rate, pcm = read_wav(self.rx_path)
        if rate != self.sample_rate:
            pass
        return decode_pcm16(pcm)


class AudioLoopbackChannel(PhysicalChannel):
    """
    Live host audio loopback via sounddevice (optional dependency).

    transmit() plays the waveform and simultaneously records.
    receive() returns the last recording.

    If sounddevice is not installed, transmit/receive raise ImportError
    with a clear message — use AudioFileChannel instead.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        device: Optional[Union[int, str]] = None,
        latency: str = "high",
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.device = device
        self.latency = latency
        self._buffer: Optional[np.ndarray] = None

    def _sd(self):
        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "AudioLoopbackChannel requires the 'sounddevice' package. "
                "pip install sounddevice  — or use AudioFileChannel for offline WAV I/O."
            ) from exc
        return sd

    def transmit(self, waveform: np.ndarray) -> None:
        sd = self._sd()
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        recorded = sd.playrec(
            samples.reshape(-1, 1),
            samplerate=self.sample_rate,
            channels=1,
            device=self.device,
            latency=self.latency,
            dtype="float32",
        )
        sd.wait()
        self._buffer = np.asarray(recorded, dtype=np.float32).reshape(-1)

    def receive(self) -> np.ndarray:
        if self._buffer is None:
            raise RuntimeError("AudioLoopbackChannel.receive() before transmit()")
        return self._buffer.copy()


def wav_roundtrip_paths(
    waveform: np.ndarray,
    tx_path: Union[str, Path],
    *,
    rx_path: Optional[Union[str, Path]] = None,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Write waveform to WAV and immediately read it back (codec self-test)."""
    ch = AudioFileChannel(tx_path, rx_path=rx_path, sample_rate=sample_rate)
    ch.transmit(waveform)
    return ch.receive()
