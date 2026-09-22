"""Hardware backends for WaveBridge physical channels."""

from .audio import AudioFileChannel, AudioLoopbackChannel

__all__ = [
    "AudioFileChannel",
    "AudioLoopbackChannel",
]
