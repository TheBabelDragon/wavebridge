"""Hardware backends for WaveBridge physical channels."""

from .audio import AudioFileChannel, AudioLoopbackChannel
from .serial_backend import ESP32SerialBackend, SerialConfig, SerialExcitationResult
from .optical import (
    OpticalBodySimulator,
    OpticalChannel,
    OpticalExcitationRecord,
    OpticalPathSpec,
    SerialOpticalChannel,
    SimulatedOpticalChannel,
)

__all__ = [
    "AudioFileChannel",
    "AudioLoopbackChannel",
    "SimulatedOpticalChannel",
    "OpticalChannel",
    "SerialOpticalChannel",
    "OpticalPathSpec",
    "OpticalBodySimulator",
    "OpticalExcitationRecord",
    "ESP32SerialBackend",
    "SerialConfig",
    "SerialExcitationResult",
]
