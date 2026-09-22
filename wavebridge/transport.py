"""
Packet ↔ waveform conversion (no modulation).

    FieldPacket
        ↕  packet_to_waveform / waveform_to_packet
    PCM waveform (normalized float32 samples)
        ↕
    PhysicalChannel

The waveform is the PCM representation of the packet payload.
Modulation (PAM, etc.) is intentionally deferred.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .bridge import FieldPacket
from .codec import decode_pcm16, encode_pcm16


def packet_to_waveform(
    packet: FieldPacket,
    *,
    as_pcm16: bool = False,
) -> np.ndarray:
    """Convert a FieldPacket payload to a mono waveform (normalized or PCM16)."""
    pcm = np.frombuffer(packet.payload, dtype="<i2")
    if as_pcm16:
        return pcm.copy()
    return decode_pcm16(pcm)


def waveform_to_packet(
    waveform: np.ndarray,
    *,
    peak_scale: float = 1.0,
    sample_rate: int = 44100,
    shape: Optional[Tuple[int, ...]] = None,
    metadata: Optional[dict] = None,
    dtype: str = "float32",
) -> FieldPacket:
    """Rebuild a FieldPacket from a normalized waveform (no sync framing)."""
    from .bridge import encode_field

    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    values = samples * float(peak_scale)
    if shape is not None and int(np.prod(shape)) == values.size:
        values = values.reshape(shape)
    return encode_field(
        values,
        metadata=metadata,
        sample_rate=sample_rate,
    )


def packet_bytes_to_waveform(data: bytes) -> np.ndarray:
    """Map arbitrary bytes to normalized waveform via int16 PCM interpretation."""
    if len(data) % 2 == 1:
        data = data + b"\x00"
    pcm = np.frombuffer(data, dtype="<i2")
    return decode_pcm16(pcm)


def waveform_to_packet_bytes(waveform: np.ndarray) -> bytes:
    """Inverse of packet_bytes_to_waveform."""
    pcm = encode_pcm16(np.asarray(waveform, dtype=np.float32).reshape(-1))
    return pcm.tobytes()
