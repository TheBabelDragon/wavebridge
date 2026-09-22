"""
WaveBridge transport-neutral packet boundary.

encode_field(values, metadata) -> packet
decode_field(packet) -> values, metadata

The packet is a logical object that can later become PCM WAV, PAM,
laser drive, optical field, photodiode/TIA, ADC, and back.  No
modulation is performed here.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from .codec import decode_pcm16, encode_pcm16, flatten_weights, normalize_peak
from .format import WaveSpec

# Packet magic / version
_MAGIC = b"WB1P"
_HEADER_FMT = ">4sIIIfII"  # magic, dtype_code, ndim, weight_count, peak_scale, sample_rate, payload_len
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# dtype codes (keep small & stable)
_DTYPE_CODES = {
    "float32": 1,
    "float64": 2,
    "int16": 3,
    "int32": 4,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_CODES.items()}


@dataclass(frozen=True)
class FieldPacket:
    """
    Transport-neutral logical packet.

    Fields match the required contract:
      dtype, shape, weight_count, peak_scale, sample_rate,
      payload_length, payload, crc32
    """

    dtype: str
    shape: Tuple[int, ...]
    weight_count: int
    peak_scale: float
    sample_rate: int
    payload_length: int
    payload: bytes
    crc32: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        """Serialize packet to a self-describing byte stream (header + shape + payload + crc)."""
        dtype_code = _DTYPE_CODES.get(self.dtype)
        if dtype_code is None:
            raise ValueError(f"unsupported dtype for packet: {self.dtype}")
        ndim = len(self.shape)
        shape_bytes = struct.pack(f">{ndim}I", *self.shape) if ndim else b""
        header = struct.pack(
            _HEADER_FMT,
            _MAGIC,
            dtype_code,
            ndim,
            self.weight_count,
            float(self.peak_scale),
            int(self.sample_rate),
            int(self.payload_length),
        )
        body = header + shape_bytes + self.payload
        crc = zlib.crc32(body) & 0xFFFFFFFF
        return body + struct.pack(">I", crc)

    @classmethod
    def from_bytes(cls, data: bytes, metadata: Optional[Mapping[str, Any]] = None) -> "FieldPacket":
        if len(data) < _HEADER_SIZE + 4:
            raise ValueError("packet too short")
        body, crc_bytes = data[:-4], data[-4:]
        crc_stored = struct.unpack(">I", crc_bytes)[0]
        crc_calc = zlib.crc32(body) & 0xFFFFFFFF
        if crc_stored != crc_calc:
            raise ValueError(
                f"crc32 mismatch: stored={crc_stored:#x} calculated={crc_calc:#x}"
            )
        magic, dtype_code, ndim, weight_count, peak_scale, sample_rate, payload_len = (
            struct.unpack(_HEADER_FMT, body[:_HEADER_SIZE])
        )
        if magic != _MAGIC:
            raise ValueError(f"bad packet magic: {magic!r}")
        dtype = _CODE_TO_DTYPE.get(dtype_code)
        if dtype is None:
            raise ValueError(f"unknown dtype code: {dtype_code}")
        shape_offset = _HEADER_SIZE
        shape_end = shape_offset + ndim * 4
        if len(body) < shape_end + payload_len:
            raise ValueError("packet truncated")
        shape = (
            struct.unpack(f">{ndim}I", body[shape_offset:shape_end])
            if ndim
            else ()
        )
        payload = body[shape_end : shape_end + payload_len]
        if len(payload) != payload_len:
            raise ValueError("payload length mismatch")
        return cls(
            dtype=dtype,
            shape=tuple(shape),
            weight_count=int(weight_count),
            peak_scale=float(peak_scale),
            sample_rate=int(sample_rate),
            payload_length=int(payload_len),
            payload=payload,
            crc32=crc_stored,
            metadata=dict(metadata or {}),
        )


def encode_field(
    values: Any,
    metadata: Optional[Mapping[str, Any]] = None,
    *,
    sample_rate: int = 44100,
) -> FieldPacket:
    """
    Encode an array of field values into a transport-neutral packet.

    Pipeline (logical only — no channel / modulation):
      values → flatten → peak-normalize → PCM16 payload → packet
    """
    arr = np.asarray(values)
    if arr.size == 0:
        flat = np.asarray([], dtype=np.float32)
        shape: Tuple[int, ...] = tuple(arr.shape) if arr.ndim else (0,)
    else:
        shape = tuple(arr.shape)
        flat = flatten_weights(arr)

    normalized, peak = normalize_peak(flat)
    pcm = encode_pcm16(normalized)
    payload = pcm.tobytes()
    meta = dict(metadata or {})
    packet = FieldPacket(
        dtype=str(arr.dtype) if arr.size else "float32",
        shape=shape,
        weight_count=int(flat.size),
        peak_scale=float(peak),
        sample_rate=int(sample_rate),
        payload_length=len(payload),
        payload=payload,
        crc32=0,  # filled by to_bytes / kept for introspection
        metadata=meta,
    )
    # Recompute crc over the canonical serialization for the attribute
    serialized = packet.to_bytes()
    crc = struct.unpack(">I", serialized[-4:])[0]
    return FieldPacket(
        dtype=packet.dtype,
        shape=packet.shape,
        weight_count=packet.weight_count,
        peak_scale=packet.peak_scale,
        sample_rate=packet.sample_rate,
        payload_length=packet.payload_length,
        payload=packet.payload,
        crc32=crc,
        metadata=packet.metadata,
    )


def decode_field(
    packet: Union[FieldPacket, bytes],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Decode a packet back to values and metadata.

    Returns (values, metadata) where values are restored to the original
    peak scale and reshaped to the stored shape.
    """
    if isinstance(packet, (bytes, bytearray)):
        packet = FieldPacket.from_bytes(packet)

    if packet.payload_length == 0 or packet.weight_count == 0:
        empty = np.zeros(packet.shape if packet.shape else (0,), dtype=np.float32)
        return empty, dict(packet.metadata)

    pcm = np.frombuffer(packet.payload, dtype="<i2")
    if pcm.size != packet.weight_count:
        raise ValueError(
            f"payload sample count {pcm.size} != weight_count {packet.weight_count}"
        )
    normalized = decode_pcm16(pcm)
    values = normalized * float(packet.peak_scale)
    if packet.shape and int(np.prod(packet.shape)) == values.size:
        values = values.reshape(packet.shape)
    meta = dict(packet.metadata)
    meta.setdefault("peak_scale", packet.peak_scale)
    meta.setdefault("sample_rate", packet.sample_rate)
    meta.setdefault("dtype", packet.dtype)
    meta.setdefault("weight_count", packet.weight_count)
    return values.astype(np.float32), meta
