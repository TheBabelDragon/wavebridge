"""
Framed packet synchronization over a continuous waveform.

Sample-domain frame:

    PREAMBLE | SYNC | LEN_CHIPS (32 bipolar) | BODY | TAIL

LEN_CHIPS encodes body byte-length as 32 bits (MSB first), ±1 per bit.
BODY maps each packet byte to one sample in [-1, 1] (binary-safe).

Receiver correlates for SYNC; AGC normalizes gain from SYNC amplitude.
Fails explicitly on missing sync / bad length / CRC — never silent success.
"""

from __future__ import annotations

import numpy as np

from .bridge import FieldPacket

_BARKER13 = np.array(
    [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
    dtype=np.float32,
)
SYNC_SAMPLES = np.tile(_BARKER13, 4)  # 52
PREAMBLE_SAMPLES = np.zeros(32, dtype=np.float32)
TAIL_SAMPLES = -SYNC_SAMPLES
_LEN_BITS = 32


class SyncError(ValueError):
    """Raised when framing / sync / integrity recovery fails."""


def _u32_to_chips(n: int) -> np.ndarray:
    n = int(n) & 0xFFFFFFFF
    chips = np.empty(_LEN_BITS, dtype=np.float32)
    for i in range(_LEN_BITS):
        bit = (n >> (31 - i)) & 1
        chips[i] = 1.0 if bit else -1.0
    return chips


def _chips_to_u32(chips: np.ndarray) -> int:
    chips = np.asarray(chips, dtype=np.float32).reshape(-1)[:_LEN_BITS]
    if chips.size < _LEN_BITS:
        raise SyncError("length chips truncated")
    n = 0
    for i in range(_LEN_BITS):
        bit = 1 if float(chips[i]) >= 0.0 else 0
        n = (n << 1) | bit
    return n


def _bytes_to_body_samples(data: bytes) -> np.ndarray:
    """Map each byte to one sample in [-1, 1]. Exact inverse exists."""
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    return ((arr - 127.5) / 127.5).astype(np.float32)


def _body_samples_to_bytes(samples: np.ndarray, nbytes: int) -> bytes:
    s = np.asarray(samples, dtype=np.float64).reshape(-1)[:nbytes]
    s = np.clip(s, -1.0, 1.0)
    arr = np.rint(s * 127.5 + 127.5).astype(np.int16)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr.tobytes()


def frame_to_waveform(packet: FieldPacket) -> np.ndarray:
    body_bytes = packet.to_bytes()
    body = _bytes_to_body_samples(body_bytes)
    length_chips = _u32_to_chips(len(body_bytes))
    return np.concatenate(
        [PREAMBLE_SAMPLES, SYNC_SAMPLES, length_chips, body, TAIL_SAMPLES]
    )


def transmit_framed(
    packet: FieldPacket,
    *,
    leading_silence: int = 0,
    trailing_silence: int = 0,
) -> np.ndarray:
    wave = frame_to_waveform(packet)
    parts = []
    if leading_silence > 0:
        parts.append(np.zeros(int(leading_silence), dtype=np.float32))
    parts.append(wave)
    if trailing_silence > 0:
        parts.append(np.zeros(int(trailing_silence), dtype=np.float32))
    return np.concatenate(parts) if len(parts) > 1 else wave


def _correlate_sync(samples: np.ndarray) -> int:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    s = SYNC_SAMPLES
    n = s.size
    if x.size < n:
        return -1
    s0 = s - float(np.mean(s))
    denom_s = float(np.sqrt(np.sum(s0 * s0))) + 1e-12
    best_idx = -1
    best_score = -1.0
    for i in range(0, x.size - n + 1):
        w = x[i : i + n]
        w0 = w - float(np.mean(w))
        denom = denom_s * (float(np.sqrt(np.sum(w0 * w0))) + 1e-12)
        score = float(np.dot(s0, w0) / denom)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_score < 0.85:
        return -1
    return best_idx


def extract_packet(waveform: np.ndarray) -> FieldPacket:
    x = np.asarray(waveform, dtype=np.float32).reshape(-1)
    idx = _correlate_sync(x)
    if idx < 0:
        raise SyncError("sync pattern not found")

    captured_sync = x[idx : idx + SYNC_SAMPLES.size]
    amp = float(np.max(np.abs(captured_sync))) or 1.0
    x = x / amp

    pos = idx + SYNC_SAMPLES.size
    if pos + _LEN_BITS > x.size:
        raise SyncError("truncated after sync (missing length)")

    body_nbytes = _chips_to_u32(x[pos : pos + _LEN_BITS])
    if body_nbytes <= 0 or body_nbytes > 10_000_000:
        raise SyncError(f"implausible body length: {body_nbytes}")

    pos += _LEN_BITS
    if pos + body_nbytes > x.size:
        raise SyncError(
            f"truncated body: need {body_nbytes} samples, have {x.size - pos}"
        )
    body_bytes = _body_samples_to_bytes(x[pos : pos + body_nbytes], body_nbytes)

    try:
        return FieldPacket.from_bytes(body_bytes)
    except ValueError as exc:
        raise SyncError(f"packet integrity failed: {exc}") from exc
