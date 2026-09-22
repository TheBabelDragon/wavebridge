"""
MetaField-facing API for WaveBridge.

WaveBridge knows nothing about MetaField internals.
It only knows: array → physical representation  and  physical → array.

Public surface:

    from wavebridge.metafield import encode_field_state
    from wavebridge.metafield import decode_field_observation
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from .bridge import FieldPacket, decode_field, encode_field
from .observation import Observation


def encode_field_state(
    values: Any,
    *,
    source: str = "metafield",
    metadata: Optional[Mapping[str, Any]] = None,
    sample_rate: int = 44100,
) -> FieldPacket:
    """
    Encode a field-state array into a transport-neutral WaveBridge packet.

    Parameters
    ----------
    values :
        Array-like field state (weights, excitation amplitudes, …).
    source :
        Provenance tag stored in packet metadata (default "metafield").
    metadata :
        Optional extra metadata merged into the packet.
    sample_rate :
        Logical sample rate for the eventual waveform (default 44.1 kHz).

    Returns
    -------
    FieldPacket
        Logical packet ready for PCM / PAM / optical transport.
    """
    meta: Dict[str, Any] = dict(metadata or {})
    meta.setdefault("source", source)
    return encode_field(values, metadata=meta, sample_rate=sample_rate)


def decode_field_observation(
    values: Any,
    *,
    source: str = "optical",
    metadata: Optional[Mapping[str, Any]] = None,
    channel: Optional[str] = None,
) -> Observation:
    """
    Wrap recovered values (or a packet) as a neutral Observation.

    Accepts either:
      - a FieldPacket / bytes → full decode
      - an already-decoded array → wrap only

    WaveBridge does not interpret MetaField geometry; it only returns
    the array + provenance.
    """
    meta: Dict[str, Any] = dict(metadata or {})
    meta.setdefault("source", source)

    if isinstance(values, (FieldPacket, bytes, bytearray)):
        arr, pkt_meta = decode_field(values)
        meta.update(pkt_meta)
        return Observation(
            values=arr,
            source=source,
            channel=channel,
            metadata=meta,
        )

    arr = np.asarray(values, dtype=np.float32)
    return Observation(
        values=arr,
        source=source,
        channel=channel,
        metadata=meta,
    )
