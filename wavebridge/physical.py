"""
Physical channel boundary.

    FieldPacket
        ↓  transport + sync framing
    waveform
        ↓  PhysicalChannel.transmit
    physical medium
        ↓  PhysicalChannel.receive
    waveform
        ↓  sync extract + CRC
    FieldPacket / Observation

Acceptance: a packet through the channel either decodes successfully
or fails explicitly (CRC / sync / integrity).  No silent reconstruction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .bridge import FieldPacket, decode_field
from .channel import ChannelSpec, simulate_channel
from .observation import Observation
from .sync import SyncError, extract_packet, transmit_framed


class PhysicalChannel(ABC):
    """Transport-neutral physical medium interface."""

    @abstractmethod
    def transmit(self, waveform: np.ndarray) -> None:
        """Send a normalized waveform into the medium."""

    @abstractmethod
    def receive(self) -> np.ndarray:
        """Capture a normalized waveform from the medium."""


class SimulatedChannel(PhysicalChannel):
    """
    In-memory channel using the existing ChannelSpec model.

    transmit() stores the waveform after applying gain / DC / bandwidth /
    noise / clip.  receive() returns the last stored capture.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        spec: Optional[ChannelSpec] = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.spec = spec or ChannelSpec()
        self._buffer: Optional[np.ndarray] = None

    def transmit(self, waveform: np.ndarray) -> None:
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        self._buffer = simulate_channel(
            samples,
            sample_rate=self.sample_rate,
            spec=self.spec,
        )

    def receive(self) -> np.ndarray:
        if self._buffer is None:
            raise RuntimeError("SimulatedChannel.receive() before transmit()")
        return self._buffer.copy()


def send_packet(
    packet: FieldPacket,
    channel: PhysicalChannel,
    *,
    leading_silence: int = 0,
    trailing_silence: int = 0,
) -> None:
    """Frame a packet and transmit the waveform on the channel."""
    wave = transmit_framed(
        packet,
        leading_silence=leading_silence,
        trailing_silence=trailing_silence,
    )
    channel.transmit(wave)


def receive_packet(channel: PhysicalChannel) -> FieldPacket:
    """
    Receive waveform and extract a FieldPacket.

    Raises SyncError / ValueError on sync or CRC failure.
    """
    wave = channel.receive()
    return extract_packet(wave)


def packet_roundtrip(
    packet: FieldPacket,
    channel: PhysicalChannel,
    *,
    leading_silence: int = 0,
    trailing_silence: int = 0,
) -> Observation:
    """
    Full path:

        packet → framed waveform → channel → waveform → packet → values

    Returns an Observation on success.
    Raises SyncError / ValueError on integrity failure (never silent).
    """
    send_packet(
        packet,
        channel,
        leading_silence=leading_silence,
        trailing_silence=trailing_silence,
    )
    recovered = receive_packet(channel)
    values, meta = decode_field(recovered)
    return Observation(
        values=values,
        source="physical",
        channel=type(channel).__name__,
        metadata=meta,
    )
