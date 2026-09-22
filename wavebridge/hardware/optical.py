"""
Optical physical channel (PAM / laser / free-space / BPW34 path).

Packet transport (same contract as SimulatedChannel / AudioFileChannel):

    FieldPacket → framed waveform
        → OpticalChannel.transmit
        → free-space optical path
        → OpticalChannel.receive
        → framed waveform → packet

For *digital packet* integrity the optical path must stay linear enough
that SYNC / length chips / body bytes survive.  Optical nonlinearity and
multi-detector geometry live in OpticalBodySimulator (characterization).

Hardware (ESP32 serial, real laser drivers) is intentionally not wired
here.  SerialOpticalChannel raises until a backend is provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..channel import ChannelSpec, simulate_channel
from ..physical import PhysicalChannel


@dataclass
class OpticalPathSpec:
    """
    Linear path model for packet transport over an optical link.

    Keep the path linear for digital framing.  Use OpticalBodySimulator
    when you need LED compression / multi-detector geometry.
    """

    gain: float = 0.98
    dark_current: float = 0.0
    noise_rms: float = 0.0005
    bandwidth_hz: Optional[float] = None
    seed: int = 0


class SimulatedOpticalChannel(PhysicalChannel):
    """
    In-memory optical path for packet roundtrips.

    Linear gain + optional noise/bandwidth (same family as SimulatedChannel)
    so framing survives.  Represents an AC-coupled photodiode + TIA + ADC path.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        path: Optional[OpticalPathSpec] = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.path = path or OpticalPathSpec()
        self._buffer: Optional[np.ndarray] = None

    def transmit(self, waveform: np.ndarray) -> None:
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        spec = ChannelSpec(
            gain=float(self.path.gain),
            dc_offset=float(self.path.dark_current),
            bandwidth_hz=self.path.bandwidth_hz,
            noise_rms=float(self.path.noise_rms),
            seed=int(self.path.seed),
        )
        self._buffer = simulate_channel(
            samples,
            sample_rate=self.sample_rate,
            spec=spec,
        )

    def receive(self) -> np.ndarray:
        if self._buffer is None:
            raise RuntimeError("SimulatedOpticalChannel.receive() before transmit()")
        return self._buffer.copy()


class SerialOpticalChannel(PhysicalChannel):
    """
    Placeholder for ESP32 / serial optical body.

    Does not implement a protocol in this commit.  Raises until a
    concrete backend is injected via `backend`.
    """

    def __init__(self, backend: Optional[Any] = None) -> None:
        self.backend = backend

    def transmit(self, waveform: np.ndarray) -> None:
        if self.backend is None:
            raise NotImplementedError(
                "SerialOpticalChannel requires a backend "
                "(ESP32 serial / Duckoustic). "
                "Use SimulatedOpticalChannel for software tests."
            )
        self.backend.transmit(np.asarray(waveform, dtype=np.float32).reshape(-1))

    def receive(self) -> np.ndarray:
        if self.backend is None:
            raise NotImplementedError(
                "SerialOpticalChannel requires a backend. "
                "Use SimulatedOpticalChannel for software tests."
            )
        return np.asarray(self.backend.receive(), dtype=np.float32).reshape(-1)


OpticalChannel = SimulatedOpticalChannel


@dataclass
class OpticalExcitationRecord:
    """One measured laser → BPW34-vector observation."""

    laser_id: int
    drive_level: float
    detector_response: np.ndarray
    ambient: float = 0.0
    wavelength_nm: Optional[float] = None
    geometry_state: str = "unknown"
    timestamp: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "laser_id": int(self.laser_id),
            "drive_level": float(self.drive_level),
            "detector_response": np.asarray(
                self.detector_response, dtype=np.float32
            ).tolist(),
            "ambient": float(self.ambient),
            "wavelength_nm": self.wavelength_nm,
            "geometry_state": self.geometry_state,
            "timestamp": self.timestamp,
            "extras": dict(self.extras),
        }


class OpticalBodySimulator:
    """
    Software stand-in for a multi-laser / multi-detector optical body.

    Coupling is a fixed random matrix (seeded) so MetaField closed loops
    can learn a stable geometry without hardware.

    This is the characterization substrate (Phase 10–12), not the
    digital packet transport path.
    """

    def __init__(
        self,
        n_lasers: int = 12,
        n_detectors: int = 20,
        *,
        seed: int = 42,
        noise_rms: float = 0.01,
        ambient: float = 0.02,
    ) -> None:
        self.n_lasers = int(n_lasers)
        self.n_detectors = int(n_detectors)
        self.noise_rms = float(noise_rms)
        self.ambient = float(ambient)
        rng = np.random.default_rng(seed)
        coupling = np.zeros((n_detectors, n_lasers), dtype=np.float32)
        for d in range(n_detectors):
            for L in range(n_lasers):
                dist = abs(L - (d % n_lasers))
                coupling[d, L] = np.exp(-0.35 * dist) * (
                    0.2 + 0.8 * rng.random()
                )
        self.coupling = coupling
        self._rng = np.random.default_rng(seed + 1)

    def excite(
        self,
        laser_id: int,
        drive_level: float = 1.0,
        *,
        geometry_state: str = "nominal",
    ) -> OpticalExcitationRecord:
        if not (0 <= laser_id < self.n_lasers):
            raise ValueError(f"laser_id {laser_id} out of range [0, {self.n_lasers})")
        drive = float(np.clip(drive_level, 0.0, 1.0))
        power = float(np.tanh(drive * 1.5))
        base = self.coupling[:, laser_id] * power
        noise = self._rng.normal(0.0, self.noise_rms, size=self.n_detectors).astype(
            np.float32
        )
        response = np.clip(base + self.ambient + noise, 0.0, 1.0).astype(np.float32)
        return OpticalExcitationRecord(
            laser_id=laser_id,
            drive_level=drive,
            detector_response=response,
            ambient=self.ambient,
            geometry_state=geometry_state,
        )

    def excite_many(
        self,
        drives: Sequence[float],
        *,
        geometry_state: str = "nominal",
    ) -> OpticalExcitationRecord:
        """Simultaneous multi-laser excitation (mixed field)."""
        drives_arr = np.asarray(drives, dtype=np.float32).reshape(-1)
        if drives_arr.size != self.n_lasers:
            raise ValueError(
                f"drives length {drives_arr.size} != n_lasers {self.n_lasers}"
            )
        power = np.tanh(np.clip(drives_arr, 0.0, 1.0) * 1.5).astype(np.float32)
        base = self.coupling @ power
        noise = self._rng.normal(0.0, self.noise_rms, size=self.n_detectors).astype(
            np.float32
        )
        response = np.clip(base + self.ambient + noise, 0.0, 1.0).astype(np.float32)
        return OpticalExcitationRecord(
            laser_id=-1,
            drive_level=float(np.mean(drives_arr)),
            detector_response=response,
            ambient=self.ambient,
            geometry_state=geometry_state,
            extras={"drives": drives_arr.tolist()},
        )

    def estimate_transfer_matrix(
        self,
        drive_level: float = 1.0,
    ) -> np.ndarray:
        """Fire each laser once; stack detector responses as columns."""
        cols = []
        for L in range(self.n_lasers):
            rec = self.excite(L, drive_level=drive_level)
            cols.append(rec.detector_response)
        return np.column_stack(cols).astype(np.float32)
