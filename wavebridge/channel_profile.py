"""
Measured physical channel profile.

ChannelSpec is the *model* used by SimulatedChannel.
OpticalChannelProfile / ChannelProfile is the *measurement* of a real path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ChannelProfile:
    """
    Empirical characterization of a physical path.

    gain, dc_offset, noise_rms, bandwidth_hz, clip_min/max mirror ChannelSpec
    so a profile can be turned into a simulator or used for calibration.
    """

    name: str = "unnamed"
    gain: float = 1.0
    dc_offset: float = 0.0
    noise_rms: float = 0.0
    bandwidth_hz: Optional[float] = None
    clip_min: float = -1.0
    clip_max: float = 1.0
    latency_samples: int = 0
    dynamic_range_db: float = 0.0
    sample_rate: int = 44100
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_channel_spec(self):
        """Project measurement into the existing simulator model."""
        from .channel import ChannelSpec

        return ChannelSpec(
            gain=float(self.gain),
            dc_offset=float(self.dc_offset),
            bandwidth_hz=self.bandwidth_hz,
            noise_rms=float(self.noise_rms),
            clip_min=float(self.clip_min),
            clip_max=float(self.clip_max),
            seed=0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelProfile":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


# Alias for optical substrate docs
OpticalChannelProfile = ChannelProfile
