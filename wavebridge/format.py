from dataclasses import dataclass
@dataclass(frozen=True)
class WaveSpec:
    sample_rate: int = 44100
    channels: int = 1
    sample_width: int = 2
    normalization: str = "peak"
    version: str = "WB1"
