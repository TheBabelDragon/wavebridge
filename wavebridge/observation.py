"""
Neutral observation structure for WaveBridge.

WaveBridge knows nothing about MetaField internals.  This is a small
transport-side container that carries recovered values plus provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Observation:
    """
    values   – recovered field samples (numpy array)
    source   – who produced / observed (e.g. "optical", "audio", "sim")
    channel  – optional channel identifier
    metadata – free-form dict (sample_rate, peak_scale, …)
    """

    values: np.ndarray
    source: str = "unknown"
    channel: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float32)
        if self.metadata is None:
            self.metadata = {}
