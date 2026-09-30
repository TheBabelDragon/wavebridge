"""Optional TensorGate numerical boundary for WaveBridge.

Dependency direction:
    WaveBridge  ──optional──→  TensorGate

TensorGate is never required for ordinary WaveBridge use.
Import this module only when TensorGate is installed:

    pip install tensorgate
    from wavebridge.tensorgate_boundary import prepare_weights, observe_recovered

WaveBridge continues to own PCM / FieldPacket / physical transport.
TensorGate owns numerical description, explicit normalization, and comparison.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import tensorgate
    from tensorgate import analyze, compare
    from tensorgate.integrations.wavebridge import (
        from_wavebridge_payload,
        to_wavebridge_payload,
    )
    _HAS_TENSORGATE = True
except ImportError:  # pragma: no cover
    _HAS_TENSORGATE = False


def available() -> bool:
    """Return True if TensorGate is importable."""
    return _HAS_TENSORGATE


def _require() -> None:
    if not _HAS_TENSORGATE:
        raise ImportError(
            "TensorGate is optional. Install it with: pip install tensorgate"
        )


def prepare_weights(
    values: Any,
    *,
    method: str = "symmetric",
    dtype: str = "float32",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Prepare weights via TensorGate for WaveBridge encode_field.

    Returns (numerical_payload, tensorgate_metadata).
    Hand the payload to wavebridge.encode_field / encode_field_state.
    """
    _require()
    return to_wavebridge_payload(values, method=method, dtype=dtype)


def observe_recovered(
    recovered: Any,
    *,
    source_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Describe a numerical array recovered from WaveBridge decode_field."""
    _require()
    return from_wavebridge_payload(recovered, source_meta=source_meta)


def roundtrip_compare(
    original: Any,
    recovered: Any,
) -> Dict[str, Any]:
    """Compare original weights to recovered numerical values via TensorGate."""
    _require()
    return compare(original, recovered)


def describe(values: Any) -> Dict[str, Any]:
    """Produce a TensorGate descriptor for an array."""
    _require()
    return analyze(values).to_dict()
