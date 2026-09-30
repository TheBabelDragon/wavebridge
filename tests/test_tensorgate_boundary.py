"""Optional TensorGate boundary tests for WaveBridge.

Skipped entirely when tensorgate is not installed.
Existing WaveBridge tests must not import this module.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorgate")

from wavebridge import encode_field, decode_field
from wavebridge.tensorgate_boundary import (
    available,
    describe,
    observe_recovered,
    prepare_weights,
    roundtrip_compare,
)


def test_tensorgate_available():
    assert available() is True


def test_ordinary_wavebridge_path_unaffected():
    """encode_field / decode_field work without calling TensorGate."""
    w = np.array([0.1, -0.5, 0.9], dtype=np.float32)
    packet = encode_field(w)
    recovered, meta = decode_field(packet)
    assert recovered.shape == w.shape
    assert float(np.max(np.abs(w - recovered))) < 1e-3


def test_prepare_and_observe_roundtrip():
    rng = np.random.default_rng(42)
    weights = rng.normal(0, 0.3, size=64).astype(np.float32)
    weights[0] = 2.5  # outlier

    payload, tg_meta = prepare_weights(weights, method="symmetric")
    assert "tensorgate_descriptor" in tg_meta
    assert "normalization" in tg_meta
    assert "provenance" in tg_meta["normalization"]

    packet = encode_field(payload, metadata={"tensorgate": tg_meta})
    recovered, wb_meta = decode_field(packet)

    observed, obs_meta = observe_recovered(recovered, source_meta=tg_meta)
    assert "tensorgate_descriptor" in obs_meta

    err = roundtrip_compare(payload.astype(np.float32), recovered)
    assert err["max_absolute_error"] < 0.01
    assert err["changed_element_count"] >= 0


def test_no_silent_mutation_in_prepare():
    w = np.array([-0.8, 0.2, 0.7], dtype=np.float32)
    payload, meta = prepare_weights(w, method="symmetric")
    assert meta["normalization"]["scale"] is not None
    assert meta["source_descriptor"]["content_hash"]
    assert meta["tensorgate_descriptor"]["content_hash"]


def test_describe():
    w = np.ones(4, dtype=np.float32)
    d = describe(w)
    assert d["element_count"] == 4
    assert d["content_hash"]
