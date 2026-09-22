"""Automated test for examples/inspect_real_payload.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_inspect_real_payload_roundtrip(tmp_path: Path):
    from examples.inspect_real_payload import inspect_payload

    rng = np.random.default_rng(0)
    arr = rng.normal(0.0, 0.2, size=(32, 16)).astype(np.float32)
    npy_path = tmp_path / "field.npy"
    np.save(npy_path, arr)

    report, metrics = inspect_payload(npy_path)

    assert "WaveBridge real payload" in report
    assert "weight count" in report
    assert "payload bytes" in report
    assert "max abs error" in report
    assert "mean abs error" in report

    assert metrics["weight_count"] == arr.size
    assert metrics["payload_bytes"] > 0
    assert metrics["payload_bytes"] == arr.size * 2  # PCM16
    assert metrics["recovered_shape"] == tuple(arr.shape)
    assert metrics["packet_type"] == "FieldPacket"
    assert metrics["source"] == "metafield"
    assert metrics["max_abs_error"] < 1e-2
    assert metrics["mean_abs_error"] < 1e-3


def test_inspect_real_payload_main_prints(tmp_path: Path, capsys):
    from examples.inspect_real_payload import main

    arr = np.linspace(-0.5, 0.5, 128, dtype=np.float32)
    npy_path = tmp_path / "w.npy"
    np.save(npy_path, arr)

    main([str(npy_path)])
    out = capsys.readouterr().out
    assert "WaveBridge real payload" in out
    assert "weight count" in out
    assert "128" in out
