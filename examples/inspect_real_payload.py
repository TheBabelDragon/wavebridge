"""Inspect a real NumPy field through the public WaveBridge API.

Usage:
    python examples/inspect_real_payload.py examples/test_weights.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from wavebridge import decode_field, encode_field_state


def inspect_payload(
    path: str | Path,
) -> Tuple[str, Dict[str, Any]]:
    """
    Round-trip a .npy field through encode_field_state / decode_field.

    Returns (report_text, metrics_dict) for tests and CLI use.
    """
    path = Path(path)
    original = np.load(path)
    packet = encode_field_state(
        original,
        source="metafield",
    )
    recovered, metadata = decode_field(packet)
    error = np.abs(original.astype(np.float32) - recovered)

    metrics: Dict[str, Any] = {
        "path": str(path),
        "original_shape": tuple(original.shape),
        "original_dtype": str(original.dtype),
        "weight_count": int(original.size),
        "packet_type": type(packet).__name__,
        "payload_bytes": len(packet.payload),
        "recovered_shape": tuple(recovered.shape),
        "recovered_dtype": str(recovered.dtype),
        "max_abs_error": float(error.max()) if error.size else 0.0,
        "mean_abs_error": float(error.mean()) if error.size else 0.0,
        "peak_scale": float(packet.peak_scale),
        "sample_rate": int(packet.sample_rate),
        "source": metadata.get("source"),
    }

    lines = [
        "=== WaveBridge real payload ===",
        f"input:              {metrics['path']}",
        f"original shape:     {metrics['original_shape']}",
        f"original dtype:     {metrics['original_dtype']}",
        f"weight count:       {metrics['weight_count']}",
        f"packet type:        {metrics['packet_type']}",
        f"payload bytes:      {metrics['payload_bytes']}",
        f"recovered shape:    {metrics['recovered_shape']}",
        f"recovered dtype:    {metrics['recovered_dtype']}",
        f"max abs error:      {metrics['max_abs_error']}",
        f"mean abs error:     {metrics['mean_abs_error']}",
        f"peak scale:         {metrics['peak_scale']}",
        f"sample rate:        {metrics['sample_rate']}",
        f"source:             {metrics['source']}",
    ]
    report = "\n".join(lines)
    return report, metrics


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Round-trip a real field through WaveBridge."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="examples/test_weights.npy",
    )
    args = parser.parse_args(argv)
    report, _ = inspect_payload(args.path)
    print(report)


if __name__ == "__main__":
    main()
