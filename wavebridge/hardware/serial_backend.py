"""
ESP32 / optical-body serial backend.

Protocol (optical-body-s3 style):

  Host → board:  EXCITE <laser_id>\n
  Board → host:  one JSON FieldObservation line

Also supports PING and optional WAVE <n> + float32 samples.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SerialConfig:
    port: str = "/dev/ttyUSB0"
    baud: int = 115200
    timeout_s: float = 2.0
    read_timeout_s: float = 3.0
    protocol: str = "excite"  # excite | wave
    n_detectors: int = 20
    dry_run: bool = False


@dataclass
class SerialExcitationResult:
    laser_id: int
    detector_response: np.ndarray
    raw_line: str = ""
    observation: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""


class ESP32SerialBackend:
    """Low-level serial I/O for optical-body firmware."""

    def __init__(self, config: Optional[SerialConfig] = None, **kwargs: Any) -> None:
        if config is None:
            config = SerialConfig(**kwargs) if kwargs else SerialConfig()
        else:
            for k, v in kwargs.items():
                setattr(config, k, v)
        self.config = config
        self._ser = None
        self._synthetic_rng = np.random.default_rng(0)
        self._last_excite: Optional[SerialExcitationResult] = None

    def open(self) -> None:
        if self.config.dry_run:
            return
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pyserial required for hardware serial.  pip install pyserial"
            ) from exc
        self._ser = serial.Serial(
            self.config.port,
            self.config.baud,
            timeout=self.config.timeout_s,
        )
        time.sleep(0.15)
        self._ser.reset_input_buffer()

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def __enter__(self) -> "ESP32SerialBackend":
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def transmit(self, waveform: np.ndarray) -> None:
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if self.config.protocol == "wave":
            self._send_wave(samples)
        else:
            laser = 0 if samples.size == 0 else int(np.clip(np.argmax(np.abs(samples)) % 64, 0, 63))
            self._last_excite = self.excite(laser)

    def receive(self) -> np.ndarray:
        if self._last_excite is not None:
            return np.asarray(self._last_excite.detector_response, dtype=np.float32).reshape(-1)
        return np.zeros(self.config.n_detectors, dtype=np.float32)

    def excite(self, laser_id: int, drive_level: float = 1.0) -> SerialExcitationResult:
        laser_id = int(laser_id)
        if self.config.dry_run:
            return self._dry_excite(laser_id, drive_level)
        if self._ser is None:
            self.open()
        assert self._ser is not None
        self._ser.write(f"EXCITE {laser_id}\n".encode("utf-8"))
        self._ser.flush()
        line = self._readline(timeout_s=self.config.read_timeout_s)
        if not line:
            return SerialExcitationResult(
                laser_id=laser_id,
                detector_response=np.zeros(self.config.n_detectors, dtype=np.float32),
                ok=False,
                error="timeout waiting for FieldObservation JSON",
            )
        return self._parse_observation_line(line, laser_id)

    def ping(self) -> bool:
        if self.config.dry_run:
            return True
        if self._ser is None:
            self.open()
        assert self._ser is not None
        self._ser.write(b"PING\n")
        self._ser.flush()
        line = self._readline(timeout_s=1.5)
        if not line:
            return False
        low = line.lower()
        return any(tok in low for tok in ("ok", "pong", "ready", "alive"))

    def _send_wave(self, samples: np.ndarray) -> None:
        if self.config.dry_run:
            return
        if self._ser is None:
            self.open()
        assert self._ser is not None
        n = int(samples.size)
        self._ser.write(f"WAVE {n}\n".encode("utf-8"))
        self._ser.write(struct.pack(f"<{n}f", *samples.tolist()))
        self._ser.flush()

    def _readline(self, timeout_s: float) -> str:
        assert self._ser is not None
        deadline = time.time() + timeout_s
        buf = bytearray()
        while time.time() < deadline:
            waiting = getattr(self._ser, "in_waiting", 0) or 0
            if waiting:
                chunk = self._ser.read(waiting)
                buf.extend(chunk)
                if b"\n" in buf:
                    line, _, _ = buf.partition(b"\n")
                    return line.decode("utf-8", errors="replace").strip()
            else:
                time.sleep(0.01)
        if buf:
            return buf.decode("utf-8", errors="replace").strip()
        return ""

    def _parse_observation_line(self, line: str, laser_id: int) -> SerialExcitationResult:
        if not line.startswith("{"):
            start = line.find("{")
            if start >= 0:
                line = line[start:]
            else:
                return SerialExcitationResult(
                    laser_id=laser_id,
                    detector_response=np.zeros(self.config.n_detectors, dtype=np.float32),
                    raw_line=line,
                    ok=False,
                    error=f"non-JSON response: {line[:80]}",
                )
        try:
            obs = json.loads(line)
        except json.JSONDecodeError as exc:
            return SerialExcitationResult(
                laser_id=laser_id,
                detector_response=np.zeros(self.config.n_detectors, dtype=np.float32),
                raw_line=line,
                ok=False,
                error=f"json decode: {exc}",
            )
        response = self._regions_to_vector(obs)
        return SerialExcitationResult(
            laser_id=laser_id,
            detector_response=response,
            raw_line=line,
            observation=obs,
            ok=True,
        )

    def _regions_to_vector(self, obs: Dict[str, Any]) -> np.ndarray:
        regions = obs.get("field_regions") or obs.get("regions") or []
        if regions:
            vals = []
            for r in regions:
                if isinstance(r, dict):
                    v = r.get("observed", r.get("value", 0.0))
                else:
                    v = r
                vals.append(float(v if v is not None else 0.0))
            return np.asarray(vals, dtype=np.float32)
        flat = obs.get("observed_response") or obs.get("detectors")
        if flat is not None:
            return np.asarray(flat, dtype=np.float32).reshape(-1)
        return np.zeros(self.config.n_detectors, dtype=np.float32)

    def _dry_excite(self, laser_id: int, drive_level: float) -> SerialExcitationResult:
        n = self.config.n_detectors
        idx = np.arange(n, dtype=np.float32)
        center = float(laser_id % max(n, 1))
        base = np.exp(-0.25 * (idx - center) ** 2) * float(np.clip(drive_level, 0, 1))
        noise = self._synthetic_rng.normal(0, 0.01, size=n).astype(np.float32)
        resp = np.clip(base + 0.02 + noise, 0, 1).astype(np.float32)
        return SerialExcitationResult(
            laser_id=laser_id,
            detector_response=resp,
            observation={
                "body_id": "dry-run",
                "body_type": "optical",
                "excitation_id": laser_id,
                "field_regions": [
                    {"region": f"detector_{i:02d}", "observed": float(resp[i])}
                    for i in range(n)
                ],
            },
            ok=True,
        )
