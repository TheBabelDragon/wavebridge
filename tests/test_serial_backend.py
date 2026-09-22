"""ESP32 serial backend tests (dry-run only — no hardware)."""

from __future__ import annotations

import numpy as np

from wavebridge.hardware.serial_backend import ESP32SerialBackend, SerialConfig


def test_dry_run_excite():
    backend = ESP32SerialBackend(SerialConfig(dry_run=True, n_detectors=16))
    with backend:
        assert backend.ping() is True
        res = backend.excite(3, drive_level=0.9)
    assert res.ok
    assert res.laser_id == 3
    assert res.detector_response.shape == (16,)
    assert float(np.max(res.detector_response)) > 0.1


def test_dry_run_transmit_receive():
    backend = ESP32SerialBackend(SerialConfig(dry_run=True, n_detectors=12))
    backend.open()
    wave = np.zeros(64, dtype=np.float32)
    wave[5] = 1.0
    backend.transmit(wave)
    rx = backend.receive()
    assert rx.shape == (12,)
    backend.close()


def test_regions_parser():
    backend = ESP32SerialBackend(SerialConfig(dry_run=True, n_detectors=4))
    line = (
        '{"body_id":"x","body_type":"optical","field_regions":['
        '{"region":"d0","observed":0.1},'
        '{"region":"d1","observed":0.2},'
        '{"region":"d2","observed":0.3},'
        '{"region":"d3","observed":0.4}]}'
    )
    res = backend._parse_observation_line(line, laser_id=1)
    assert res.ok
    assert list(np.round(res.detector_response, 1)) == [0.1, 0.2, 0.3, 0.4]
