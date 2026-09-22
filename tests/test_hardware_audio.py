"""Audio file channel + packet roundtrip through WAV (no live device required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wavebridge.bridge import encode_field, decode_field
from wavebridge.hardware.audio import AudioFileChannel, wav_roundtrip_paths
from wavebridge.physical import packet_roundtrip, send_packet, receive_packet
from wavebridge.sync import frame_to_waveform, extract_packet


def test_wav_file_roundtrip_samples(tmp_path: Path):
    tx = tmp_path / "t.wav"
    wave = np.linspace(-0.5, 0.5, 512, dtype=np.float32)
    recovered = wav_roundtrip_paths(wave, tx, sample_rate=44100)
    assert recovered.shape == wave.shape
    err = float(np.max(np.abs(recovered - wave)))
    assert err < 1e-3


def test_packet_through_audio_file_channel(tmp_path: Path):
    """FieldPacket → framed wave → WAV → read → extract packet."""
    values = np.array([0.1, -0.3, 0.7, -0.9, 0.0, 0.5], dtype=np.float32)
    packet = encode_field(values)
    path = tmp_path / "packet.wav"

    ch = AudioFileChannel(path, sample_rate=44100)
    obs = packet_roundtrip(packet, ch, leading_silence=64, trailing_silence=32)
    assert obs.values.shape == values.shape
    assert float(np.max(np.abs(obs.values - values))) < 1e-3


def test_separate_tx_rx_paths(tmp_path: Path):
    """Simulate external record by copying tx → rx."""
    values = np.linspace(-1, 1, 40, dtype=np.float32)
    packet = encode_field(values)
    tx = tmp_path / "out.wav"
    rx = tmp_path / "in.wav"

    ch = AudioFileChannel(tx, rx_path=rx, sample_rate=44100)
    send_packet(packet, ch, leading_silence=20)
    rx.write_bytes(tx.read_bytes())
    recovered = receive_packet(ch)
    arr, _ = decode_field(recovered)
    assert float(np.max(np.abs(arr - values))) < 1e-3


def test_receive_missing_file_raises(tmp_path: Path):
    ch = AudioFileChannel(tmp_path / "missing.wav")
    with pytest.raises(FileNotFoundError):
        ch.receive()


def test_loopback_import_message():
    from wavebridge.hardware.audio import AudioLoopbackChannel

    ch = AudioLoopbackChannel()
    try:
        import sounddevice  # noqa: F401
        pytest.skip("sounddevice installed; skip import-error path")
    except ImportError:
        with pytest.raises(ImportError, match="sounddevice"):
            ch.transmit(np.zeros(100, dtype=np.float32))
