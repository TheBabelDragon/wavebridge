from .channel import (
    ChannelSpec,
    channel_metrics,
    simulate_channel,
)
from .codec import (
    flatten_weights,
    normalize_peak,
    encode_pcm16,
    decode_pcm16,
    quantization_error,
)
from .experiment import (
    SurvivalReport,
    run_weight_survival,
    sweep_channel_survival,
)
from .format import WaveSpec
from .bridge import FieldPacket, encode_field, decode_field
from .observation import Observation
from .metafield import encode_field_state, decode_field_observation
from .physical import (
    PhysicalChannel,
    SimulatedChannel,
    packet_roundtrip,
    receive_packet,
    send_packet,
)
from .sync import SyncError, extract_packet, frame_to_waveform, transmit_framed
from .transport import packet_to_waveform, waveform_to_packet
from .channel_profile import ChannelProfile, OpticalChannelProfile
from .calibration import calibrate_from_probe, probe_waveform
from .hardware.audio import AudioFileChannel, AudioLoopbackChannel
from .hardware.optical import (
    SimulatedOpticalChannel,
    OpticalChannel,
    OpticalBodySimulator,
    OpticalPathSpec,
)

__all__ = [
    "WaveSpec",
    "ChannelSpec",
    "channel_metrics",
    "simulate_channel",
    "flatten_weights",
    "normalize_peak",
    "encode_pcm16",
    "decode_pcm16",
    "quantization_error",
    "SurvivalReport",
    "run_weight_survival",
    "sweep_channel_survival",
    "FieldPacket",
    "encode_field",
    "decode_field",
    "Observation",
    "encode_field_state",
    "decode_field_observation",
    "PhysicalChannel",
    "SimulatedChannel",
    "packet_roundtrip",
    "send_packet",
    "receive_packet",
    "packet_to_waveform",
    "waveform_to_packet",
    "SyncError",
    "extract_packet",
    "frame_to_waveform",
    "transmit_framed",
    "ChannelProfile",
    "OpticalChannelProfile",
    "calibrate_from_probe",
    "probe_waveform",
    "AudioFileChannel",
    "AudioLoopbackChannel",
    "SimulatedOpticalChannel",
    "OpticalChannel",
    "OpticalBodySimulator",
    "OpticalPathSpec",
]
