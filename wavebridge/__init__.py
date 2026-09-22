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
    # Transport-neutral boundary
    "FieldPacket",
    "encode_field",
    "decode_field",
    "Observation",
    # MetaField-facing API
    "encode_field_state",
    "decode_field_observation",
    # Physical / transport / sync
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
]
