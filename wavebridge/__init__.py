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
]
