"""Frozen Pydantic v2 validator for MUSE ``InstrumentPatch`` payloads."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


_STRICT_CONFIG = ConfigDict(extra="forbid", frozen=True)


class EnvelopeSpecV1(BaseModel):
    model_config = _STRICT_CONFIG

    attack_ms: float = Field(default=10.0, ge=0.0)
    decay_ms: float = Field(default=80.0, ge=0.0)
    sustain_level: float = Field(default=0.7, ge=0.0, le=1.0)
    release_ms: float = Field(default=200.0, ge=0.0)
    curve: Literal["linear", "exp"] = "exp"
    amount: float = 1.0


class VelocityMapV1(BaseModel):
    model_config = _STRICT_CONFIG

    amp: float = 1.0
    cutoff_delta_hz: float = 0.0
    transient_level: float = 0.0
    noise_level: float = 0.0


class OscillatorSpecV1(BaseModel):
    model_config = _STRICT_CONFIG

    algorithm: Literal[
        "sine",
        "triangle",
        "saw",
        "square",
        "pulse",
        "noise",
        "fm",
        "karplus_strong",
    ] = "sine"
    level: float = Field(default=1.0, ge=0.0)
    detune_cents: float = 0.0
    phase_offset: float = 0.0
    duty_cycle: float = Field(default=0.5, ge=0.0, le=1.0)


class FilterSpecV1(BaseModel):
    model_config = _STRICT_CONFIG

    mode: Literal["lowpass", "highpass", "bandpass"] = "lowpass"
    cutoff_hz: float = Field(default=18000.0, gt=0.0)
    resonance: float = Field(default=0.0, ge=0.0)
    drive: float = Field(default=0.0, ge=0.0)


class LfoSpecV1(BaseModel):
    model_config = _STRICT_CONFIG

    target: Literal["amp", "cutoff", "pitch", "pan"] = "amp"
    shape: Literal["sine", "triangle", "square", "sample_hold"] = "sine"
    rate_hz: float = Field(default=0.0, ge=0.0)
    depth: float = Field(default=0.0, ge=0.0)
    phase_offset: float = 0.0


class NoiseLayerSpecV1(BaseModel):
    model_config = _STRICT_CONFIG

    color: Literal["white", "pink", "filtered"] = "filtered"
    level: float = Field(default=0.0, ge=0.0)
    attack_ms: float = Field(default=0.0, ge=0.0)
    decay_ms: float = Field(default=20.0, ge=0.0)
    band_low_hz: float = Field(default=200.0, gt=0.0)
    band_high_hz: float = Field(default=6000.0, gt=0.0)


class TransientLayerSpecV1(BaseModel):
    model_config = _STRICT_CONFIG

    source: Literal["pick", "hammer", "click", "breath"] = "pick"
    level: float = Field(default=0.0, ge=0.0)
    duration_ms: float = Field(default=10.0, ge=0.0)
    band_low_hz: float = Field(default=400.0, gt=0.0)
    band_high_hz: float = Field(default=5000.0, gt=0.0)


class SaturationSpecV1(BaseModel):
    model_config = _STRICT_CONFIG

    model: Literal["none", "tanh", "soft_clip", "tape"] = "none"
    drive: float = Field(default=0.0, ge=0.0)
    mix: float = Field(default=0.0, ge=0.0, le=1.0)


class SampleLayerSpecV1(BaseModel):
    model_config = _STRICT_CONFIG

    source_id: str = ""
    path_hint: str = ""
    root_midi_note: Optional[int] = Field(default=None, ge=0, le=127)
    low_velocity: int = Field(default=1, ge=0, le=127)
    high_velocity: int = Field(default=127, ge=0, le=127)
    gain_db: float = 0.0
    start_offset_ms: float = Field(default=0.0, ge=0.0)
    one_shot: bool = False
    category_hint: str = ""


class FallbackBindingV1(BaseModel):
    model_config = _STRICT_CONFIG

    gm_program: Optional[int] = Field(default=None, ge=0, le=127)
    category_preferences: List[str] = Field(default_factory=list)
    instrument_id_hints: List[str] = Field(default_factory=list)
    sample_search_tags: List[str] = Field(default_factory=list)
    allow_builtin_synth: bool = True


class PatchProfileV1(BaseModel):
    model_config = _STRICT_CONFIG

    family: str = Field(..., min_length=1)
    genre_tags: List[str] = Field(default_factory=list)
    brightness: float = Field(default=0.5, ge=0.0, le=1.0)
    warmth: float = Field(default=0.5, ge=0.0, le=1.0)
    punch: float = Field(default=0.5, ge=0.0, le=1.0)
    richness: float = Field(default=0.5, ge=0.0, le=1.0)
    noise_level: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: List[str] = Field(default_factory=list)


class SynthesisVoiceV1(BaseModel):
    model_config = _STRICT_CONFIG

    oscillators: List[OscillatorSpecV1] = Field(default_factory=list)
    amp_envelope: EnvelopeSpecV1 = Field(default_factory=EnvelopeSpecV1)
    filter: Optional[FilterSpecV1] = None
    filter_envelope: Optional[EnvelopeSpecV1] = None
    lfos: List[LfoSpecV1] = Field(default_factory=list)
    velocity_map: VelocityMapV1 = Field(default_factory=VelocityMapV1)
    noise_layer: Optional[NoiseLayerSpecV1] = None
    transient_layer: Optional[TransientLayerSpecV1] = None
    saturation: Optional[SaturationSpecV1] = None


class InstrumentPatchV1(BaseModel):
    """Frozen Pydantic v2 contract matching ``multimodal_gen.instrument_patch.InstrumentPatch``."""

    model_config = _STRICT_CONFIG

    patch_id: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    track_role: str = Field(..., min_length=1)
    scope: Literal["track"] = "track"
    patch_profile: PatchProfileV1 = Field(default_factory=lambda: PatchProfileV1(family="unknown"))
    synthesis_voice: Optional[SynthesisVoiceV1] = None
    sample_layers: List[SampleLayerSpecV1] = Field(default_factory=list)
    fallback: FallbackBindingV1 = Field(default_factory=FallbackBindingV1)
    preview_subset_notes: List[str] = Field(default_factory=list)
    render_notes: List[str] = Field(default_factory=list)


def validate_instrument_patch(
    data: Union[InstrumentPatchV1, Dict[str, Any]],
) -> InstrumentPatchV1:
    """Validate an instrument patch dict strictly."""
    if isinstance(data, InstrumentPatchV1):
        return data
    return InstrumentPatchV1.model_validate(data)
