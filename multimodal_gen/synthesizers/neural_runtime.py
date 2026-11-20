"""Strict Pydantic v2 I/O contract and optional neural runtime adapters.

This module defines the frozen ``neural_io_v1`` contract so that a future
Gemma 4 model (local GGUF via ``llama_cpp`` or a remote OpenAI-compatible HTTP
endpoint) can drop in for ScorePlan / InstrumentPatch generation or opt-in
neural audio rendering without modifying upstream callers in ``worker.py``,
``audio_renderer.py``, or JUCE.

Truthfulness guarantee:
- No production neural model ships inside this repository.
- Every adapter reports capability and dependency status honestly via ``probe()``.
- All neural request/response payloads are validated strictly with Pydantic v2
  (``extra="forbid"``, frozen models).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


NEURAL_CONTRACT_VERSION = "neural_io_v1"

NeuralTransport = Literal["disabled", "callback", "local_torch", "local_gguf", "remote_http"]
NeuralCapability = Literal["score_plan", "instrument_patch", "audio_render"]
ScaleModeLiteral = Literal["major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "locrian"]
SectionTypeLiteral = Literal[
    "intro", "verse", "pre_chorus", "chorus", "drop", "bridge", "breakdown", "outro"
]
CuePointTypeLiteral = Literal["fill", "build", "drop", "breakdown"]
TrackRoleLiteral = Literal["drums", "bass", "keys", "lead", "strings", "fx", "pad"]

_STRICT_CONFIG = ConfigDict(extra="forbid", frozen=True)


def _reject_bool(value: Any, field_name: str) -> Any:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, not bool")
    return value


# =============================================================================
# Frozen ScorePlan v1 Pydantic Contract (1:1 with score_plan.v1.schema.json)
# =============================================================================


class ScorePlanSectionV1(BaseModel):
    model_config = _STRICT_CONFIG

    name: str
    type: SectionTypeLiteral
    bars: int = Field(..., ge=1)
    energy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tension: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("bars", mode="before")
    @classmethod
    def _validate_bars_not_bool(cls, v: Any) -> Any:
        return _reject_bool(v, "bars")

    @field_validator("energy", "tension", mode="before")
    @classmethod
    def _validate_unit_floats(cls, v: Any) -> Any:
        if v is not None:
            _reject_bool(v, "section energy/tension")
        return v


class ChordMapEntryV1(BaseModel):
    model_config = _STRICT_CONFIG

    bar: int = Field(..., ge=1)
    chord: str

    @field_validator("bar", mode="before")
    @classmethod
    def _validate_bar_not_bool(cls, v: Any) -> Any:
        return _reject_bool(v, "bar")


class CuePointV1(BaseModel):
    model_config = _STRICT_CONFIG

    bar: int = Field(..., ge=1)
    type: CuePointTypeLiteral
    intensity: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("bar", mode="before")
    @classmethod
    def _validate_bar_not_bool(cls, v: Any) -> Any:
        return _reject_bool(v, "bar")

    @field_validator("intensity", mode="before")
    @classmethod
    def _validate_intensity_not_bool(cls, v: Any) -> Any:
        if v is not None:
            _reject_bool(v, "intensity")
        return v


class TrackActivationV1(BaseModel):
    model_config = _STRICT_CONFIG

    section: str
    active: bool


class ScorePlanTrackV1(BaseModel):
    model_config = _STRICT_CONFIG

    role: TrackRoleLiteral
    instrument: str
    pattern_hint: Optional[str] = None
    octave: Optional[int] = None
    density: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    activation: Optional[List[TrackActivationV1]] = None

    @field_validator("octave", "density", mode="before")
    @classmethod
    def _validate_numeric_not_bool(cls, v: Any) -> Any:
        if v is not None:
            _reject_bool(v, "track numeric field")
        return v


class ScorePlanConstraintsV1(BaseModel):
    model_config = _STRICT_CONFIG

    avoid_instruments: Optional[List[str]] = None
    avoid_drums: Optional[List[str]] = None
    max_polyphony: Optional[int] = Field(default=None, ge=1)

    @field_validator("max_polyphony", mode="before")
    @classmethod
    def _validate_polyphony_not_bool(cls, v: Any) -> Any:
        if v is not None:
            _reject_bool(v, "max_polyphony")
        return v


class ScorePlanMasteringV1(BaseModel):
    model_config = _STRICT_CONFIG

    production_preset: Optional[str] = None
    target_lufs: Optional[float] = Field(default=None, ge=-30.0, le=-6.0)
    master_ceiling_db: Optional[float] = Field(default=None, ge=-6.0, le=0.0)
    stem_headroom_db: Optional[float] = Field(default=None, ge=-18.0, le=0.0)
    brightness_target: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    warmth_target: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator(
        "target_lufs",
        "master_ceiling_db",
        "stem_headroom_db",
        "brightness_target",
        "warmth_target",
        mode="before",
    )
    @classmethod
    def _validate_mastering_not_bool(cls, v: Any) -> Any:
        if v is not None:
            _reject_bool(v, "mastering numeric field")
        return v


class ScorePlanV1(BaseModel):
    """Frozen Pydantic v2 contract matching ``docs/muse-specs/schemas/score_plan.v1.schema.json``."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["score_plan_v1"] = "score_plan_v1"
    request_id: Optional[str] = None
    prompt: str = Field(..., min_length=1)
    bpm: float = Field(..., ge=30.0, le=220.0)
    key: str = Field(..., pattern=r"^[A-G](#|b)?$")
    mode: ScaleModeLiteral
    time_signature: Optional[Tuple[int, int]] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    influences: Optional[List[str]] = None
    seed: Optional[int] = None
    duration_bars: Optional[int] = Field(default=None, ge=1)
    sections: List[ScorePlanSectionV1] = Field(..., min_length=1)
    chord_map: Optional[List[ChordMapEntryV1]] = None
    tension_curve: Optional[List[float]] = None
    cue_points: Optional[List[CuePointV1]] = None
    tracks: List[ScorePlanTrackV1] = Field(..., min_length=1)
    constraints: Optional[ScorePlanConstraintsV1] = None
    mastering: Optional[ScorePlanMasteringV1] = None

    @field_validator("bpm", "seed", "duration_bars", mode="before")
    @classmethod
    def _validate_scalar_not_bool(cls, v: Any) -> Any:
        if v is not None:
            _reject_bool(v, "score plan numeric field")
        return v

    @field_validator("time_signature", mode="before")
    @classmethod
    def _validate_time_sig(cls, v: Any) -> Any:
        if v is None:
            return None
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError("time_signature must be a 2-element integer sequence [numerator, denominator]")
        for item in v:
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise ValueError("time_signature entries must be integers >= 1")
        return (int(v[0]), int(v[1]))

    @field_validator("tension_curve")
    @classmethod
    def _validate_tension_curve(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return None
        for val in v:
            if isinstance(val, bool) or val < 0.0 or val > 1.0:
                raise ValueError("tension_curve entries must be floats in [0.0, 1.0]")
        return v

    def to_schema_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-schema-compliant dict omitting unset optional fields."""
        return self.model_dump(mode="json", exclude_none=True)


# =============================================================================
# Frozen InstrumentPatch v1 Pydantic Contract
# =============================================================================


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


# =============================================================================
# Frozen Neural Runtime I/O Contract (neural_io_v1)
# =============================================================================


class AudioConditioningRef(BaseModel):
    """Reference to an optional 16 kHz mono audio clip for multimodal models (e.g. Gemma 4)."""

    model_config = _STRICT_CONFIG

    path: Optional[str] = None
    uri: Optional[str] = None
    sha256: Optional[str] = None
    sample_rate_hz: Literal[16000] = 16000
    channels: Literal[1] = 1
    encoding: Literal["pcm_s16le", "wav", "flac"] = "pcm_s16le"
    duration_seconds: Optional[float] = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _require_path_or_uri(self) -> "AudioConditioningRef":
        if not (self.path or self.uri):
            raise ValueError("AudioConditioningRef requires at least one of 'path' or 'uri'")
        return self


class NeuralBackendConfig(BaseModel):
    """Configuration for a pluggable neural runtime backend."""

    model_config = _STRICT_CONFIG

    contract_version: Literal["neural_io_v1"] = NEURAL_CONTRACT_VERSION
    backend_name: str = Field(default="bounded_neural_runtime", min_length=1)
    transport: NeuralTransport = "disabled"
    enabled: bool = False
    model_id: str = ""
    model_path: Optional[str] = None
    endpoint_url: Optional[str] = None
    api_key_env: Optional[str] = None
    capabilities: Tuple[NeuralCapability, ...] = ("score_plan",)
    required_dependencies: Tuple[str, ...] = ()
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=64, le=32768)
    seed: Optional[int] = None


class NeuralGenerationRequest(BaseModel):
    """Strict input contract for neural ScorePlan / InstrumentPatch generation."""

    model_config = _STRICT_CONFIG

    contract_version: Literal["neural_io_v1"] = NEURAL_CONTRACT_VERSION
    request_id: str = Field(default="", max_length=128)
    prompt: str = Field(..., min_length=1)
    genre_hint: Optional[str] = None
    bpm_hint: Optional[float] = Field(default=None, ge=30.0, le=220.0)
    key_hint: Optional[str] = Field(default=None, pattern=r"^[A-G](#|b)?$")
    mode_hint: Optional[ScaleModeLiteral] = None
    duration_bars: Optional[int] = Field(default=None, ge=1)
    seed: Optional[int] = None
    audio_conditioning: Optional[AudioConditioningRef] = None
    session_context: Optional[Dict[str, Any]] = None
    constraints: Optional[ScorePlanConstraintsV1] = None
    requested_outputs: Tuple[Literal["score_plan", "instrument_patches"], ...] = ("score_plan",)

    @field_validator("bpm_hint", "duration_bars", "seed", mode="before")
    @classmethod
    def _validate_hints_not_bool(cls, v: Any) -> Any:
        if v is not None:
            _reject_bool(v, "request hint numeric field")
        return v


class NeuralGenerationResponse(BaseModel):
    """Strict output contract for neural ScorePlan / InstrumentPatch generation."""

    model_config = _STRICT_CONFIG

    contract_version: Literal["neural_io_v1"] = NEURAL_CONTRACT_VERSION
    request_id: str = ""
    success: bool
    attempted: bool = False
    backend_name: str = "bounded_neural_runtime"
    model_id: str = ""
    transport: NeuralTransport = "disabled"
    score_plan: Optional[ScorePlanV1] = None
    instrument_patches: List[InstrumentPatchV1] = Field(default_factory=list)
    skip_reason: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: Optional[float] = Field(default=None, ge=0.0)
    usage: Dict[str, int] = Field(default_factory=dict)
    details: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_success_has_payload(self) -> "NeuralGenerationResponse":
        if self.success and self.score_plan is None and not self.instrument_patches:
            raise ValueError(
                "NeuralGenerationResponse with success=True must include a score_plan or at least one instrument_patch"
            )
        return self


class NeuralAudioRenderRequest(BaseModel):
    """Strict input contract for opt-in MIDI-to-audio neural rendering."""

    model_config = _STRICT_CONFIG

    contract_version: Literal["neural_io_v1"] = NEURAL_CONTRACT_VERSION
    midi_path: str = Field(..., min_length=1)
    output_path: str = Field(..., min_length=1)
    sample_rate: int = Field(default=44100, ge=8000, le=192000)
    genre: Optional[str] = None
    bpm: Optional[float] = Field(default=None, ge=30.0, le=220.0)
    seed: Optional[int] = None


class NeuralAudioRenderResponse(BaseModel):
    """Strict output contract mirroring ``NeuralRenderResult``."""

    model_config = _STRICT_CONFIG

    contract_version: Literal["neural_io_v1"] = NEURAL_CONTRACT_VERSION
    success: bool
    attempted: bool = False
    backend_name: str = "bounded_neural_runtime"
    output_path: Optional[str] = None
    skip_reason: Optional[str] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class NeuralRuntimeContractV1(BaseModel):
    """Root envelope used to export the canonical ``neural_runtime.v1.schema.json`` artifact."""

    model_config = _STRICT_CONFIG

    contract_version: Literal["neural_io_v1"] = NEURAL_CONTRACT_VERSION
    config: NeuralBackendConfig
    generation_request: NeuralGenerationRequest
    generation_response: NeuralGenerationResponse
    audio_render_request: NeuralAudioRenderRequest
    audio_render_response: NeuralAudioRenderResponse


def export_neural_runtime_json_schema() -> Dict[str, Any]:
    """Generate the canonical frozen JSON Schema for ``neural_io_v1``."""
    schema = NeuralRuntimeContractV1.model_json_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "muse://schemas/neural-runtime.v1.json",
        **schema,
    }


# =============================================================================
# Legacy-Compatible Dataclasses & Runtime Protocol
# =============================================================================


@dataclass(frozen=True)
class NeuralBackendStatus:
    """Truthful capability snapshot for an optional neural runtime."""

    backend_name: str
    enabled: bool
    available: bool
    missing_dependencies: Sequence[str] = field(default_factory=tuple)
    model_path: Optional[str] = None
    model_exists: bool = False
    can_render: bool = False
    skip_reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "enabled": bool(self.enabled),
            "available": bool(self.available),
            "missing_dependencies": list(self.missing_dependencies),
            "model_path": self.model_path,
            "model_exists": bool(self.model_exists),
            "can_render": bool(self.can_render),
            "skip_reason": self.skip_reason,
            "details": dict(self.details),
        }


@dataclass
class NeuralRenderResult:
    """Result of a neural render attempt (preserves AudioRenderer report keys)."""

    success: bool
    attempted: bool = False
    backend_name: str = "bounded_neural_runtime"
    output_path: Optional[str] = None
    skip_reason: Optional[str] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "attempted": bool(self.attempted),
            "backend_name": self.backend_name,
            "output_path": self.output_path,
            "skip_reason": self.skip_reason,
            "error_message": self.error_message,
            "details": dict(self.details),
        }

    def to_contract(self) -> NeuralAudioRenderResponse:
        return NeuralAudioRenderResponse(
            success=bool(self.success),
            attempted=bool(self.attempted),
            backend_name=self.backend_name,
            output_path=self.output_path,
            skip_reason=self.skip_reason,
            error_message=self.error_message,
            details=dict(self.details),
        )


@runtime_checkable
class NeuralRuntimeProtocol(Protocol):
    """Protocol implemented by all pluggable neural backends."""

    def probe(self) -> NeuralBackendStatus: ...

    def generate_plan(
        self,
        request: Union[NeuralGenerationRequest, Dict[str, Any]],
    ) -> NeuralGenerationResponse: ...

    def render_midi_to_audio(
        self,
        midi_path: str,
        output_path: str,
        *,
        parsed: Any = None,
        sample_rate: int = 44100,
    ) -> NeuralRenderResult: ...


# =============================================================================
# Shared Structured-Output Helpers
# =============================================================================


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Parse a JSON object from model text output, tolerating a single markdown fence."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Model returned empty output")

    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from neural model, got {type(parsed).__name__}")
    return parsed


def _coerce_generation_request(
    request: Union[NeuralGenerationRequest, Dict[str, Any]],
) -> NeuralGenerationRequest:
    if isinstance(request, NeuralGenerationRequest):
        return request
    return NeuralGenerationRequest.model_validate(request)


def _build_score_plan_system_prompt() -> str:
    return (
        "You are MUSE's symbolic music planner. Return ONLY a single valid JSON object "
        "conforming strictly to schema_version='score_plan_v1' with required keys: "
        "schema_version, prompt, bpm (30..220), key (A-G with optional #/b), mode "
        "(major|minor|dorian|phrygian|lydian|mixolydian|locrian), sections (non-empty list of "
        "{name, type, bars[, energy, tension]}), and tracks (non-empty list of {role, instrument}). "
        "Do not emit unknown keys or explanatory prose."
    )


def _parse_model_payload_to_response(
    payload: Dict[str, Any],
    *,
    request: NeuralGenerationRequest,
    backend_name: str,
    model_id: str,
    transport: NeuralTransport,
    latency_ms: float,
    usage: Optional[Dict[str, int]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> NeuralGenerationResponse:
    """Validate raw model JSON (either bare ScorePlanV1 or wrapped envelope) strictly."""
    if "score_plan" in payload or "instrument_patches" in payload:
        plan_raw = payload.get("score_plan")
        patches_raw = payload.get("instrument_patches") or []
    else:
        plan_raw = payload
        patches_raw = []

    score_plan = ScorePlanV1.model_validate(plan_raw) if plan_raw is not None else None
    instrument_patches = [InstrumentPatchV1.model_validate(p) for p in patches_raw]

    return NeuralGenerationResponse(
        request_id=request.request_id,
        success=True,
        attempted=True,
        backend_name=backend_name,
        model_id=model_id,
        transport=transport,
        score_plan=score_plan,
        instrument_patches=instrument_patches,
        latency_ms=latency_ms,
        usage=dict(usage or {}),
        details=dict(details or {}),
    )


# =============================================================================
# Default Backward-Compatible Runtime + Gemma 4 Drop-In Adapters
# =============================================================================


class OptionalNeuralRuntime:
    """Optional neural backend seam.

    Preserves full backward compatibility with ``AudioRenderer`` and existing
    tests while also supporting strict ``neural_io_v1`` ScorePlan generation
    via an injected ``plan_callback``.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        model_path: Optional[str] = None,
        required_dependencies: Sequence[str] = ("torch",),
        backend_name: str = "bounded_neural_runtime",
        render_callback: Optional[Callable[..., Any]] = None,
        plan_callback: Optional[Callable[[NeuralGenerationRequest], Any]] = None,
        config: Optional[NeuralBackendConfig] = None,
    ) -> None:
        if config is not None:
            self.enabled = bool(config.enabled)
            self.model_path = config.model_path
            self.required_dependencies = tuple(dep for dep in config.required_dependencies if dep)
            self.backend_name = config.backend_name
            self.model_id = config.model_id
            self.transport: NeuralTransport = config.transport
            self.config = config
        else:
            self.enabled = bool(enabled)
            self.model_path = model_path
            self.required_dependencies = tuple(dep for dep in required_dependencies if dep)
            self.backend_name = backend_name
            self.model_id = ""
            self.transport = "callback" if (render_callback or plan_callback) else (
                "local_torch" if enabled else "disabled"
            )
            self.config = NeuralBackendConfig(
                backend_name=self.backend_name,
                transport=self.transport,
                enabled=self.enabled,
                model_path=self.model_path,
                required_dependencies=self.required_dependencies,
                capabilities=("score_plan", "instrument_patch", "audio_render"),
            )
        self._render_callback = render_callback
        self._plan_callback = plan_callback

    def probe(self) -> NeuralBackendStatus:
        missing_dependencies = tuple(
            dependency for dependency in self.required_dependencies if find_spec(dependency) is None
        )
        normalized_model_path = str(Path(self.model_path).expanduser()) if self.model_path else None
        model_exists = bool(normalized_model_path and Path(normalized_model_path).exists())
        available = len(missing_dependencies) == 0

        skip_reason: Optional[str]
        if not self.enabled:
            skip_reason = "disabled"
        elif not normalized_model_path:
            skip_reason = "missing_model_path"
        elif not model_exists:
            skip_reason = "model_path_not_found"
        elif missing_dependencies:
            skip_reason = "missing_dependencies"
        elif self._render_callback is None:
            skip_reason = "no_runtime_impl"
        else:
            skip_reason = None

        can_generate_plan = bool(
            self.enabled and available and self._plan_callback is not None
        )

        return NeuralBackendStatus(
            backend_name=self.backend_name,
            enabled=self.enabled,
            available=available,
            missing_dependencies=missing_dependencies,
            model_path=normalized_model_path,
            model_exists=model_exists,
            can_render=skip_reason is None,
            skip_reason=skip_reason,
            details={
                "contract_version": NEURAL_CONTRACT_VERSION,
                "transport": self.transport,
                "model_id": self.model_id,
                "can_generate_plan": can_generate_plan,
                "required_dependencies": list(self.required_dependencies),
            },
        )

    def generate_plan(
        self,
        request: Union[NeuralGenerationRequest, Dict[str, Any]],
    ) -> NeuralGenerationResponse:
        req = _coerce_generation_request(request)
        status = self.probe()
        if not self.enabled:
            return NeuralGenerationResponse(
                request_id=req.request_id,
                success=False,
                attempted=False,
                backend_name=self.backend_name,
                model_id=self.model_id,
                transport=self.transport,
                skip_reason="disabled",
                details=status.to_dict(),
            )
        if status.missing_dependencies:
            return NeuralGenerationResponse(
                request_id=req.request_id,
                success=False,
                attempted=False,
                backend_name=self.backend_name,
                model_id=self.model_id,
                transport=self.transport,
                skip_reason="missing_dependencies",
                details=status.to_dict(),
            )
        callback = self._plan_callback
        if callback is None:
            return NeuralGenerationResponse(
                request_id=req.request_id,
                success=False,
                attempted=False,
                backend_name=self.backend_name,
                model_id=self.model_id,
                transport=self.transport,
                skip_reason="no_plan_runtime_impl",
                details=status.to_dict(),
            )

        t0 = time.perf_counter()
        try:
            raw = callback(req)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if isinstance(raw, NeuralGenerationResponse):
                return raw
            if isinstance(raw, ScorePlanV1):
                return NeuralGenerationResponse(
                    request_id=req.request_id,
                    success=True,
                    attempted=True,
                    backend_name=self.backend_name,
                    model_id=self.model_id,
                    transport=self.transport,
                    score_plan=raw,
                    latency_ms=latency_ms,
                    details=status.to_dict(),
                )
            if isinstance(raw, str):
                raw = _extract_json_object(raw)
            if isinstance(raw, dict):
                return _parse_model_payload_to_response(
                    raw,
                    request=req,
                    backend_name=self.backend_name,
                    model_id=self.model_id,
                    transport=self.transport,
                    latency_ms=latency_ms,
                    details=status.to_dict(),
                )
            raise TypeError(f"Unsupported plan_callback return type: {type(raw).__name__}")
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return NeuralGenerationResponse(
                request_id=req.request_id,
                success=False,
                attempted=True,
                backend_name=self.backend_name,
                model_id=self.model_id,
                transport=self.transport,
                skip_reason="contract_validation_failed",
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                details=status.to_dict(),
            )
        except Exception as exc:
            return NeuralGenerationResponse(
                request_id=req.request_id,
                success=False,
                attempted=True,
                backend_name=self.backend_name,
                model_id=self.model_id,
                transport=self.transport,
                skip_reason="generation_exception",
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                details=status.to_dict(),
            )

    def render_midi_to_audio(
        self,
        midi_path: str,
        output_path: str,
        *,
        parsed: Any = None,
        sample_rate: int = 44100,
    ) -> NeuralRenderResult:
        status = self.probe()
        callback = self._render_callback
        if not status.can_render or callback is None:
            return NeuralRenderResult(
                success=False,
                attempted=False,
                backend_name=self.backend_name,
                output_path=output_path,
                skip_reason=status.skip_reason or "no_runtime_impl",
                details=status.to_dict(),
            )

        try:
            callback_result = callback(
                midi_path,
                output_path,
                parsed=parsed,
                sample_rate=sample_rate,
            )
        except Exception as exc:
            return NeuralRenderResult(
                success=False,
                attempted=True,
                backend_name=self.backend_name,
                output_path=output_path,
                skip_reason="render_exception",
                error_message=f"{type(exc).__name__}: {exc}",
                details=status.to_dict(),
            )

        if isinstance(callback_result, NeuralRenderResult):
            result = callback_result
        elif isinstance(callback_result, bool):
            result = NeuralRenderResult(
                success=callback_result,
                attempted=True,
                backend_name=self.backend_name,
                output_path=output_path,
                skip_reason=None if callback_result else "render_failed",
                details=status.to_dict(),
            )
        elif isinstance(callback_result, dict):
            result = NeuralRenderResult(
                success=bool(callback_result.get("success")),
                attempted=bool(callback_result.get("attempted", True)),
                backend_name=str(callback_result.get("backend_name", self.backend_name)),
                output_path=callback_result.get("output_path", output_path),
                skip_reason=callback_result.get("skip_reason"),
                error_message=callback_result.get("error_message"),
                details={**status.to_dict(), **dict(callback_result.get("details") or {})},
            )
        else:
            result = NeuralRenderResult(
                success=False,
                attempted=True,
                backend_name=self.backend_name,
                output_path=output_path,
                skip_reason="invalid_result_type",
                error_message=f"Unsupported callback result type: {type(callback_result).__name__}",
                details=status.to_dict(),
            )

        result.details = {**status.to_dict(), **dict(result.details or {})}
        return result


class RemoteOpenAICompatibleNeuralAdapter:
    """Drop-in HTTP adapter for remote Gemma 4 / OpenAI-compatible endpoints.

    Uses only stdlib ``urllib.request`` so it adds zero extra dependencies and
    validates every returned JSON payload strictly against ``ScorePlanV1`` /
    ``InstrumentPatchV1``.
    """

    def __init__(
        self,
        config: NeuralBackendConfig,
        *,
        urlopen_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.config = config
        self._urlopen = urlopen_fn or urllib.request.urlopen

    def probe(self) -> NeuralBackendStatus:
        endpoint = (self.config.endpoint_url or "").strip() or None
        can_generate_plan = bool(self.config.enabled and endpoint)
        if not self.config.enabled:
            skip_reason: Optional[str] = "disabled"
        elif not endpoint:
            skip_reason = "missing_endpoint_url"
        else:
            skip_reason = "audio_render_unsupported"

        return NeuralBackendStatus(
            backend_name=self.config.backend_name,
            enabled=self.config.enabled,
            available=True,
            missing_dependencies=(),
            model_path=endpoint,
            model_exists=bool(endpoint),
            can_render=False,
            skip_reason=skip_reason,
            details={
                "contract_version": NEURAL_CONTRACT_VERSION,
                "transport": "remote_http",
                "model_id": self.config.model_id,
                "endpoint_url": endpoint,
                "can_generate_plan": can_generate_plan,
                "capabilities": list(self.config.capabilities),
            },
        )

    def generate_plan(
        self,
        request: Union[NeuralGenerationRequest, Dict[str, Any]],
    ) -> NeuralGenerationResponse:
        req = _coerce_generation_request(request)
        status = self.probe()
        endpoint = (self.config.endpoint_url or "").strip()
        if not self.config.enabled:
            return NeuralGenerationResponse(
                request_id=req.request_id,
                success=False,
                attempted=False,
                backend_name=self.config.backend_name,
                model_id=self.config.model_id,
                transport="remote_http",
                skip_reason="disabled",
                details=status.to_dict(),
            )
        if not endpoint:
            return NeuralGenerationResponse(
                request_id=req.request_id,
                success=False,
                attempted=False,
                backend_name=self.config.backend_name,
                model_id=self.config.model_id,
                transport="remote_http",
                skip_reason="missing_endpoint_url",
                details=status.to_dict(),
            )

        body: Dict[str, Any] = {
            "model": self.config.model_id or "gemma-4-e4b-it",
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _build_score_plan_system_prompt()},
                {"role": "user", "content": req.model_dump_json(exclude_none=True)},
            ],
        }
        effective_seed = req.seed if req.seed is not None else self.config.seed
        if effective_seed is not None:
            body["seed"] = int(effective_seed)

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key_env:
            token = os.environ.get(self.config.api_key_env, "").strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        http_req = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        t0 = time.perf_counter()
        try:
            with self._urlopen(http_req, timeout=self.config.timeout_seconds) as resp:
                raw_bytes = resp.read()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            envelope = json.loads(raw_bytes.decode("utf-8"))
            choices = envelope.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                raise ValueError("Remote endpoint returned no choices")
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "")
            payload = _extract_json_object(content)

            raw_usage = envelope.get("usage") or {}
            usage = {
                str(k): int(v)
                for k, v in raw_usage.items()
                if isinstance(v, int) and not isinstance(v, bool)
            }
            return _parse_model_payload_to_response(
                payload,
                request=req,
                backend_name=self.config.backend_name,
                model_id=str(envelope.get("model") or self.config.model_id),
                transport="remote_http",
                latency_ms=latency_ms,
                usage=usage,
                details={"endpoint_url": endpoint},
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return NeuralGenerationResponse(
                request_id=req.request_id,
                success=False,
                attempted=True,
                backend_name=self.config.backend_name,
                model_id=self.config.model_id,
                transport="remote_http",
                skip_reason="remote_http_error",
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                details=status.to_dict(),
            )
        except (ValidationError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return NeuralGenerationResponse(
                request_id=req.request_id,
                success=False,
                attempted=True,
                backend_name=self.config.backend_name,
                model_id=self.config.model_id,
                transport="remote_http",
                skip_reason="contract_validation_failed",
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                details=status.to_dict(),
            )

    def render_midi_to_audio(
        self,
        midi_path: str,
        output_path: str,
        *,
        parsed: Any = None,
        sample_rate: int = 44100,
    ) -> NeuralRenderResult:
        status = self.probe()
        return NeuralRenderResult(
            success=False,
            attempted=False,
            backend_name=self.config.backend_name,
            output_path=output_path,
            skip_reason="audio_render_unsupported",
            details=status.to_dict(),
        )


class LocalGGUFNeuralAdapter:
    """Drop-in local GGUF adapter for Gemma 4 weights via optional ``llama_cpp``."""

    def __init__(
        self,
        config: NeuralBackendConfig,
        *,
        llama_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.config = config
        self._llama_factory = llama_factory
        self._llama_instance: Any = None

    def probe(self) -> NeuralBackendStatus:
        required = tuple(self.config.required_dependencies or ("llama_cpp",))
        missing = (
            ()
            if self._llama_factory is not None
            else tuple(dep for dep in required if find_spec(dep) is None)
        )
        normalized_path = (
            str(Path(self.config.model_path).expanduser()) if self.config.model_path else None
        )
        model_exists = bool(normalized_path and Path(normalized_path).exists())
        available = len(missing) == 0

        if not self.config.enabled:
            skip_reason: Optional[str] = "disabled"
        elif not normalized_path:
            skip_reason = "missing_model_path"
        elif not model_exists:
            skip_reason = "model_path_not_found"
        elif missing:
            skip_reason = "missing_dependencies"
        else:
            skip_reason = "audio_render_unsupported"

        can_generate_plan = bool(
            self.config.enabled and normalized_path and model_exists and available
        )

        return NeuralBackendStatus(
            backend_name=self.config.backend_name,
            enabled=self.config.enabled,
            available=available,
            missing_dependencies=missing,
            model_path=normalized_path,
            model_exists=model_exists,
            can_render=False,
            skip_reason=skip_reason,
            details={
                "contract_version": NEURAL_CONTRACT_VERSION,
                "transport": "local_gguf",
                "model_id": self.config.model_id,
                "can_generate_plan": can_generate_plan,
                "required_dependencies": list(required),
            },
        )

    def _get_llama(self) -> Any:
        if self._llama_instance is not None:
            return self._llama_instance
        factory = self._llama_factory
        if factory is None:
            import llama_cpp  # type: ignore[import-not-found]

            factory = llama_cpp.Llama
        self._llama_instance = factory(model_path=str(self.config.model_path))
        return self._llama_instance

    def generate_plan(
        self,
        request: Union[NeuralGenerationRequest, Dict[str, Any]],
    ) -> NeuralGenerationResponse:
        req = _coerce_generation_request(request)
        status = self.probe()
        if not status.details.get("can_generate_plan"):
            return NeuralGenerationResponse(
                request_id=req.request_id,
                success=False,
                attempted=False,
                backend_name=self.config.backend_name,
                model_id=self.config.model_id,
                transport="local_gguf",
                skip_reason=status.skip_reason,
                details=status.to_dict(),
            )

        t0 = time.perf_counter()
        try:
            llm = self._get_llama()
            kwargs: Dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": _build_score_plan_system_prompt()},
                    {"role": "user", "content": req.model_dump_json(exclude_none=True)},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "response_format": {"type": "json_object"},
            }
            effective_seed = req.seed if req.seed is not None else self.config.seed
            if effective_seed is not None:
                kwargs["seed"] = int(effective_seed)

            completion = llm.create_chat_completion(**kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            choices = completion.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                raise ValueError("Local GGUF model returned no choices")
            content = str((choices[0].get("message") or {}).get("content") or "")
            payload = _extract_json_object(content)

            raw_usage = completion.get("usage") or {}
            usage = {
                str(k): int(v)
                for k, v in raw_usage.items()
                if isinstance(v, int) and not isinstance(v, bool)
            }
            return _parse_model_payload_to_response(
                payload,
                request=req,
                backend_name=self.config.backend_name,
                model_id=self.config.model_id or Path(str(self.config.model_path)).name,
                transport="local_gguf",
                latency_ms=latency_ms,
                usage=usage,
                details={"model_path": status.model_path},
            )
        except (ValidationError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return NeuralGenerationResponse(
                request_id=req.request_id,
                success=False,
                attempted=True,
                backend_name=self.config.backend_name,
                model_id=self.config.model_id,
                transport="local_gguf",
                skip_reason="contract_validation_failed",
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                details=status.to_dict(),
            )
        except Exception as exc:
            return NeuralGenerationResponse(
                request_id=req.request_id,
                success=False,
                attempted=True,
                backend_name=self.config.backend_name,
                model_id=self.config.model_id,
                transport="local_gguf",
                skip_reason="gguf_inference_error",
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                details=status.to_dict(),
            )

    def render_midi_to_audio(
        self,
        midi_path: str,
        output_path: str,
        *,
        parsed: Any = None,
        sample_rate: int = 44100,
    ) -> NeuralRenderResult:
        status = self.probe()
        return NeuralRenderResult(
            success=False,
            attempted=False,
            backend_name=self.config.backend_name,
            output_path=output_path,
            skip_reason="audio_render_unsupported",
            details=status.to_dict(),
        )


def create_neural_runtime(config: NeuralBackendConfig) -> NeuralRuntimeProtocol:
    """Factory that instantiates the appropriate neural runtime from a frozen config."""
    if config.transport == "remote_http":
        return RemoteOpenAICompatibleNeuralAdapter(config)
    if config.transport == "local_gguf":
        return LocalGGUFNeuralAdapter(config)
    return OptionalNeuralRuntime(config=config)


__all__ = [
    "NEURAL_CONTRACT_VERSION",
    "NeuralTransport",
    "NeuralCapability",
    "ScaleModeLiteral",
    "SectionTypeLiteral",
    "CuePointTypeLiteral",
    "TrackRoleLiteral",
    "ScorePlanSectionV1",
    "ChordMapEntryV1",
    "CuePointV1",
    "TrackActivationV1",
    "ScorePlanTrackV1",
    "ScorePlanConstraintsV1",
    "ScorePlanMasteringV1",
    "ScorePlanV1",
    "EnvelopeSpecV1",
    "VelocityMapV1",
    "OscillatorSpecV1",
    "FilterSpecV1",
    "LfoSpecV1",
    "NoiseLayerSpecV1",
    "TransientLayerSpecV1",
    "SaturationSpecV1",
    "SampleLayerSpecV1",
    "FallbackBindingV1",
    "PatchProfileV1",
    "SynthesisVoiceV1",
    "InstrumentPatchV1",
    "AudioConditioningRef",
    "NeuralBackendConfig",
    "NeuralGenerationRequest",
    "NeuralGenerationResponse",
    "NeuralAudioRenderRequest",
    "NeuralAudioRenderResponse",
    "NeuralRuntimeContractV1",
    "export_neural_runtime_json_schema",
    "NeuralBackendStatus",
    "NeuralRenderResult",
    "NeuralRuntimeProtocol",
    "OptionalNeuralRuntime",
    "RemoteOpenAICompatibleNeuralAdapter",
    "LocalGGUFNeuralAdapter",
    "create_neural_runtime",
]
