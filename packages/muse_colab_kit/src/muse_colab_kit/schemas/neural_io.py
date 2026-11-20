"""Frozen Pydantic v2 ``neural_io_v1`` contract mirrored for Colab training."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .instrument_patch import InstrumentPatchV1
from .score_plan import ScaleModeLiteral, ScorePlanConstraintsV1, ScorePlanV1, _reject_bool


NEURAL_CONTRACT_VERSION = "neural_io_v1"

NeuralTransport = Literal["disabled", "callback", "local_torch", "local_gguf", "remote_http"]
NeuralCapability = Literal["score_plan", "instrument_patch", "audio_render"]

_STRICT_CONFIG = ConfigDict(extra="forbid", frozen=True)


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
    """Generate the canonical JSON Schema dict for ``neural_io_v1``."""
    schema = NeuralRuntimeContractV1.model_json_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "muse://schemas/neural-runtime.v1.json",
        **schema,
    }
