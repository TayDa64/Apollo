"""Frozen Pydantic v2 validator for MUSE ``score_plan.v1.schema.json``."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


def validate_score_plan(data: Union[ScorePlanV1, Dict[str, Any]]) -> ScorePlanV1:
    """Validate a dict or ``ScorePlanV1`` strictly against the frozen contract."""
    if isinstance(data, ScorePlanV1):
        return data
    return ScorePlanV1.model_validate(data)
