"""Frozen Pydantic v2 validator for MUSE ``arrangement_template.schema.json``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


ArrangementSectionType = Literal[
    "intro",
    "verse",
    "pre_chorus",
    "chorus",
    "drop",
    "breakdown",
    "buildup",
    "bridge",
    "outro",
    "variation",
]

MotifTransformation = Literal[
    "original",
    "invert",
    "retrograde",
    "augment",
    "diminish",
    "sequence",
]

_STRICT_CONFIG = ConfigDict(extra="forbid", frozen=True)


class TransformParamsV1(BaseModel):
    model_config = _STRICT_CONFIG

    pivot: Optional[int] = None
    factor: Optional[float] = None
    steps: Optional[int] = None


class MotifAssignmentSpecV1(BaseModel):
    model_config = _STRICT_CONFIG

    motif_index: int = Field(..., ge=0)
    transformation: MotifTransformation
    transform_params: Optional[TransformParamsV1] = None


class SectionConfigSpecV1(BaseModel):
    model_config = _STRICT_CONFIG

    typical_bars: int = Field(default=8, ge=1, le=64)
    energy_level: float = Field(default=0.5, ge=0.0, le=1.0)
    drum_density: float = Field(default=0.5, ge=0.0, le=1.0)
    instrument_density: float = Field(default=0.5, ge=0.0, le=1.0)
    texture_amount: float = Field(default=0.5, ge=0.0, le=1.0)
    filter_cutoff: float = Field(default=1.0, ge=0.0, le=1.0)
    enable_kick: bool = True
    enable_snare: bool = True
    enable_hihat: bool = True
    enable_bass: bool = True
    enable_chords: bool = True
    enable_melody: bool = False
    enable_textures: bool = False


class ArrangementSectionSpecV1(BaseModel):
    model_config = _STRICT_CONFIG

    type: ArrangementSectionType
    bars: int = Field(..., ge=1, le=64)
    label: Optional[str] = None

    @field_validator("bars", mode="before")
    @classmethod
    def _reject_bool_bars(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("bars must be an integer, not bool")
        return v


class ArrangementTemplateV1(BaseModel):
    """Frozen Pydantic v2 validator matching ``arrangement_template.schema.json``."""

    model_config = _STRICT_CONFIG

    template_id: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str = Field(..., min_length=1)
    genre_ref: str = Field(..., min_length=1)
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    sections: List[ArrangementSectionSpecV1] = Field(..., min_length=1)

    @property
    def total_bars(self) -> int:
        return sum(section.bars for section in self.sections)


def validate_arrangement_template(
    data: Union[ArrangementTemplateV1, Dict[str, Any]],
) -> ArrangementTemplateV1:
    """Validate an arrangement template dict strictly."""
    if isinstance(data, ArrangementTemplateV1):
        return data
    return ArrangementTemplateV1.model_validate(data)


def load_arrangement_template_yaml(path: Union[str, Path]) -> ArrangementTemplateV1:
    """Load and validate an arrangement template YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(raw).__name__}")
    # PyYAML safe_load parses unquoted numeric tags like `- 808` as ints; normalize to str.
    if isinstance(raw.get("tags"), list):
        raw = dict(raw)
        raw["tags"] = [
            str(tag) if (isinstance(tag, int) and not isinstance(tag, bool)) else tag
            for tag in raw["tags"]
        ]
    return ArrangementTemplateV1.model_validate(raw)
