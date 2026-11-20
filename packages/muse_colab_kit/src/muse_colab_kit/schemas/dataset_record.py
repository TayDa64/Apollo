"""Frozen Pydantic v2 dataset row contract for Colab JSONL shards."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .instrument_patch import InstrumentPatchV1
from .neural_io import AudioConditioningRef
from .score_plan import ScorePlanV1


DATASET_SCHEMA_VERSION = "muse_dataset_v1"

_STRICT_CONFIG = ConfigDict(extra="forbid", frozen=True)


class MidiNoteEventV1(BaseModel):
    model_config = _STRICT_CONFIG

    start_tick: int = Field(..., ge=0)
    duration_ticks: int = Field(..., ge=0)
    pitch: int = Field(..., ge=0, le=127)
    velocity: int = Field(..., ge=0, le=127)
    channel: int = Field(default=0, ge=0, le=15)
    track: str = ""


class MidiArtifactV1(BaseModel):
    model_config = _STRICT_CONFIG

    relative_path: Optional[str] = None
    sha256: Optional[str] = None
    ticks_per_beat: int = Field(default=480, ge=1)
    track_names: List[str] = Field(default_factory=list)
    note_count: int = Field(..., ge=0)
    events: List[MidiNoteEventV1] = Field(default_factory=list)


class VerifierScoreSnapshotV1(BaseModel):
    model_config = _STRICT_CONFIG

    total_reward: float = Field(..., ge=0.0, le=1.0)
    schema_reward: float = Field(..., ge=0.0, le=1.0)
    genre_rules_reward: float = Field(..., ge=0.0, le=1.0)
    instrument_range_reward: float = Field(..., ge=0.0, le=1.0)
    critics_reward: float = Field(..., ge=0.0, le=1.0)
    midi_quality_reward: float = Field(..., ge=0.0, le=1.0)


class DatasetRecordV1(BaseModel):
    """One validated JSONL shard record for Colab SFT / GRPO training."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["muse_dataset_v1"] = DATASET_SCHEMA_VERSION
    record_id: str = Field(..., min_length=1)
    split: Literal["train", "eval"] = "train"
    seed: int
    prompt: str = Field(..., min_length=1)
    genre: str = Field(..., min_length=1)
    audio_clip_16k: Optional[AudioConditioningRef] = None
    score_plan: ScorePlanV1
    instrument_patches: List[InstrumentPatchV1] = Field(default_factory=list)
    midi: MidiArtifactV1
    verifier_scores: VerifierScoreSnapshotV1
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_jsonl_line(self) -> str:
        return self.model_dump_json(exclude_none=True)


def validate_dataset_record(
    data: Union[DatasetRecordV1, Dict[str, Any]],
) -> DatasetRecordV1:
    if isinstance(data, DatasetRecordV1):
        return data
    return DatasetRecordV1.model_validate(data)
