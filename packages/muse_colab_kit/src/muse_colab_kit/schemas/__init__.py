"""Frozen Pydantic v2 schema validators and bundled JSON schemas for MUSE."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, Dict, Literal

from .arrangement_template import (
    ArrangementSectionSpecV1,
    ArrangementSectionType,
    ArrangementTemplateV1,
    MotifAssignmentSpecV1,
    MotifTransformation,
    SectionConfigSpecV1,
    TransformParamsV1,
    load_arrangement_template_yaml,
    validate_arrangement_template,
)
from .dataset_record import (
    DATASET_SCHEMA_VERSION,
    DatasetRecordV1,
    MidiArtifactV1,
    MidiNoteEventV1,
    VerifierScoreSnapshotV1,
    validate_dataset_record,
)
from .instrument_patch import (
    EnvelopeSpecV1,
    FallbackBindingV1,
    FilterSpecV1,
    InstrumentPatchV1,
    LfoSpecV1,
    NoiseLayerSpecV1,
    OscillatorSpecV1,
    PatchProfileV1,
    SampleLayerSpecV1,
    SaturationSpecV1,
    SynthesisVoiceV1,
    TransientLayerSpecV1,
    VelocityMapV1,
    validate_instrument_patch,
)
from .neural_io import (
    NEURAL_CONTRACT_VERSION,
    AudioConditioningRef,
    NeuralAudioRenderRequest,
    NeuralAudioRenderResponse,
    NeuralBackendConfig,
    NeuralCapability,
    NeuralGenerationRequest,
    NeuralGenerationResponse,
    NeuralRuntimeContractV1,
    NeuralTransport,
    export_neural_runtime_json_schema,
)
from .score_plan import (
    ChordMapEntryV1,
    CuePointTypeLiteral,
    CuePointV1,
    ScaleModeLiteral,
    ScorePlanConstraintsV1,
    ScorePlanMasteringV1,
    ScorePlanSectionV1,
    ScorePlanTrackV1,
    ScorePlanV1,
    SectionTypeLiteral,
    TrackActivationV1,
    TrackRoleLiteral,
    validate_score_plan,
)

BundledSchemaName = Literal[
    "score_plan.v1.schema.json",
    "arrangement_template.schema.json",
    "neural_runtime.v1.schema.json",
]


def load_bundled_schema(name: BundledSchemaName) -> Dict[str, Any]:
    """Load one of the frozen JSON Schema files bundled in ``muse_colab_kit.schemas.data``."""
    data_file = resources.files("muse_colab_kit.schemas").joinpath("data", name)
    return json.loads(data_file.read_text(encoding="utf-8"))


__all__ = [
    "BundledSchemaName",
    "load_bundled_schema",
    # ScorePlan
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
    "validate_score_plan",
    # ArrangementTemplate
    "ArrangementSectionType",
    "MotifTransformation",
    "TransformParamsV1",
    "MotifAssignmentSpecV1",
    "SectionConfigSpecV1",
    "ArrangementSectionSpecV1",
    "ArrangementTemplateV1",
    "validate_arrangement_template",
    "load_arrangement_template_yaml",
    # InstrumentPatch
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
    "validate_instrument_patch",
    # Neural I/O
    "NEURAL_CONTRACT_VERSION",
    "NeuralTransport",
    "NeuralCapability",
    "AudioConditioningRef",
    "NeuralBackendConfig",
    "NeuralGenerationRequest",
    "NeuralGenerationResponse",
    "NeuralAudioRenderRequest",
    "NeuralAudioRenderResponse",
    "NeuralRuntimeContractV1",
    "export_neural_runtime_json_schema",
    # Dataset record
    "DATASET_SCHEMA_VERSION",
    "MidiNoteEventV1",
    "MidiArtifactV1",
    "VerifierScoreSnapshotV1",
    "DatasetRecordV1",
    "validate_dataset_record",
]
