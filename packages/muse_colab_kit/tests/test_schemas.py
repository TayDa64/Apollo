"""Standalone schema validation tests for muse_colab_kit.schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from muse_colab_kit.schemas import (
    ArrangementTemplateV1,
    AudioConditioningRef,
    DatasetRecordV1,
    InstrumentPatchV1,
    MidiArtifactV1,
    MidiNoteEventV1,
    NeuralBackendConfig,
    NeuralGenerationRequest,
    NeuralGenerationResponse,
    ScorePlanV1,
    VerifierScoreSnapshotV1,
    export_neural_runtime_json_schema,
    load_arrangement_template_yaml,
    load_bundled_schema,
    validate_arrangement_template,
    validate_dataset_record,
    validate_instrument_patch,
    validate_score_plan,
)


def _valid_plan_dict():
    return {
        "schema_version": "score_plan_v1",
        "prompt": "G-Funk beat at 92 BPM in A minor",
        "bpm": 92,
        "key": "A",
        "mode": "minor",
        "time_signature": [4, 4],
        "genre": "g_funk",
        "sections": [
            {"name": "intro", "type": "intro", "bars": 4, "energy": 0.3, "tension": 0.2},
            {"name": "verse", "type": "verse", "bars": 8, "energy": 0.7, "tension": 0.65},
        ],
        "tracks": [
            {"role": "drums", "instrument": "kick"},
            {"role": "bass", "instrument": "bass"},
            {"role": "lead", "instrument": "synth"},
        ],
    }


def test_bundled_json_schemas_load_cleanly():
    sp = load_bundled_schema("score_plan.v1.schema.json")
    at = load_bundled_schema("arrangement_template.schema.json")
    nr = load_bundled_schema("neural_runtime.v1.schema.json")

    assert sp["$id"] == "muse://schemas/score-plan.v1.json"
    assert at["title"] == "Arrangement Template Schema"
    assert nr == export_neural_runtime_json_schema()


def test_score_plan_v1_accepts_valid_and_rejects_invalid():
    plan = validate_score_plan(_valid_plan_dict())
    assert plan.bpm == 92.0
    assert plan.mode == "minor"
    assert plan.to_schema_dict()["schema_version"] == "score_plan_v1"

    with pytest.raises(ValidationError):
        validate_score_plan({**_valid_plan_dict(), "extra_key": 1})

    with pytest.raises(ValidationError):
        validate_score_plan({**_valid_plan_dict(), "bpm": 20})

    with pytest.raises(ValidationError):
        validate_score_plan({**_valid_plan_dict(), "key": "H#"})


def test_arrangement_template_yaml_and_dict_validation(tmp_path: Path):
    yaml_file = tmp_path / "sample.yaml"
    yaml_file.write_text(
        "template_id: test_trap\n"
        "display_name: Test Trap\n"
        "genre_ref: trap\n"
        "version: '1.0.0'\n"
        "sections:\n"
        "  - type: intro\n"
        "    bars: 4\n"
        "  - type: drop\n"
        "    bars: 8\n",
        encoding="utf-8",
    )
    tpl = load_arrangement_template_yaml(yaml_file)
    assert isinstance(tpl, ArrangementTemplateV1)
    assert tpl.total_bars == 12

    with pytest.raises(ValidationError):
        validate_arrangement_template(
            {
                "template_id": "InvalidCamelCase",
                "display_name": "Bad",
                "genre_ref": "trap",
                "version": "1.0.0",
                "sections": [{"type": "intro", "bars": 4}],
            }
        )


def test_instrument_patch_neural_io_and_dataset_record_roundtrip():
    patch = validate_instrument_patch(
        {
            "patch_id": "core.bass.track.v1",
            "display_name": "Bass",
            "track_role": "bass",
        }
    )
    assert isinstance(patch, InstrumentPatchV1)

    req = NeuralGenerationRequest(
        request_id="r-1",
        prompt="g-funk beat",
        audio_conditioning=AudioConditioningRef(path="audio/clip.wav", duration_seconds=2.5),
    )
    resp = NeuralGenerationResponse(
        request_id=req.request_id,
        success=True,
        attempted=True,
        score_plan=validate_score_plan(_valid_plan_dict()),
        instrument_patches=[patch],
    )
    assert resp.score_plan is not None
    assert NeuralBackendConfig().transport == "disabled"

    record = validate_dataset_record(
        {
            "record_id": "rec_001",
            "split": "train",
            "seed": 42,
            "prompt": "G-Funk beat at 92 BPM in A minor",
            "genre": "g_funk",
            "score_plan": _valid_plan_dict(),
            "instrument_patches": [patch.model_dump()],
            "midi": MidiArtifactV1(
                relative_path="artifacts/midi/rec_001.mid",
                note_count=1,
                events=[MidiNoteEventV1(start_tick=0, duration_ticks=480, pitch=60, velocity=96)],
            ).model_dump(),
            "verifier_scores": VerifierScoreSnapshotV1(
                total_reward=0.9,
                schema_reward=1.0,
                genre_rules_reward=0.9,
                instrument_range_reward=1.0,
                critics_reward=0.8,
                midi_quality_reward=0.85,
            ).model_dump(),
        }
    )
    line = record.to_jsonl_line()
    reloaded = DatasetRecordV1.model_validate_json(line)
    assert reloaded == record
