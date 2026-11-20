import pytest

from multimodal_gen.score_plan_adapter import (
    ScorePlanError,
    extract_mastering_overrides,
    score_plan_to_parsed_prompt,
    score_plan_to_performance_score,
    validate_score_plan,
)
from multimodal_gen.utils import ScaleType, bars_to_ticks


def _sample_plan():
    return {
        "schema_version": "score_plan_v1",
        "prompt": "ambient cinematic soundscape in C minor with atmospheric pads",
        "bpm": 72,
        "key": "C",
        "mode": "minor",
        "time_signature": [4, 4],
        "sections": [
            {"name": "intro", "type": "intro", "bars": 4},
            {"name": "main", "type": "verse", "bars": 8},
        ],
        "tracks": [
            {"role": "pad", "instrument": "pad"},
            {"role": "strings", "instrument": "strings"},
            {"role": "drums", "instrument": "kick"},
        ],
        "chord_map": [{"bar": 1, "chord": "Cm"}],
        "cue_points": [{"bar": 5, "type": "build", "intensity": 0.6}],
        "constraints": {"avoid_instruments": ["brass"], "avoid_drums": ["snare"]},
        "mastering": {
            "production_preset": "ambient_ethereal",
            "target_lufs": -16.0,
            "master_ceiling_db": -1.5,
            "brightness_target": 0.6,
            "warmth_target": 0.7,
        },
    }


def test_score_plan_overrides_parsed_prompt():
    plan = _sample_plan()
    parsed = score_plan_to_parsed_prompt(plan)
    assert parsed.bpm == 72
    assert parsed.key == "C"
    assert parsed.scale_type.name.lower() == "minor"
    assert "pad" in parsed.instruments
    assert "strings" in parsed.instruments
    assert "kick" in parsed.drum_elements
    assert "brass" in parsed.excluded_instruments
    assert "snare" in parsed.excluded_drums
    assert parsed.production_preset == "ambient_ethereal"
    assert parsed.brightness == 0.6
    assert parsed.warmth == 0.7


def test_score_plan_reapplies_avoided_drums_after_track_adaptation():
    plan = _sample_plan()
    plan["tracks"] = [{"role": "drums", "instrument": "snare"}]
    plan["constraints"] = {"avoid_drums": ["snare"]}

    parsed = score_plan_to_parsed_prompt(plan)

    assert "snare" in parsed.excluded_drums
    assert "snare" not in parsed.drum_elements


def test_score_plan_reapplies_avoided_instruments_after_track_adaptation():
    plan = _sample_plan()
    plan["tracks"] = [{"role": "strings", "instrument": "brass"}]
    plan["constraints"] = {"avoid_instruments": ["brass"]}

    parsed = score_plan_to_parsed_prompt(plan)

    assert "brass" in parsed.excluded_instruments
    assert "brass" not in parsed.instruments


def test_score_plan_builds_performance_score():
    plan = _sample_plan()
    score = score_plan_to_performance_score(plan)
    assert score.sections
    assert score.tempo_map.get(0) == 72
    assert score.key_map.get(0) == "C"
    assert score.chord_map.get(0) == "Cm"

    build_tick = bars_to_ticks(4, (4, 4))
    cues = [c for c in score.cue_points if c.get("type") == "build"]
    assert cues and cues[0]["tick"] == build_tick


def test_extract_mastering_overrides_returns_structured_block():
    mastering = extract_mastering_overrides(_sample_plan())
    assert mastering["production_preset"] == "ambient_ethereal"
    assert mastering["target_lufs"] == -16.0
    assert mastering["master_ceiling_db"] == -1.5


@pytest.mark.parametrize(
    "mode_str,expected",
    [
        ("major", ScaleType.MAJOR),
        ("minor", ScaleType.MINOR),
        ("dorian", ScaleType.DORIAN),
        ("phrygian", ScaleType.PHRYGIAN),
        ("lydian", ScaleType.LYDIAN),
        ("mixolydian", ScaleType.MIXOLYDIAN),
        ("locrian", ScaleType.LOCRIAN),
    ],
)
def test_score_plan_preserves_all_seven_modes(mode_str, expected):
    plan = _sample_plan()
    plan["mode"] = mode_str
    parsed = score_plan_to_parsed_prompt(plan)
    assert parsed.scale_type == expected


def test_validate_score_plan_rejects_extra_fields_and_invalid_bpm():
    bad_extra = _sample_plan()
    bad_extra["unexpected_key"] = 123
    with pytest.raises(ScorePlanError):
        validate_score_plan(bad_extra)

    bad_bpm = _sample_plan()
    bad_bpm["bpm"] = 999
    with pytest.raises(ScorePlanError):
        validate_score_plan(bad_bpm)


def test_score_plan_applies_section_energy_and_tension():
    plan = _sample_plan()
    plan["sections"] = [
        {"name": "intro", "type": "intro", "bars": 4, "energy": 0.22, "tension": 0.15},
        {"name": "drop", "type": "drop", "bars": 8, "energy": 0.94, "tension": 0.88},
    ]
    score = score_plan_to_performance_score(plan)
    assert len(score.sections) == 2
    assert score.sections[0].config.energy_level == pytest.approx(0.22)
    assert score.sections[1].config.energy_level == pytest.approx(0.94)
    assert score.tension_curve == [0.15, 0.88]

