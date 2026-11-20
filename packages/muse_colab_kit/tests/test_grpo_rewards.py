"""Deterministic GRPO reward tests for muse_colab_kit.verifiers.rewards."""

from __future__ import annotations

import json
from pathlib import Path

import mido

from muse_colab_kit.verifiers import (
    RewardBreakdown,
    composite_grpo_reward,
    critics_reward,
    genre_rules_reward,
    grpo_batch_reward_func,
    instrument_range_reward,
    midi_quality_reward,
    schema_reward,
)


def _good_gfunk_plan():
    return {
        "schema_version": "score_plan_v1",
        "prompt": "G-Funk cruising beat at 92 BPM in A minor",
        "bpm": 92,
        "key": "A",
        "mode": "minor",
        "genre": "g_funk",
        "sections": [
            {"name": "intro", "type": "intro", "bars": 4, "energy": 0.3, "tension": 0.2},
            {"name": "verse", "type": "verse", "bars": 8, "energy": 0.75, "tension": 0.7},
        ],
        "tension_curve": [0.2, 0.4, 0.7, 0.85],
        "tracks": [
            {"role": "drums", "instrument": "hihat"},
            {"role": "bass", "instrument": "bass"},
            {"role": "lead", "instrument": "synth_lead"},
        ],
    }


def test_schema_reward_shaping_and_strictness():
    assert schema_reward("not json at all") == 0.0
    partial = schema_reward(json.dumps({"schema_version": "score_plan_v1", "prompt": "x"}))
    assert 0.15 < partial < 0.5
    assert schema_reward(json.dumps(_good_gfunk_plan())) == 1.0
    fenced = f"```json\n{json.dumps(_good_gfunk_plan())}\n```"
    assert schema_reward(fenced) == 1.0


def test_genre_rules_reward_distinguishes_clean_vs_forbidden():
    clean = genre_rules_reward(_good_gfunk_plan())
    bad_plan = _good_gfunk_plan()
    bad_plan["tracks"] = [
        {"role": "drums", "instrument": "hihat_roll"},
        {"role": "bass", "instrument": "808_distorted"},
    ]
    violated = genre_rules_reward(bad_plan)
    assert clean > violated
    assert 0.0 <= violated <= 1.0
    assert clean == 1.0


def test_instrument_range_critics_and_quality_rewards_are_bounded_and_monotonic():
    good_krar = instrument_range_reward([64, 67, 69, 72], instrument="krar")
    bad_krar = instrument_range_reward([24, 28, 30, 115], instrument="krar")
    assert good_krar == 1.0
    assert bad_krar == 0.0

    c_good = critics_reward(
        voicings=[[60, 64, 67], [60, 65, 69]],
        bass_notes=[{"tick": 0}, {"tick": 480}],
        kick_ticks=[0, 480],
        tension_curve=[0.2, 0.5, 0.8],
        note_density_per_beat=[1.0, 2.5, 4.0],
    )
    c_bad = critics_reward(
        voicings=[[48, 52, 55], [72, 76, 79]],
        bass_notes=[{"tick": 0}],
        kick_ticks=[240, 720, 1200],
        tension_curve=[0.2, 0.5, 0.8],
        note_density_per_beat=[4.0, 2.5, 1.0],
    )
    assert 0.0 <= c_bad < c_good <= 1.0

    notes = [(i * 240, 200, 60 + (i % 7), 70 + (i % 4) * 12) for i in range(20)]
    q = midi_quality_reward(notes, genre="pop", key="C", scale="major", tempo=120)
    assert 0.0 < q <= 1.0
    assert midi_quality_reward([]) == 0.0


def test_composite_grpo_reward_deterministic_and_batch_api(tmp_path: Path):
    plan_json = json.dumps(_good_gfunk_plan())
    b1 = composite_grpo_reward(plan_json)
    b2 = composite_grpo_reward(plan_json)
    assert isinstance(b1, RewardBreakdown)
    assert b1 == b2
    assert b1.total_reward == 1.0

    # Batch API compatible with TRL/Unsloth conversational completions
    rewards = grpo_batch_reward_func(
        [
            [{"role": "assistant", "content": plan_json}],
            [{"role": "assistant", "content": "garbage"}],
        ],
        genres=["g_funk", "g_funk"],
    )
    assert len(rewards) == 2
    assert rewards[0] == 1.0
    assert rewards[1] == 0.0

    # With a MIDI file attached
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    for i in range(16):
        pitch = 60 + (i % 5)
        tr.append(mido.Message("note_on", note=pitch, velocity=85, time=0 if i == 0 else 240))
        tr.append(mido.Message("note_off", note=pitch, velocity=0, time=200))
    mid.tracks.append(tr)
    midi_path = tmp_path / "rollout.mid"
    mid.save(str(midi_path))

    b_midi = composite_grpo_reward(plan_json, midi_path=midi_path)
    assert 0.0 <= b_midi.total_reward <= 1.0
    assert b_midi.diagnostics["has_midi_evidence"] is True
