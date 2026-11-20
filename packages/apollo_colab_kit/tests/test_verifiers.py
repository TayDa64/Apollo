"""Standalone pure-Python verifier engine tests."""

from __future__ import annotations

from pathlib import Path

import mido

from apollo_colab_kit.verifiers import (
    GenreRulesEngine,
    QualityValidator,
    clamp_to_range,
    compute_adc,
    compute_bkas,
    compute_vlc,
    extract_critic_features_from_midi,
    extract_note_tuples_from_midi,
    get_range,
    get_timpani_pitches,
    quick_validate,
    run_all_critics,
)


def _write_test_midi(path: Path) -> Path:
    mid = mido.MidiFile(ticks_per_beat=480)

    # Chord + bass track
    mel = mido.MidiTrack()
    mel.append(mido.MetaMessage("track_name", name="Keys", time=0))
    mel.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    # Bar 0: C major triad + bass C3 at tick 0
    for pitch in (48, 60, 64, 67):
        mel.append(mido.Message("note_on", channel=0, note=pitch, velocity=90, time=0))
    for idx, pitch in enumerate((48, 60, 64, 67)):
        mel.append(mido.Message("note_off", channel=0, note=pitch, velocity=0, time=480 if idx == 0 else 0))
    # Bar 1: F major inversion + bass F3
    for idx, pitch in enumerate((53, 60, 65, 69)):
        mel.append(mido.Message("note_on", channel=0, note=pitch, velocity=100, time=1440 if idx == 0 else 0))
    for idx, pitch in enumerate((53, 60, 65, 69)):
        mel.append(mido.Message("note_off", channel=0, note=pitch, velocity=0, time=480 if idx == 0 else 0))
    mid.tracks.append(mel)

    # Drum track with kicks aligned on ticks 0 and 1920
    drums = mido.MidiTrack()
    drums.append(mido.MetaMessage("track_name", name="Drums", time=0))
    drums.append(mido.Message("note_on", channel=9, note=36, velocity=110, time=0))
    drums.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=120))
    drums.append(mido.Message("note_on", channel=9, note=36, velocity=110, time=1800))
    drums.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=120))
    mid.tracks.append(drums)

    mid.save(str(path))
    return path


def test_critics_and_midi_feature_extraction(tmp_path: Path):
    midi_path = _write_test_midi(tmp_path / "sample.mid")
    features = extract_critic_features_from_midi(midi_path)

    assert len(features.voicings) == 2
    assert features.kick_ticks == [0, 1920]
    assert len(features.bass_notes) >= 2

    vlc = compute_vlc(features.voicings)
    bkas = compute_bkas(features.bass_notes, features.kick_ticks)
    adc = compute_adc([0.2, 0.5, 0.8], [1.0, 2.0, 4.0])
    assert vlc.passed is True
    assert bkas.passed is True
    assert adc.passed is True

    report = run_all_critics(
        voicings=features.voicings,
        bass_notes=features.bass_notes,
        kick_ticks=features.kick_ticks,
        tension_curve=[0.2, 0.8],
        note_density_per_beat=[2.0, 6.0],
    )
    assert report.overall_passed is True

    tpb, melodic, drum_tuples = extract_note_tuples_from_midi(midi_path)
    assert tpb == 480
    assert len(melodic) == 8
    assert len(drum_tuples) == 2


def test_instrument_ranges_and_genre_rules_and_quality():
    krar = get_range("krar")
    assert krar.low == 60 and krar.high == 84
    assert 60 <= clamp_to_range(36, "krar") <= 84
    assert all(38 <= p <= 60 for p in get_timpani_pitches("C"))

    engine = GenreRulesEngine()
    bad_gfunk = engine.validate_elements("g_funk", ["synth", "808_distorted"], ["hihat_roll"])
    assert bad_gfunk.valid is False
    repaired, log = engine.repair_violations("g_funk", ["808_distorted", "hihat_roll", "synth"])
    assert "808_distorted" not in repaired
    assert "clean_sine_bass" in repaired
    assert len(log) >= 2

    notes = [
        (i * 240, 200, 60 + (i % 7), 75 + (i % 4) * 10)
        for i in range(24)
    ]
    qv = QualityValidator()
    report = qv.validate(notes, genre="pop", key="C", scale="major", tempo=120)
    assert 0.0 <= report.overall_score <= 1.0
    passed, score, _issues = quick_validate(notes, genre="pop")
    assert isinstance(passed, bool)
    assert 0.0 <= score <= 1.0
