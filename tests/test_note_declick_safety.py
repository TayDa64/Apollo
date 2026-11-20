"""Boundary-continuity regressions for the offline procedural renderer.

These tests lock the Phase 0/1 pop-fix contract: melodic notes are rendered
through a shared edge-safety wrapper before track assembly, generic samples are
safe when padded/truncated, hybrid attack remains opt-in, and same-pitch MIDI
re-triggers do not overwrite active note state.
"""

import numpy as np
import mido

from multimodal_gen.assets_gen import (
    SAMPLE_RATE,
    generate_bass_tone,
    generate_brass_tone,
    generate_guitar_tone,
    generate_harp_tone,
    generate_piano_tone,
    generate_strings_tone,
)
from multimodal_gen.audio_renderer import AudioRenderer, ProceduralRenderer, SynthNote


def _edge_ratio(audio: np.ndarray, samples: int = 132) -> float:
    a = np.asarray(audio, dtype=np.float64)
    if a.size == 0:
        return 0.0
    peak = float(np.max(np.abs(a))) or 1.0
    tail = a[-min(samples, a.size):]
    return float(np.mean(np.abs(tail)) / peak)


def _boundary_jump(audio: np.ndarray, sample_index: int) -> float:
    a = np.asarray(audio, dtype=np.float64)
    if sample_index <= 0 or sample_index >= a.size:
        return 0.0
    return float(abs(a[sample_index] - a[sample_index - 1]))


def _max_boundary_jump(audio: np.ndarray, starts: list[int]) -> float:
    return max((_boundary_jump(audio, start) for start in starts if start > 0), default=0.0)


_ENGINE_CASES = [
    ("guitar", generate_guitar_tone, dict(frequency=196.0, velocity=0.8, drive=0.45)),
    ("bass", generate_bass_tone, dict(frequency=55.0, velocity=0.8)),
    ("piano", generate_piano_tone, dict(frequency=261.63, velocity=0.8)),
    ("brass", generate_brass_tone, dict(frequency=220.0, velocity=0.8)),
    ("strings", generate_strings_tone, dict(frequency=220.0, velocity=0.8)),
    ("harp", generate_harp_tone, dict(frequency=220.0, velocity=0.8)),
]


def test_shared_note_edge_safety_makes_generator_outputs_click_safe():
    renderer = ProceduralRenderer(sample_rate=SAMPLE_RATE, genre="cinematic")
    for family, fn, base_kw in _ENGINE_CASES:
        for duration in (0.02, 0.06, 0.4):
            kw = dict(base_kw)
            kw["duration"] = duration
            raw = fn(**kw)
            safe = renderer._apply_note_edge_safety(raw)
            assert safe.dtype == np.float32, family
            assert safe.shape[0] == raw.shape[0], family
            assert np.all(np.isfinite(safe)), family
            assert np.max(np.abs(safe)) <= 1.0 + 1e-6, family
            assert abs(float(safe[0])) < 1e-6, family
            assert _edge_ratio(safe) < 0.05, (family, duration, _edge_ratio(safe))


def _render_repeated(program: int, *, starts: list[int], duration: int, pitch: int = 60) -> tuple[np.ndarray, list[int]]:
    renderer = ProceduralRenderer(sample_rate=SAMPLE_RATE, genre="cinematic")
    notes = [
        SynthNote(
            pitch=pitch,
            start_sample=start,
            duration_samples=duration,
            velocity=0.8,
            channel=0,
            program=program,
        )
        for start in starts
    ]
    total = max(starts) + duration + 512
    return renderer.render_notes(notes, total, is_drums=False), starts


def test_render_notes_abutting_gapped_and_overlapped_boundaries_are_continuous():
    duration = int(0.12 * SAMPLE_RATE)
    cases = [
        [0, duration, duration * 2],  # abutting
        [0, duration + int(0.005 * SAMPLE_RATE), 2 * duration + int(0.010 * SAMPLE_RATE)],  # small gaps
        [0, duration - int(0.020 * SAMPLE_RATE), 2 * duration - int(0.040 * SAMPLE_RATE)],  # overlaps
    ]
    for starts in cases:
        audio, note_starts = _render_repeated(0, starts=starts, duration=duration, pitch=60)
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) <= 1.05
        assert _max_boundary_jump(audio, note_starts) < 0.02


def test_render_notes_same_pitch_retrigger_and_chord_starts_are_continuous():
    duration = int(0.10 * SAMPLE_RATE)
    retrigger_starts = [0, int(0.075 * SAMPLE_RATE), int(0.150 * SAMPLE_RATE)]
    retrigger, starts = _render_repeated(48, starts=retrigger_starts, duration=duration, pitch=60)
    assert _max_boundary_jump(retrigger, starts) < 0.025

    renderer = ProceduralRenderer(sample_rate=SAMPLE_RATE, genre="cinematic")
    chord_start = int(0.14 * SAMPLE_RATE)
    notes = [
        SynthNote(pitch=pitch, start_sample=0, duration_samples=duration, velocity=0.75, channel=0, program=48)
        for pitch in (60, 64, 67, 72)
    ] + [
        SynthNote(pitch=pitch, start_sample=chord_start, duration_samples=duration, velocity=0.75, channel=0, program=48)
        for pitch in (62, 65, 69, 74)
    ]
    chord = renderer.render_notes(notes, chord_start + duration + 512, is_drums=False)
    assert np.all(np.isfinite(chord))
    assert _boundary_jump(chord, chord_start) < 0.035


def test_hybrid_attack_default_off_and_enabled_edges_are_safe():
    renderer = ProceduralRenderer(sample_rate=SAMPLE_RATE, genre="funk")
    proc = generate_guitar_tone(196.0, 0.20, 0.85, SAMPLE_RATE, drive=0.72, voice=renderer._resolve_patch_voice("guitar"))

    off = renderer._apply_attack_layer("guitar", proc, 0.85)
    assert np.array_equal(off, proc)

    renderer._hybrid_attack_on = True
    hybrid = renderer._apply_attack_layer("guitar", proc, 0.85)
    assert hybrid.dtype == np.float32
    assert np.all(np.isfinite(hybrid))
    assert np.max(np.abs(hybrid)) <= 1.0 + 1e-6
    assert abs(float(hybrid[0])) < 1e-6
    assert _edge_ratio(hybrid) < 0.05


def test_custom_sample_pad_and_truncate_paths_are_click_safe():
    renderer = ProceduralRenderer(sample_rate=SAMPLE_RATE, genre="cinematic")
    nonzero_sample = np.ones(1000, dtype=np.float32) * 0.5
    variants = [{"audio": nonzero_sample, "root_note": 60, "sample_rate": SAMPLE_RATE}]

    padded = renderer._pitch_shift_sample(variants, target_pitch=60, duration=0.08, velocity=1.0)
    assert abs(float(padded[0])) < 1e-6
    assert _edge_ratio(padded) < 0.05
    assert _boundary_jump(padded, len(nonzero_sample)) < 0.02

    long_sample = np.ones(6000, dtype=np.float32) * 0.5
    truncated = renderer._pitch_shift_sample(
        [{"audio": long_sample, "root_note": 60, "sample_rate": SAMPLE_RATE}],
        target_pitch=60,
        duration=0.04,
        velocity=1.0,
    )
    assert abs(float(truncated[0])) < 1e-6
    assert _edge_ratio(truncated) < 0.05


def test_expansion_sample_pad_and_truncate_paths_are_click_safe():
    renderer = ProceduralRenderer(sample_rate=SAMPLE_RATE, genre="cinematic")
    renderer._expansion_sample_cache = {"strings": np.ones(1000, dtype=np.float32) * 0.5}

    padded = renderer._get_expansion_sample_for_program(48, 220.0, 0.08, 1.0)
    assert padded is not None
    assert abs(float(padded[0])) < 1e-6
    assert _edge_ratio(padded) < 0.05
    assert _boundary_jump(padded, 1000) < 0.02

    renderer._expansion_sample_cache = {"strings": np.ones(6000, dtype=np.float32) * 0.5}
    truncated = renderer._get_expansion_sample_for_program(48, 220.0, 0.04, 1.0)
    assert truncated is not None
    assert abs(float(truncated[0])) < 1e-6
    assert _edge_ratio(truncated) < 0.05


def test_render_track_procedural_preserves_overlapping_same_pitch_notes(monkeypatch):
    midi_track = mido.MidiTrack()
    midi_track.name = "Overlap"
    midi_track.append(mido.Message("program_change", program=48, channel=0, time=0))
    midi_track.append(mido.Message("note_on", note=60, velocity=64, channel=0, time=0))
    midi_track.append(mido.Message("note_on", note=60, velocity=80, channel=0, time=10))
    midi_track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=10))
    midi_track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=10))

    renderer = AudioRenderer(sample_rate=SAMPLE_RATE, use_fluidsynth=False, genre="cinematic")
    captured = {}

    def fake_render_notes(notes, total_samples, is_drums=False):
        captured["notes"] = list(notes)
        return np.zeros(total_samples, dtype=np.float32)

    monkeypatch.setattr(renderer.procedural, "render_notes", fake_render_notes)
    out = renderer._render_track_procedural(midi_track, ticks_per_beat=480, total_samples=4096)

    assert out is not None
    notes = captured["notes"]
    assert len(notes) == 2
    assert [note.velocity for note in notes] == [64 / 127.0, 80 / 127.0]
    assert all(note.program == 48 for note in notes)
    assert notes[0].start_sample < notes[1].start_sample
