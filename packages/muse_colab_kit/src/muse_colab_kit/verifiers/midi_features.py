"""Pure-Python + mido MIDI critic feature extraction.

Mirrors ``multimodal_gen.intelligence.midi_critic_features`` with zero heavy dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class CriticFeatures:
    voicings: List[List[int]]
    bass_notes: List[Dict[str, int]]
    kick_ticks: List[int]
    tension_curve: List[float]
    note_density_per_beat: List[float]
    ticks_per_beat: int


def _iter_note_events(
    midi_source: Union[str, Path],
) -> Tuple[int, List[Tuple[int, int, int, int, int, str]]]:
    """Return ``(ticks_per_beat, notes)`` where notes are
    ``(start_tick, duration_ticks, pitch, velocity, channel, track_name)``.
    """
    import mido

    midi = mido.MidiFile(str(midi_source))
    ticks_per_beat = int(getattr(midi, "ticks_per_beat", 480) or 480)

    notes: List[Tuple[int, int, int, int, int, str]] = []

    for track in midi.tracks:
        abs_tick = 0
        track_name = getattr(track, "name", "") or ""
        active: Dict[Tuple[int, int], Tuple[int, int]] = {}

        for msg in track:
            abs_tick += int(getattr(msg, "time", 0) or 0)

            if not hasattr(msg, "type"):
                continue

            if msg.type == "note_on" and int(getattr(msg, "velocity", 0) or 0) > 0:
                ch = int(getattr(msg, "channel", 0) or 0)
                pitch = int(getattr(msg, "note", 0) or 0)
                vel = int(getattr(msg, "velocity", 0) or 0)
                active[(ch, pitch)] = (abs_tick, vel)
            elif msg.type in ("note_off", "note_on"):
                if msg.type == "note_on" and int(getattr(msg, "velocity", 0) or 0) != 0:
                    continue
                ch = int(getattr(msg, "channel", 0) or 0)
                pitch = int(getattr(msg, "note", 0) or 0)
                key = (ch, pitch)
                if key in active:
                    start_tick, vel = active.pop(key)
                    duration = max(0, abs_tick - start_tick)
                    notes.append((start_tick, duration, pitch, vel, ch, track_name))

        for (ch, pitch), (start_tick, vel) in list(active.items()):
            duration = max(0, abs_tick - start_tick)
            notes.append((start_tick, duration, int(pitch), int(vel), int(ch), track_name))

    return ticks_per_beat, notes


def extract_note_events_from_midi(
    midi_path: Union[str, Path],
) -> Tuple[int, List[Tuple[int, int, int, int, int, str]]]:
    """Public helper returning ``(ticks_per_beat, note_events)`` from a MIDI file."""
    return _iter_note_events(midi_path)


def extract_note_tuples_from_midi(
    midi_path: Union[str, Path],
) -> Tuple[int, List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """Return ``(ticks_per_beat, melodic_notes, drum_notes)`` as 4-tuples for ``QualityValidator``."""
    ticks_per_beat, events = _iter_note_events(midi_path)
    melodic: List[Tuple[int, int, int, int]] = []
    drums: List[Tuple[int, int, int, int]] = []
    for start_tick, duration, pitch, velocity, channel, _track in events:
        tup = (int(start_tick), int(duration), int(pitch), int(velocity))
        if channel == 9:
            drums.append(tup)
        else:
            melodic.append(tup)
    return ticks_per_beat, melodic, drums


def extract_critic_features_from_midi(
    midi_path: Union[str, Path],
    *,
    time_signature: Tuple[int, int] = (4, 4),
    tension_curve: Optional[List[float]] = None,
) -> CriticFeatures:
    """Extract deterministic critic inputs from a MIDI file."""
    ticks_per_beat, notes = _iter_note_events(midi_path)

    kick_ticks = sorted(
        int(t)
        for (t, _dur, pitch, vel, ch, _name) in notes
        if ch == 9 and pitch == 36 and vel > 0
    )

    bass_notes = [
        {"tick": int(t), "pitch": int(pitch)}
        for (t, _dur, pitch, _vel, ch, _name) in notes
        if ch != 9 and int(pitch) < 60
    ]

    tick_groups: Dict[int, List[int]] = defaultdict(list)
    for (t, _dur, pitch, vel, ch, _name) in notes:
        if ch == 9 or vel <= 0:
            continue
        p = int(pitch)
        if 48 <= p <= 96:
            tick_groups[int(t)].append(p)

    voicings = [
        sorted(v)
        for t, v in sorted(tick_groups.items(), key=lambda kv: kv[0])
        if len(v) >= 2
    ]

    numerator, denominator = time_signature
    beats_per_bar = float(numerator) * (4.0 / float(denominator))
    ticks_per_bar = int(round(ticks_per_beat * beats_per_bar))
    if ticks_per_bar <= 0:
        ticks_per_bar = ticks_per_beat * 4

    max_tick = 0
    for (t, dur, _pitch, _vel, _ch, _name) in notes:
        max_tick = max(max_tick, int(t) + int(dur))
    total_bars = max(1, (max_tick // ticks_per_bar) + 1)

    bar_counts = [0] * total_bars
    for (t, _dur, _pitch, vel, ch, _name) in notes:
        if ch == 9 or vel <= 0:
            continue
        bar = int(t) // ticks_per_bar
        if 0 <= bar < total_bars:
            bar_counts[bar] += 1

    note_density_per_beat = [float(c) for c in bar_counts]

    if tension_curve is None:
        if total_bars == 1:
            resolved_curve = [0.5]
        else:
            resolved_curve = [
                round(0.2 + (0.7 * (i / float(total_bars - 1))), 4) for i in range(total_bars)
            ]
    else:
        resolved_curve = list(tension_curve)

    return CriticFeatures(
        voicings=voicings,
        bass_notes=bass_notes,
        kick_ticks=kick_ticks,
        tension_curve=resolved_curve,
        note_density_per_beat=note_density_per_beat,
        ticks_per_beat=ticks_per_beat,
    )
