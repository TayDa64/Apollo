"""Pure-Python instrument pitch ranges and register knowledge base.

Mirrors ``multimodal_gen.instrument_ranges`` with zero heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_TO_SHARP = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}


def note_name_to_midi(note_name: str, octave: int = 4) -> int:
    """Convert a note name (e.g. ``'C'``, ``'F#'``, ``'Bb'``) and octave to MIDI pitch."""
    note_upper = note_name.upper().strip()
    if note_upper in _FLAT_TO_SHARP:
        note_upper = _FLAT_TO_SHARP[note_upper]
    try:
        note_index = _NOTE_NAMES.index(note_upper)
    except ValueError:
        note_index = _NOTE_NAMES.index(note_upper[0])
    return (octave + 1) * 12 + note_index


@dataclass(frozen=True)
class InstrumentRange:
    low: int
    high: int
    sweet_low: int
    sweet_high: int
    chord_octave: int
    melody_octave: int
    bass_octave: Optional[int]
    role: str


INSTRUMENT_RANGES: Dict[str, InstrumentRange] = {
    # Orchestral Strings
    "violin": InstrumentRange(
        low=55, high=100, sweet_low=60, sweet_high=84,
        chord_octave=4, melody_octave=5, bass_octave=None, role="melody",
    ),
    "viola": InstrumentRange(
        low=48, high=88, sweet_low=55, sweet_high=79,
        chord_octave=4, melody_octave=4, bass_octave=None, role="harmony",
    ),
    "cello": InstrumentRange(
        low=36, high=76, sweet_low=43, sweet_high=67,
        chord_octave=3, melody_octave=3, bass_octave=None, role="harmony",
    ),
    "contrabass": InstrumentRange(
        low=28, high=60, sweet_low=28, sweet_high=48,
        chord_octave=2, melody_octave=2, bass_octave=1, role="bass",
    ),
    "strings": InstrumentRange(
        low=36, high=96, sweet_low=48, sweet_high=84,
        chord_octave=4, melody_octave=5, bass_octave=None, role="harmony",
    ),
    # Orchestral Brass
    "trumpet": InstrumentRange(
        low=54, high=84, sweet_low=58, sweet_high=79,
        chord_octave=4, melody_octave=4, bass_octave=None, role="melody",
    ),
    "trombone": InstrumentRange(
        low=40, high=72, sweet_low=46, sweet_high=65,
        chord_octave=3, melody_octave=3, bass_octave=None, role="harmony",
    ),
    "french_horn": InstrumentRange(
        low=35, high=77, sweet_low=43, sweet_high=67,
        chord_octave=3, melody_octave=4, bass_octave=None, role="harmony",
    ),
    "tuba": InstrumentRange(
        low=26, high=65, sweet_low=30, sweet_high=53,
        chord_octave=2, melody_octave=2, bass_octave=1, role="bass",
    ),
    "brass": InstrumentRange(
        low=40, high=84, sweet_low=48, sweet_high=72,
        chord_octave=3, melody_octave=4, bass_octave=None, role="harmony",
    ),
    # Orchestral Woodwinds / jazz saxes
    "sax": InstrumentRange(
        low=49, high=80, sweet_low=55, sweet_high=76,
        chord_octave=3, melody_octave=4, bass_octave=None, role="melody",
    ),
    "alto_sax": InstrumentRange(
        low=49, high=80, sweet_low=55, sweet_high=76,
        chord_octave=3, melody_octave=4, bass_octave=None, role="melody",
    ),
    "tenor_sax": InstrumentRange(
        low=44, high=77, sweet_low=50, sweet_high=72,
        chord_octave=3, melody_octave=3, bass_octave=None, role="melody",
    ),
    "flute": InstrumentRange(
        low=60, high=96, sweet_low=67, sweet_high=88,
        chord_octave=5, melody_octave=5, bass_octave=None, role="melody",
    ),
    "oboe": InstrumentRange(
        low=58, high=93, sweet_low=62, sweet_high=84,
        chord_octave=4, melody_octave=5, bass_octave=None, role="melody",
    ),
    "clarinet": InstrumentRange(
        low=50, high=91, sweet_low=58, sweet_high=82,
        chord_octave=4, melody_octave=4, bass_octave=None, role="melody",
    ),
    "bassoon": InstrumentRange(
        low=34, high=72, sweet_low=41, sweet_high=60,
        chord_octave=3, melody_octave=3, bass_octave=None, role="harmony",
    ),
    # Orchestral Percussion
    "timpani": InstrumentRange(
        low=38, high=60, sweet_low=40, sweet_high=55,
        chord_octave=2, melody_octave=2, bass_octave=None, role="percussion",
    ),
    "harp": InstrumentRange(
        low=24, high=103, sweet_low=36, sweet_high=84,
        chord_octave=4, melody_octave=4, bass_octave=None, role="harmony",
    ),
    # Choir
    "choir": InstrumentRange(
        low=43, high=84, sweet_low=48, sweet_high=79,
        chord_octave=3, melody_octave=4, bass_octave=None, role="harmony",
    ),
    "soprano": InstrumentRange(
        low=60, high=84, sweet_low=62, sweet_high=81,
        chord_octave=5, melody_octave=5, bass_octave=None, role="melody",
    ),
    "alto": InstrumentRange(
        low=53, high=77, sweet_low=55, sweet_high=74,
        chord_octave=4, melody_octave=4, bass_octave=None, role="harmony",
    ),
    "tenor": InstrumentRange(
        low=48, high=72, sweet_low=50, sweet_high=69,
        chord_octave=4, melody_octave=4, bass_octave=None, role="harmony",
    ),
    "bass_voice": InstrumentRange(
        low=36, high=60, sweet_low=38, sweet_high=57,
        chord_octave=3, melody_octave=3, bass_octave=None, role="bass",
    ),
    # Keys / Synth
    "piano": InstrumentRange(
        low=21, high=108, sweet_low=40, sweet_high=84,
        chord_octave=4, melody_octave=5, bass_octave=None, role="harmony",
    ),
    "rhodes": InstrumentRange(
        low=36, high=96, sweet_low=48, sweet_high=84,
        chord_octave=4, melody_octave=5, bass_octave=None, role="harmony",
    ),
    "organ": InstrumentRange(
        low=36, high=96, sweet_low=48, sweet_high=84,
        chord_octave=4, melody_octave=4, bass_octave=None, role="harmony",
    ),
    "synth": InstrumentRange(
        low=24, high=108, sweet_low=36, sweet_high=96,
        chord_octave=4, melody_octave=5, bass_octave=None, role="melody",
    ),
    "synth_lead": InstrumentRange(
        low=24, high=108, sweet_low=36, sweet_high=96,
        chord_octave=4, melody_octave=5, bass_octave=None, role="melody",
    ),
    "pad": InstrumentRange(
        low=36, high=96, sweet_low=48, sweet_high=84,
        chord_octave=3, melody_octave=4, bass_octave=None, role="harmony",
    ),
    # Bass Instruments
    "bass": InstrumentRange(
        low=28, high=67, sweet_low=28, sweet_high=48,
        chord_octave=2, melody_octave=2, bass_octave=1, role="bass",
    ),
    "808": InstrumentRange(
        low=12, high=48, sweet_low=24, sweet_high=40,
        chord_octave=1, melody_octave=1, bass_octave=1, role="bass",
    ),
    # Ethiopian Instruments
    "krar": InstrumentRange(
        low=60, high=84, sweet_low=62, sweet_high=79,
        chord_octave=4, melody_octave=5, bass_octave=None, role="melody",
    ),
    "masenqo": InstrumentRange(
        low=55, high=79, sweet_low=60, sweet_high=74,
        chord_octave=4, melody_octave=4, bass_octave=None, role="melody",
    ),
    "washint": InstrumentRange(
        low=62, high=86, sweet_low=65, sweet_high=79,
        chord_octave=5, melody_octave=5, bass_octave=None, role="melody",
    ),
    "begena": InstrumentRange(
        low=48, high=72, sweet_low=52, sweet_high=67,
        chord_octave=3, melody_octave=3, bass_octave=None, role="harmony",
    ),
    # Guitar
    "guitar": InstrumentRange(
        low=40, high=88, sweet_low=48, sweet_high=79,
        chord_octave=3, melody_octave=4, bass_octave=None, role="harmony",
    ),
}

_ALIASES: Dict[str, str] = {
    "electric_piano": "rhodes",
    "e_piano": "rhodes",
    "epiano": "rhodes",
    "acoustic_guitar": "guitar",
    "electric_guitar": "guitar",
    "string_ensemble": "strings",
    "string_section": "strings",
    "double_bass": "contrabass",
    "upright_bass": "contrabass",
    "horn": "french_horn",
    "french horn": "french_horn",
    "saxophone": "sax",
    "saxes": "sax",
    "saxophones": "sax",
    "alto sax": "alto_sax",
    "alto_saxophone": "alto_sax",
    "alto saxophone": "alto_sax",
    "tenor sax": "tenor_sax",
    "tenor_saxophone": "tenor_sax",
    "tenor saxophone": "tenor_sax",
    "bass guitar": "bass",
    "electric_bass": "bass",
    "synth_bass": "808",
    "sub_bass": "808",
    "sub bass": "808",
    "synthesizer": "synth",
    "lead": "synth_lead",
    "melody": "synth_lead",
    "vocals": "choir",
    "voice": "choir",
    "harpsichord": "piano",
    "celesta": "piano",
}

_DEFAULT_RANGE = INSTRUMENT_RANGES["piano"]


def get_range(instrument: str) -> InstrumentRange:
    """Get the pitch range for an instrument (falls back to piano)."""
    key = instrument.strip().lower()
    if key in INSTRUMENT_RANGES:
        return INSTRUMENT_RANGES[key]
    alias = _ALIASES.get(key)
    if alias and alias in INSTRUMENT_RANGES:
        return INSTRUMENT_RANGES[alias]
    for name in INSTRUMENT_RANGES:
        if name in key or key in name:
            return INSTRUMENT_RANGES[name]
    return _DEFAULT_RANGE


def clamp_to_range(pitch: int, instrument: str) -> int:
    """Clamp a MIDI pitch to the playable range of an instrument, octave-shifting if needed."""
    r = get_range(instrument)
    if r.low <= pitch <= r.high:
        return pitch
    while pitch < r.low:
        pitch += 12
    while pitch > r.high:
        pitch -= 12
    return max(r.low, min(r.high, pitch))


def get_chord_octave(instrument: str) -> int:
    return get_range(instrument).chord_octave


def get_melody_octave(instrument: str) -> int:
    return get_range(instrument).melody_octave


def get_bass_octave(instrument: str) -> int:
    r = get_range(instrument)
    return r.bass_octave if r.bass_octave is not None else 1


def clamp_chord_notes(notes: Sequence[int], instrument: str) -> List[int]:
    return [clamp_to_range(int(n), instrument) for n in notes]


def get_timpani_pitches(key: str) -> List[int]:
    """Get musically correct timpani pitches for a given key (D2-C4)."""
    tonic = note_name_to_midi(key, 2)
    while tonic < 38:
        tonic += 12
    while tonic > 55:
        tonic -= 12

    dominant = tonic + 7
    subdominant = tonic + 5
    pitches = [p for p in (tonic, subdominant, dominant) if 38 <= p <= 60]
    if not pitches:
        pitches = [43]
    return sorted(pitches)
