"""Deterministic pure-Python prompt-sweep generator for Apollo Gemma 4 training.

Permutes presets, arrangement templates, genres (including Ethiopian modes),
tempos, keys/all 7 modes, section counts, instrumentation, constraints, and
mastering targets while enforcing strict deduplication and zero data leakage
against held-out golden evaluation prompts.

Zero heavy dependencies: stdlib + PyYAML + Pydantic only.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml

from .schemas.arrangement_template import load_arrangement_template_yaml
from .schemas.score_plan import ScaleModeLiteral, ScorePlanV1, SectionTypeLiteral, TrackRoleLiteral


_KEYS: Tuple[str, ...] = (
    "C", "C#", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"
)
_ALL_MODES: Tuple[ScaleModeLiteral, ...] = (
    "major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "locrian"
)

_NOTE_SEMITONES: Dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}
_SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
_FLAT_KEYS = frozenset({"F", "Bb", "Eb", "Ab", "Db"})

_MODE_INTERVALS: Dict[str, Tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
}

_MODE_QUALITIES: Dict[str, Tuple[str, ...]] = {
    "major": ("", "m", "m", "", "", "m", "dim"),
    "minor": ("m", "dim", "", "m", "m", "", ""),
    "dorian": ("m", "m", "", "", "m", "dim", ""),
    "phrygian": ("m", "", "", "m", "dim", "", "m"),
    "lydian": ("", "", "m", "dim", "", "m", "m"),
    "mixolydian": ("", "m", "dim", "", "m", "m", ""),
    "locrian": ("dim", "", "m", "m", "", "", "m"),
}

_SECTION_MAP_TO_SCORE_PLAN: Dict[str, SectionTypeLiteral] = {
    "intro": "intro",
    "verse": "verse",
    "pre_chorus": "pre_chorus",
    "chorus": "chorus",
    "drop": "drop",
    "bridge": "bridge",
    "breakdown": "breakdown",
    "outro": "outro",
    "buildup": "pre_chorus",
    "variation": "verse",
}

_SECTION_BASE_ENERGY: Dict[str, Tuple[float, float]] = {
    "intro": (0.28, 0.22),
    "verse": (0.56, 0.48),
    "pre_chorus": (0.72, 0.78),
    "chorus": (0.86, 0.82),
    "drop": (0.94, 0.90),
    "bridge": (0.64, 0.68),
    "breakdown": (0.36, 0.42),
    "outro": (0.30, 0.24),
}


@dataclass(frozen=True)
class GenreSweepProfile:
    genre: str
    display_aliases: Tuple[str, ...]
    bpm_range: Tuple[int, int]
    preferred_modes: Tuple[ScaleModeLiteral, ...]
    time_signatures: Tuple[Tuple[int, int], ...]
    template_ref: str
    presets: Tuple[str, ...]
    moods: Tuple[str, ...]
    influences: Tuple[str, ...]
    melodic_tracks: Tuple[Tuple[TrackRoleLiteral, str, int, str], ...]  # (role, instrument, octave, pattern_hint)
    drum_instruments: Tuple[str, ...]
    avoid_instruments: Tuple[str, ...]
    avoid_drums: Tuple[str, ...]
    target_lufs_range: Tuple[float, float]
    brightness_range: Tuple[float, float]
    warmth_range: Tuple[float, float]
    modal_flavor_tags: Tuple[str, ...] = ()
    requires_drums: bool = True


GENRE_SWEEP_PROFILES: Tuple[GenreSweepProfile, ...] = (
    GenreSweepProfile(
        genre="trap",
        display_aliases=("trap", "atlanta trap", "modern trap", "dark trap"),
        bpm_range=(130, 156),
        preferred_modes=("minor", "phrygian", "dorian", "locrian"),
        time_signatures=((4, 4),),
        template_ref="trap",
        presets=("trap_atlanta", "wide_modern", "punchy_club"),
        moods=("dark", "menacing", "energetic", "hypnotic", "cinematic", "icy"),
        influences=("atlanta", "metro", "southern", "modern_808"),
        melodic_tracks=(
            ("bass", "808", 1, "half_time_sub_bounce"),
            ("keys", "piano", 4, "sparse_dark_triads"),
            ("pad", "pad", 4, "haunting_sustained_bed"),
            ("lead", "synth", 5, "syncopated_bell_motif"),
            ("strings", "strings", 4, "tense_staccato_ostinato"),
        ),
        drum_instruments=("kick", "snare", "hihat", "clap"),
        avoid_instruments=("brass", "organ", "guitar"),
        avoid_drums=("ride", "timpani_drum"),
        target_lufs_range=(-10.5, -8.0),
        brightness_range=(0.50, 0.78),
        warmth_range=(0.25, 0.50),
    ),
    GenreSweepProfile(
        genre="trap_soul",
        display_aliases=("trap soul", "late-night trap soul", "melodic trap soul", "moody trap-soul"),
        bpm_range=(74, 94),
        preferred_modes=("minor", "dorian", "major", "lydian"),
        time_signatures=((4, 4),),
        template_ref="trap_soul",
        presets=("trap_soul", "rnb_neosoul", "vintage_warm"),
        moods=("emotional", "moody", "velvet", "introspective", "lush", "warm"),
        influences=("toronto", "late_night_rnb", "soul_trap"),
        melodic_tracks=(
            ("bass", "808", 1, "clean_sustained_808"),
            ("keys", "rhodes", 4, "lush_minor9_voicings"),
            ("pad", "pad", 4, "airy_filtered_wash"),
            ("keys", "piano", 4, "gentle_broken_chords"),
            ("lead", "synth", 5, "soft_velvet_topline"),
        ),
        drum_instruments=("kick", "snare", "hihat", "rim"),
        avoid_instruments=("808_distorted", "supersaw", "brass"),
        avoid_drums=("hihat_roll", "hihat_triplet_roll"),
        target_lufs_range=(-12.5, -10.0),
        brightness_range=(0.35, 0.58),
        warmth_range=(0.62, 0.85),
    ),
    GenreSweepProfile(
        genre="boom_bap",
        display_aliases=("boom bap", "90s boom bap", "golden era hip hop", "east coast boom-bap"),
        bpm_range=(84, 98),
        preferred_modes=("minor", "dorian", "mixolydian"),
        time_signatures=((4, 4),),
        template_ref="boom_bap",
        presets=("boom_bap_classic", "vintage_warm", "sp1200_crunch"),
        moods=("gritty", "nostalgic", "dusty", "headnod", "soulful", "raw"),
        influences=("90s_nyc", "sp1200", "mpc_swing", "jazz_rap"),
        melodic_tracks=(
            ("bass", "bass", 2, "staccato_fingered_pocket"),
            ("keys", "rhodes", 4, "chopped_jazz_stabs"),
            ("keys", "piano", 4, "dusty_sample_chords"),
            ("lead", "sax", 4, "smoky_horn_phrases"),
            ("strings", "strings", 4, "vinyl_sampled_swells"),
        ),
        drum_instruments=("kick", "snare", "hihat"),
        avoid_instruments=("808_glide", "808_slide", "supersaw"),
        avoid_drums=("hihat_roll", "hihat_triplet_roll"),
        target_lufs_range=(-13.5, -11.0),
        brightness_range=(0.30, 0.54),
        warmth_range=(0.68, 0.90),
    ),
    GenreSweepProfile(
        genre="g_funk",
        display_aliases=("g-funk", "west coast g-funk", "90s g-funk", "laid-back gfunk"),
        bpm_range=(86, 98),
        preferred_modes=("minor", "dorian", "mixolydian"),
        time_signatures=((4, 4),),
        template_ref="boom_bap",
        presets=("vintage_warm", "wide_modern", "g_funk_90s"),
        moods=("cruising", "sunny", "laid_back", "groovy", "confident", "west_coast"),
        influences=("west_coast_90s", "pfunk", "analog_synth"),
        melodic_tracks=(
            ("bass", "bass", 2, "bouncy_moog_funk_bass"),
            ("lead", "synth_lead", 5, "portamento_whine_lead"),
            ("keys", "rhodes", 4, "funky_comping_chords"),
            ("keys", "organ", 4, "warm_drawbar_stabs"),
            ("lead", "guitar", 4, "clean_funk_skank"),
        ),
        drum_instruments=("kick", "snare", "hihat", "clap"),
        avoid_instruments=("808_distorted", "808_trap"),
        avoid_drums=("hihat_roll", "trap_snare"),
        target_lufs_range=(-12.5, -10.5),
        brightness_range=(0.45, 0.64),
        warmth_range=(0.72, 0.90),
    ),
    GenreSweepProfile(
        genre="lofi",
        display_aliases=("lofi hip hop", "chill lofi", "dusty lo-fi", "study lofi beat"),
        bpm_range=(70, 88),
        preferred_modes=("major", "minor", "dorian", "lydian"),
        time_signatures=((4, 4),),
        template_ref="lofi",
        presets=("lofi_chill", "vintage_warm", "tape_dust"),
        moods=("cozy", "mellow", "nostalgic", "dreamy", "rainy", "relaxed"),
        influences=("jazzhop", "cassette_tape", "chillhop"),
        melodic_tracks=(
            ("keys", "rhodes", 4, "warm_maj7_lofi_chords"),
            ("keys", "piano", 4, "felt_upright_voicings"),
            ("bass", "bass", 2, "round_muted_bass"),
            ("pad", "pad", 4, "soft_tape_pad"),
            ("fx", "vinyl_crackle", 3, "continuous_dust_bed"),
        ),
        drum_instruments=("kick", "snare", "hihat"),
        avoid_instruments=("supersaw", "synth_lead_bright", "brass"),
        avoid_drums=("hihat_roll", "trap_snare"),
        target_lufs_range=(-16.5, -13.5),
        brightness_range=(0.20, 0.39),
        warmth_range=(0.80, 0.95),
    ),
    GenreSweepProfile(
        genre="house",
        display_aliases=("deep house", "uplifting house", "classic house", "club house"),
        bpm_range=(120, 130),
        preferred_modes=("minor", "dorian", "major", "mixolydian"),
        time_signatures=((4, 4),),
        template_ref="house",
        presets=("house_deep", "punchy_club", "wide_modern"),
        moods=("uplifting", "hypnotic", "driving", "euphoric", "warm", "groovy"),
        influences=("chicago_house", "uk_garage", "deep_club"),
        melodic_tracks=(
            ("bass", "bass", 2, "pumping_syncopated_sub"),
            ("keys", "rhodes", 4, "staccato_house_stabs"),
            ("pad", "pad", 4, "sidechained_warm_pad"),
            ("keys", "organ", 4, "classic_m1_organ_chords"),
            ("lead", "synth", 5, "plucky_arp_hook"),
        ),
        drum_instruments=("kick", "clap", "hihat", "hihat_open"),
        avoid_instruments=("808_trap",),
        avoid_drums=("hihat_roll", "trap_snare"),
        target_lufs_range=(-11.0, -8.5),
        brightness_range=(0.55, 0.78),
        warmth_range=(0.45, 0.72),
    ),
    GenreSweepProfile(
        genre="drill",
        display_aliases=("uk drill", "dark drill", "melodic drill", "sliding 808 drill"),
        bpm_range=(138, 146),
        preferred_modes=("minor", "phrygian", "locrian"),
        time_signatures=((4, 4),),
        template_ref="trap",
        presets=("drill_uk", "wide_modern", "dark_punch"),
        moods=("gritty", "cold", "intense", "ominous", "street", "tense"),
        influences=("london_drill", "ny_drill", "sliding_808"),
        melodic_tracks=(
            ("bass", "808", 1, "gliding_octave_808"),
            ("keys", "piano", 4, "minor_arpeggio_loop"),
            ("strings", "violin", 5, "dark_spiccato_motif"),
            ("pad", "choir", 4, "eerie_minor_choir"),
        ),
        drum_instruments=("kick", "snare", "hihat", "rim"),
        avoid_instruments=("organ", "brass"),
        avoid_drums=("ride",),
        target_lufs_range=(-10.5, -8.0),
        brightness_range=(0.30, 0.49),
        warmth_range=(0.30, 0.55),
    ),
    GenreSweepProfile(
        genre="rnb",
        display_aliases=("modern rnb", "smooth r&b", "velvet rnb", "slow jam r&b"),
        bpm_range=(78, 104),
        preferred_modes=("major", "minor", "dorian", "lydian"),
        time_signatures=((4, 4), (6, 8)),
        template_ref="rnb",
        presets=("rnb_neosoul", "vintage_warm", "wide_modern"),
        moods=("sensual", "silky", "romantic", "warm", "intimate", "smooth"),
        influences=("90s_rnb", "modern_soul", "quiet_storm"),
        melodic_tracks=(
            ("keys", "rhodes", 4, "extended_maj9_voicings"),
            ("bass", "bass", 2, "smooth_pocket_bass"),
            ("pad", "pad", 4, "lush_analog_pad"),
            ("keys", "piano", 4, "expressive_gospel_fills"),
            ("lead", "guitar", 4, "clean_chorus_licks"),
        ),
        drum_instruments=("kick", "snare", "hihat", "rim"),
        avoid_instruments=("808_distorted", "supersaw"),
        avoid_drums=("hihat_roll",),
        target_lufs_range=(-13.5, -11.0),
        brightness_range=(0.38, 0.62),
        warmth_range=(0.76, 0.92),
    ),
    GenreSweepProfile(
        genre="neo_soul",
        display_aliases=("neo soul", "organic neo-soul", "jazzy neo soul", "dilla-inspired neo soul"),
        bpm_range=(82, 98),
        preferred_modes=("dorian", "minor", "major", "mixolydian"),
        time_signatures=((4, 4),),
        template_ref="rnb",
        presets=("rnb_neosoul", "vintage_warm", "jazz_swing"),
        moods=("soulful", "earthy", "warm", "laid_back", "jazzy", "expressive"),
        influences=("erykah_vibe", "dilla_swing", "philadelphia_soul"),
        melodic_tracks=(
            ("keys", "rhodes", 4, "rootless_neo_soul_chords"),
            ("bass", "bass", 2, "syncopated_warm_electric_bass"),
            ("lead", "guitar", 4, "warm_hollowbody_riffs"),
            ("keys", "organ", 4, "gospel_swells"),
            ("pad", "strings", 4, "subtle_soul_strings"),
        ),
        drum_instruments=("kick", "snare", "hihat"),
        avoid_instruments=("808", "supersaw"),
        avoid_drums=("808", "hihat_roll"),
        target_lufs_range=(-14.0, -11.5),
        brightness_range=(0.35, 0.58),
        warmth_range=(0.75, 0.92),
    ),
    GenreSweepProfile(
        genre="rock",
        display_aliases=("indie rock", "90s alternative rock", "classic rock band", "energetic rock"),
        bpm_range=(98, 142),
        preferred_modes=("minor", "major", "mixolydian", "dorian"),
        time_signatures=((4, 4),),
        template_ref="trap_soul",
        presets=("wide_modern", "vintage_warm", "rock_band"),
        moods=("driving", "anthemic", "raw", "energetic", "gritty", "uplifting"),
        influences=("90s_alt_rock", "garage_rock", "live_band"),
        melodic_tracks=(
            ("keys", "guitar", 3, "crunchy_power_chords"),
            ("bass", "bass", 2, "picked_root_eighth_bass"),
            ("lead", "guitar", 4, "melodic_overdrive_hook"),
            ("keys", "organ", 4, "sustaining_rock_organ"),
            ("keys", "piano", 4, "driving_chord_comping"),
        ),
        drum_instruments=("kick", "snare", "hihat", "crash", "ride"),
        avoid_instruments=("808", "choir"),
        avoid_drums=("808", "clap"),
        target_lufs_range=(-11.5, -9.0),
        brightness_range=(0.52, 0.78),
        warmth_range=(0.45, 0.70),
    ),
    GenreSweepProfile(
        genre="jazz",
        display_aliases=("modern jazz", "cool jazz quartet", "swinging jazz", "modal jazz"),
        bpm_range=(92, 168),
        preferred_modes=("dorian", "major", "mixolydian", "lydian", "minor"),
        time_signatures=((4, 4), (3, 4)),
        template_ref="jazz",
        presets=("jazz_swing", "vintage_warm"),
        moods=("sophisticated", "swinging", "smoky", "intimate", "playful", "modal"),
        influences=("blue_note", "modal_jazz", "bebop_trio"),
        melodic_tracks=(
            ("keys", "piano", 4, "syncopated_rootless_comping"),
            ("bass", "contrabass", 2, "walking_quarter_bass"),
            ("lead", "sax", 4, "lyrical_bebop_lines"),
            ("lead", "trumpet", 4, "muted_call_response"),
            ("keys", "rhodes", 4, "warm_modal_voicings"),
        ),
        drum_instruments=("kick", "snare", "ride", "hihat"),
        avoid_instruments=("808", "supersaw"),
        avoid_drums=("808", "hihat_roll"),
        target_lufs_range=(-16.5, -13.5),
        brightness_range=(0.40, 0.65),
        warmth_range=(0.65, 0.88),
    ),
    GenreSweepProfile(
        genre="pop",
        display_aliases=("modern pop", "synth pop", "upbeat pop", "indie pop"),
        bpm_range=(100, 128),
        preferred_modes=("major", "minor", "mixolydian", "lydian"),
        time_signatures=((4, 4),),
        template_ref="rnb",
        presets=("wide_modern", "punchy_club"),
        moods=("bright", "catchy", "uplifting", "shimmering", "energetic", "clean"),
        influences=("synth_pop", "radio_pop", "scandi_pop"),
        melodic_tracks=(
            ("keys", "piano", 4, "anthemic_four_chord_comp"),
            ("bass", "bass", 2, "tight_pop_groove"),
            ("lead", "synth", 5, "catchy_hook_melody"),
            ("pad", "pad", 4, "shimmering_chorus_lift"),
            ("keys", "guitar", 4, "crisp_funk_rhythm"),
        ),
        drum_instruments=("kick", "snare", "clap", "hihat"),
        avoid_instruments=("808_distorted",),
        avoid_drums=("timpani_drum",),
        target_lufs_range=(-11.0, -8.5),
        brightness_range=(0.55, 0.82),
        warmth_range=(0.45, 0.70),
    ),
    GenreSweepProfile(
        genre="ambient",
        display_aliases=("ambient soundscape", "ethereal ambient", "atmospheric ambient", "meditative ambient"),
        bpm_range=(58, 82),
        preferred_modes=("lydian", "minor", "dorian", "major"),
        time_signatures=((4, 4), (3, 4)),
        template_ref="ambient",
        presets=("ambient_ethereal", "vintage_warm"),
        moods=("ethereal", "vast", "serene", "floating", "celestial", "dreamy"),
        influences=("eno_ambient", "arctic_pads", "modular_drift"),
        melodic_tracks=(
            ("pad", "pad", 4, "slow_evolving_harmonics"),
            ("strings", "strings", 4, "long_legato_swells"),
            ("keys", "piano", 4, "sparse_reverberant_notes"),
            ("pad", "choir", 4, "airy_distant_voices"),
            ("fx", "pad", 3, "low_harmonic_drone"),
        ),
        drum_instruments=(),
        avoid_instruments=("808", "brass"),
        avoid_drums=("kick", "snare", "hihat", "clap"),
        target_lufs_range=(-18.5, -14.5),
        brightness_range=(0.30, 0.65),
        warmth_range=(0.60, 0.90),
        requires_drums=False,
    ),
    GenreSweepProfile(
        genre="cinematic",
        display_aliases=("cinematic score", "orchestral trailer cue", "dramatic film score", "epic orchestral"),
        bpm_range=(64, 108),
        preferred_modes=("minor", "dorian", "phrygian", "lydian"),
        time_signatures=((4, 4), (3, 4), (6, 8)),
        template_ref="ambient",
        presets=("cinematic_stage", "ambient_ethereal", "wide_modern"),
        moods=("epic", "heroic", "dramatic", "suspenseful", "sweeping", "emotional"),
        influences=("hans_score", "modern_orchestral", "chamber_cinema"),
        melodic_tracks=(
            ("strings", "strings", 4, "sweeping_legato_ostinato"),
            ("lead", "brass", 4, "bold_horn_theme"),
            ("pad", "choir", 4, "wide_cathedral_choir"),
            ("bass", "cello", 3, "deep_bowed_pedal"),
            ("keys", "piano", 4, "intimate_opening_arpeggio"),
        ),
        drum_instruments=("timpani",),
        avoid_instruments=("808", "synth_lead"),
        avoid_drums=("hihat_roll", "trap_snare"),
        target_lufs_range=(-16.0, -12.0),
        brightness_range=(0.45, 0.72),
        warmth_range=(0.55, 0.82),
        requires_drums=False,
    ),
    GenreSweepProfile(
        genre="ethio_jazz",
        display_aliases=("ethio-jazz", "addis ethio jazz", "mulatu-inspired ethio-jazz", "modal ethio jazz"),
        bpm_range=(88, 122),
        preferred_modes=("dorian", "minor", "phrygian", "mixolydian"),
        time_signatures=((4, 4), (6, 8)),
        template_ref="ethio_jazz",
        presets=("ethio_jazz", "ethio_tube_warm", "vintage_warm"),
        moods=("hypnotic", "smoky", "nocturnal", "soulful", "groovy", "mystical"),
        influences=("addis_swing", "mulatu_astatke", "qenet_modal", "ethio_funk"),
        melodic_tracks=(
            ("lead", "krar", 4, "syncopated_qenet_ostinato"),
            ("lead", "washint", 5, "breathy_modal_phrasing"),
            ("keys", "organ", 4, "warm_farfisa_comping"),
            ("bass", "bass", 2, "hypnotic_ethio_groove_bass"),
            ("lead", "sax", 4, "smoky_pentatonic_melody"),
            ("keys", "rhodes", 4, "modal_vamp_voicings"),
        ),
        drum_instruments=("kick", "snare", "hihat", "kebero"),
        avoid_instruments=("808_distorted", "supersaw"),
        avoid_drums=("hihat_roll",),
        target_lufs_range=(-15.0, -12.5),
        brightness_range=(0.40, 0.62),
        warmth_range=(0.72, 0.90),
        modal_flavor_tags=("tizita", "bati", "ambassel", "anchihoye", "qenet"),
    ),
    GenreSweepProfile(
        genre="ethiopian_traditional",
        display_aliases=("traditional ethiopian", "ethiopian folk", "azmari acoustic ensemble", "habesha traditional"),
        bpm_range=(92, 128),
        preferred_modes=("minor", "dorian", "major", "phrygian"),
        time_signatures=((6, 8), (4, 4), (3, 4)),
        template_ref="ethiopian_traditional",
        presets=("ethio_jazz", "vintage_warm"),
        moods=("ceremonial", "devotional", "warm", "expressive", "earthy", "joyful"),
        influences=("azmari_tradition", "highland_folk", "habesha_qenet"),
        melodic_tracks=(
            ("lead", "krar", 4, "pentatonic_plucked_theme"),
            ("lead", "masenqo", 4, "expressive_bowed_answers"),
            ("lead", "washint", 5, "ornamented_flute_melody"),
            ("bass", "begena", 3, "resonant_drone_pulse"),
        ),
        drum_instruments=("kebero",),
        avoid_instruments=("808", "808_distorted", "synth_lead", "synth_lead_bright"),
        avoid_drums=("808", "hihat_roll"),
        target_lufs_range=(-16.5, -13.5),
        brightness_range=(0.40, 0.65),
        warmth_range=(0.72, 0.90),
        modal_flavor_tags=("tizita", "bati", "ambassel", "anchihoye", "qenet"),
    ),
    GenreSweepProfile(
        genre="eskista",
        display_aliases=("eskista", "eskista shoulder dance", "energetic eskista groove", "ethiopian eskista"),
        bpm_range=(112, 138),
        preferred_modes=("minor", "dorian", "mixolydian"),
        time_signatures=((6, 8), (4, 4)),
        template_ref="eskista",
        presets=("ethio_jazz", "vintage_warm"),
        moods=("festive", "energetic", "percussive", "driving", "celebratory", "rhythmic"),
        influences=("eskista_dance", "kebero_groove", "azmari_dance"),
        melodic_tracks=(
            ("lead", "krar", 4, "driving_rhythmic_plucks"),
            ("lead", "washint", 5, "bright_call_response_riffs"),
            ("lead", "masenqo", 4, "sharp_accented_phrases"),
            ("bass", "begena", 3, "grounded_modal_anchor"),
        ),
        drum_instruments=("kebero",),
        avoid_instruments=("808", "808_distorted", "synth_lead"),
        avoid_drums=("808", "hihat_roll"),
        target_lufs_range=(-15.0, -12.0),
        brightness_range=(0.45, 0.70),
        warmth_range=(0.65, 0.85),
        modal_flavor_tags=("tizita", "bati", "ambassel", "anchihoye"),
    ),
)

_FALLBACK_TEMPLATES: Dict[str, List[Tuple[SectionTypeLiteral, int, str]]] = {
    "trap": [
        ("intro", 4, "Intro"),
        ("drop", 8, "Drop 1"),
        ("breakdown", 4, "Breakdown"),
        ("pre_chorus", 4, "Build"),
        ("drop", 8, "Drop 2"),
        ("outro", 4, "Outro"),
    ],
    "trap_soul": [
        ("intro", 4, "Intro"),
        ("verse", 8, "Verse 1"),
        ("chorus", 8, "Chorus 1"),
        ("bridge", 4, "Bridge"),
        ("chorus", 8, "Chorus 2"),
        ("outro", 4, "Outro"),
    ],
    "boom_bap": [
        ("intro", 4, "Intro"),
        ("verse", 8, "Verse 1"),
        ("chorus", 4, "Hook 1"),
        ("verse", 8, "Verse 2"),
        ("chorus", 4, "Hook 2"),
        ("outro", 4, "Outro"),
    ],
    "lofi": [
        ("intro", 4, "Intro"),
        ("verse", 8, "Loop A"),
        ("breakdown", 4, "Tape Break"),
        ("verse", 8, "Loop B"),
        ("outro", 4, "Fade Out"),
    ],
    "house": [
        ("intro", 4, "Intro"),
        ("pre_chorus", 4, "Build"),
        ("drop", 8, "Main Groove"),
        ("breakdown", 4, "Breakdown"),
        ("drop", 8, "Peak Drop"),
        ("outro", 4, "Outro"),
    ],
    "rnb": [
        ("intro", 4, "Intro"),
        ("verse", 8, "Verse"),
        ("pre_chorus", 4, "Pre-Chorus"),
        ("chorus", 8, "Chorus"),
        ("outro", 4, "Outro"),
    ],
    "jazz": [
        ("intro", 4, "Intro"),
        ("verse", 8, "Head"),
        ("bridge", 8, "Solo"),
        ("chorus", 8, "Head Out"),
        ("outro", 4, "Tag"),
    ],
    "ambient": [
        ("intro", 4, "Atmosphere"),
        ("verse", 8, "Drift"),
        ("chorus", 8, "Bloom"),
        ("outro", 4, "Dissolve"),
    ],
    "ethio_jazz": [
        ("intro", 4, "Intro"),
        ("verse", 8, "Qenet Theme"),
        ("bridge", 4, "Solo"),
        ("chorus", 8, "Main Groove"),
        ("outro", 4, "Outro"),
    ],
    "ethiopian_traditional": [
        ("intro", 4, "Call"),
        ("verse", 8, "Azmari Verse"),
        ("chorus", 8, "Ensemble Response"),
        ("outro", 4, "Cadence"),
    ],
    "eskista": [
        ("intro", 4, "Kebero Entry"),
        ("verse", 8, "Dance Groove"),
        ("chorus", 8, "Eskista Peak"),
        ("outro", 4, "Outro"),
    ],
}


@dataclass(frozen=True)
class SweepCandidate:
    """One deduplicated synthetic prompt paired with its verified ``ScorePlanV1``."""

    record_id: str
    prompt: str
    genre: str
    seed: int
    score_plan: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_prompt_text(text: str) -> str:
    """Normalize prompt text for exact deduplication and leakage checks."""
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _token_set(text: str) -> Set[str]:
    norm = normalize_prompt_text(text)
    return {tok for tok in norm.split() if len(tok) >= 2}


def token_jaccard_similarity(a: str, b: str) -> float:
    ta = _token_set(a)
    tb = _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


def is_leaking_against_excluded(
    candidate_prompt: str,
    excluded_normalized: Set[str],
    excluded_token_sets: Sequence[Set[str]],
    *,
    jaccard_threshold: float = 0.65,
) -> bool:
    """Return True if *candidate_prompt* matches or closely overlaps any excluded prompt."""
    norm = normalize_prompt_text(candidate_prompt)
    if not norm or norm in excluded_normalized:
        return True
    cand_tokens = _token_set(norm)
    if not cand_tokens:
        return True
    for ref_tokens in excluded_token_sets:
        if not ref_tokens:
            continue
        union = len(cand_tokens | ref_tokens)
        if union > 0 and (len(cand_tokens & ref_tokens) / float(union)) >= jaccard_threshold:
            return True
    return False


def load_yaml_templates(templates_dir: Optional[Path]) -> Dict[str, List[Tuple[SectionTypeLiteral, int, str]]]:
    """Load real arrangement templates from YAML (falling back cleanly to built-ins)."""
    loaded: Dict[str, List[Tuple[SectionTypeLiteral, int, str]]] = {
        k: list(v) for k, v in _FALLBACK_TEMPLATES.items()
    }
    if templates_dir is None or not templates_dir.is_dir():
        return loaded

    for yaml_path in sorted(templates_dir.glob("*.yaml")):
        try:
            tpl = load_arrangement_template_yaml(yaml_path)
        except Exception:
            continue
        mapped: List[Tuple[SectionTypeLiteral, int, str]] = []
        for idx, sec in enumerate(tpl.sections):
            sec_type = _SECTION_MAP_TO_SCORE_PLAN.get(sec.type, "verse")
            label = sec.label or f"{sec_type.replace('_', ' ').title()} {idx + 1}"
            mapped.append((sec_type, max(2, min(16, int(sec.bars))), label))
        if mapped:
            loaded[tpl.template_id] = mapped
    return loaded


def load_yaml_preset_metadata(presets_dir: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Load optional preset mix/mastering targets from ``configs/presets/*.yaml``."""
    presets: Dict[str, Dict[str, Any]] = {}
    if presets_dir is None or not presets_dir.is_dir():
        return presets

    for yaml_path in sorted(presets_dir.glob("*.yaml")):
        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(raw, dict) and raw.get("name"):
            presets[str(raw["name"])] = raw
    return presets


def _build_diatonic_chord_map(
    key: str,
    mode: ScaleModeLiteral,
    total_bars: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    root_pc = _NOTE_SEMITONES.get(key, 0)
    intervals = _MODE_INTERVALS[mode]
    qualities = _MODE_QUALITIES[mode]
    note_names = _FLAT_NAMES if (key in _FLAT_KEYS or "b" in key) else _SHARP_NAMES

    progressions: Tuple[Tuple[int, ...], ...] = (
        (0, 5, 3, 4),
        (0, 3, 4, 0),
        (0, 6, 5, 4),
        (0, 2, 5, 3),
        (0, 4, 5, 3),
        (0, 1, 4, 0),
        (0, 3, 6, 2),
    )
    degrees = rng.choice(progressions)
    bars_per_chord = rng.choice((1, 2))
    chord_map: List[Dict[str, Any]] = []
    bar = 1
    idx = 0
    while bar <= total_bars:
        deg = degrees[idx % len(degrees)]
        pc = (root_pc + intervals[deg]) % 12
        qual = qualities[deg]
        chord_name = f"{note_names[pc]}{qual}"
        chord_map.append({"bar": bar, "chord": chord_name})
        bar += bars_per_chord
        idx += 1
    return chord_map


def _build_sections_and_cues(
    template_sections: Sequence[Tuple[SectionTypeLiteral, int, str]],
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[float], List[Dict[str, Any]], int]:
    """Slice/permute a template into 3..6 varied sections with energy, tension, and cue points."""
    base = list(template_sections)
    target_count = rng.randint(3, min(6, max(3, len(base))))
    if len(base) > target_count:
        # Always preserve opening and closing feel while sampling middle structure
        middle = base[1:-1]
        picked_mid = middle[: max(1, target_count - 2)]
        chosen = [base[0], *picked_mid, base[-1]]
    else:
        chosen = base

    bar_scale = rng.choice((0.5, 1.0, 1.0))
    sections: List[Dict[str, Any]] = []
    tension_curve: List[float] = []
    cue_points: List[Dict[str, Any]] = []
    running_bar = 1

    for idx, (sec_type, raw_bars, label) in enumerate(chosen):
        bars = max(2, min(12, int(round(raw_bars * bar_scale))))
        if bars % 2 == 1 and bars > 2:
            bars -= 1
        base_e, base_t = _SECTION_BASE_ENERGY.get(sec_type, (0.55, 0.50))
        energy = round(max(0.10, min(0.98, base_e + rng.uniform(-0.10, 0.10))), 2)
        tension = round(max(0.10, min(0.98, base_t + rng.uniform(-0.10, 0.10))), 2)

        sections.append(
            {
                "name": label if rng.random() < 0.6 else sec_type.replace("_", " ").title(),
                "type": sec_type,
                "bars": bars,
                "energy": energy,
                "tension": tension,
            }
        )

        # Build bar-aligned tension ramp inside section
        next_t = min(0.98, tension + (0.12 if sec_type in ("intro", "pre_chorus") else 0.0))
        for b in range(bars):
            frac = b / float(max(1, bars - 1)) if bars > 1 else 0.0
            val = tension + (next_t - tension) * frac
            tension_curve.append(round(max(0.05, min(0.99, val)), 3))

        # Add musical cue points at structural boundaries
        if idx > 0:
            cue_type = (
                "drop"
                if sec_type in ("drop", "chorus")
                else ("breakdown" if sec_type == "breakdown" else ("build" if sec_type == "pre_chorus" else "fill"))
            )
            cue_points.append(
                {
                    "bar": max(1, running_bar - (1 if cue_type == "fill" else 0)),
                    "type": cue_type,
                    "intensity": round(max(0.25, min(0.98, energy)), 2),
                }
            )
        running_bar += bars

    total_bars = sum(int(s["bars"]) for s in sections)
    return sections, tension_curve, cue_points, total_bars


def _format_prompt_variant(
    *,
    profile: GenreSweepProfile,
    bpm: int,
    key: str,
    mode: ScaleModeLiteral,
    time_sig: Tuple[int, int],
    mood: str,
    preset: str,
    sections: Sequence[Dict[str, Any]],
    melodic_names: Sequence[str],
    drum_names: Sequence[str],
    avoid_names: Sequence[str],
    rng: random.Random,
) -> str:
    genre_alias = rng.choice(profile.display_aliases)
    mood_clean = mood.replace("_", " ")
    inst_str = ", ".join(i.replace("_", " ") for i in melodic_names)
    drum_str = ", ".join(d.replace("_", " ") for d in drum_names)
    sec_summary = " -> ".join(str(s["type"]).replace("_", " ") for s in sections)
    total_bars = sum(int(s["bars"]) for s in sections)
    meter_tag = f" in {time_sig[0]}/{time_sig[1]} time" if time_sig != (4, 4) else ""
    modal_tag = f" ({rng.choice(profile.modal_flavor_tags)} feel)" if profile.modal_flavor_tags and rng.random() < 0.7 else ""
    avoid_clause = (
        f"; avoid {', '.join(a.replace('_', ' ') for a in avoid_names[:2])}"
        if avoid_names and rng.random() < 0.45
        else ""
    )
    drum_clause = f" and {drum_str} percussion" if drum_str else " with no drums"
    preset_clause = preset.replace("_", " ")

    style = rng.randrange(8)
    if style == 0:
        return (
            f"{mood_clean} {genre_alias}{modal_tag} at {bpm} BPM in {key} {mode}{meter_tag} "
            f"featuring {inst_str}{drum_clause}, {total_bars} bars{avoid_clause}"
        )
    if style == 1:
        return (
            f"Compose a {total_bars}-bar {mood_clean} {genre_alias}{modal_tag} track ({bpm} BPM, {key} {mode}{meter_tag}) "
            f"with {inst_str}{drum_clause}. Structure: {sec_summary}{avoid_clause}."
        )
    if style == 2:
        return (
            f"{genre_alias} production in {key} {mode} at {bpm} bpm{meter_tag}, {mood_clean} energy, "
            f"layered with {inst_str}{drum_clause}, {preset_clause} mix aesthetic{avoid_clause}"
        )
    if style == 3:
        return (
            f"Build an arrangement ({sec_summary}) for a {mood_clean} {genre_alias}{modal_tag} piece "
            f"at {bpm} BPM in {key} {mode}{meter_tag} using {inst_str}{drum_clause}{avoid_clause}"
        )
    if style == 4:
        return (
            f"{bpm} BPM {key} {mode} {genre_alias}{modal_tag} — {mood_clean} vibe, "
            f"primary instruments: {inst_str}{drum_clause}, {total_bars} bars total{avoid_clause}"
        )
    if style == 5:
        return (
            f"Create a {mood_clean} {genre_alias} instrumental{modal_tag} ({key} {mode}, {bpm} BPM{meter_tag}) "
            f"with {inst_str}{drum_clause} and a {preset_clause} master finish{avoid_clause}"
        )
    if style == 6:
        return (
            f"Producer brief: {genre_alias}{modal_tag}, {mood_clean} tone, {bpm} BPM, {key} {mode}{meter_tag}, "
            f"{len(sections)} sections ({sec_summary}), instruments [{inst_str}]{drum_clause}{avoid_clause}"
        )
    return (
        f"Write a {total_bars} bar {genre_alias}{modal_tag} cue in {key} {mode} at {bpm} BPM{meter_tag} "
        f"carrying a {mood_clean} atmosphere with {inst_str}{drum_clause}{avoid_clause}"
    )


def generate_prompt_sweep(
    *,
    count: int,
    seed: int = 2026,
    exclude_prompts: Iterable[str] = (),
    templates_dir: Optional[Path] = None,
    presets_dir: Optional[Path] = None,
    jaccard_threshold: float = 0.65,
    max_attempts_multiplier: int = 12,
) -> List[SweepCandidate]:
    """Generate *count* deduplicated, schema-valid synthetic ``SweepCandidate`` entries.

    Balances evenly across all 17 genres (including Ethiopian modes), all 12 keys,
    all 7 modes, real arrangement templates, presets, section lengths, and
    genre-compliant instrumentation with zero leakage against *exclude_prompts*.
    """
    if count <= 0:
        return []

    rng = random.Random(int(seed))
    templates = load_yaml_templates(templates_dir)
    preset_meta = load_yaml_preset_metadata(presets_dir)

    excluded_norm: Set[str] = set()
    excluded_tokens: List[Set[str]] = []
    for raw in exclude_prompts:
        norm = normalize_prompt_text(raw)
        if norm:
            excluded_norm.add(norm)
            excluded_tokens.append(_token_set(norm))

    seen_prompts: Set[str] = set()
    seen_structures: Set[Tuple[Any, ...]] = set()
    candidates: List[SweepCandidate] = []

    max_attempts = max(count * max_attempts_multiplier, count + 500)
    profile_count = len(GENRE_SWEEP_PROFILES)

    for attempt in range(max_attempts):
        if len(candidates) >= count:
            break

        # Round-robin genres with jitter so every genre (including Ethiopian) is evenly represented
        profile = GENRE_SWEEP_PROFILES[(len(candidates) + (attempt // max(1, count))) % profile_count]

        bpm = rng.randint(profile.bpm_range[0], profile.bpm_range[1])
        key = rng.choice(_KEYS)
        # 88% genre-idiomatic mode, 12% full 7-mode exploration for mode coverage
        mode: ScaleModeLiteral = (
            rng.choice(profile.preferred_modes)
            if rng.random() < 0.88
            else rng.choice(_ALL_MODES)
        )
        time_sig = rng.choice(profile.time_signatures)
        mood = rng.choice(profile.moods)
        preset = rng.choice(profile.presets)

        tpl_sections = templates.get(profile.template_ref) or _FALLBACK_TEMPLATES["trap_soul"]
        sections, tension_curve, cue_points, total_bars = _build_sections_and_cues(tpl_sections, rng)

        # Pick 2..4 melodic tracks (preserving genre-mandatory elements first)
        n_melodic = rng.randint(2, min(4, len(profile.melodic_tracks)))
        # Keep first 2 core tracks frequently for genre conformance (e.g. 808/rhodes/krar/bass)
        core_count = min(2, n_melodic)
        rest_pool = list(profile.melodic_tracks[core_count:])
        rng.shuffle(rest_pool)
        chosen_melodic = list(profile.melodic_tracks[:core_count]) + rest_pool[: n_melodic - core_count]

        # Pick genre-compliant drums (if genre uses drums)
        chosen_drums: List[str] = []
        if profile.drum_instruments:
            if profile.requires_drums:
                # Always keep core kit elements so mandatory drum rules pass cleanly
                k_min = min(len(profile.drum_instruments), 3)
                k = rng.randint(k_min, len(profile.drum_instruments))
                chosen_drums = list(profile.drum_instruments[:k])
            elif rng.random() < 0.5:
                chosen_drums = list(profile.drum_instruments)

        tracks: List[Dict[str, Any]] = []
        section_names = [str(s["name"]) for s in sections]
        for role, inst, octave, pattern_hint in chosen_melodic:
            track_entry: Dict[str, Any] = {
                "role": role,
                "instrument": inst,
                "pattern_hint": pattern_hint,
                "octave": octave,
                "density": round(rng.uniform(0.30, 0.85), 2),
            }
            if rng.random() < 0.45 and len(section_names) >= 3:
                track_entry["activation"] = [
                    {
                        "section": s_name,
                        "active": not (idx == 0 and role == "lead"),
                    }
                    for idx, s_name in enumerate(section_names)
                ]
            tracks.append(track_entry)

        for drum_inst in chosen_drums:
            tracks.append(
                {
                    "role": "drums",
                    "instrument": drum_inst,
                    "density": round(rng.uniform(0.45, 0.90), 2),
                }
            )

        # Ensure lofi texture track if lofi genre so mandatory texture pattern passes
        if profile.genre == "lofi" and not any(t["instrument"] == "vinyl_crackle" for t in tracks):
            tracks.append({"role": "fx", "instrument": "vinyl_crackle", "density": 0.25})

        # Constraints
        avoid_insts = list(profile.avoid_instruments[: rng.randint(1, len(profile.avoid_instruments))]) if profile.avoid_instruments else []
        avoid_drums = list(profile.avoid_drums[: rng.randint(1, len(profile.avoid_drums))]) if profile.avoid_drums else []
        constraints: Dict[str, Any] = {"max_polyphony": rng.choice((8, 10, 12, 16))}
        if avoid_insts:
            constraints["avoid_instruments"] = avoid_insts
        if avoid_drums:
            constraints["avoid_drums"] = avoid_drums

        # Mastering (pulling from YAML preset when present)
        p_yaml = preset_meta.get(preset, {})
        raw_mix = p_yaml.get("mix")
        raw_mast = p_yaml.get("mastering")
        yaml_mix: Dict[str, Any] = raw_mix if isinstance(raw_mix, dict) else {}
        yaml_mast: Dict[str, Any] = raw_mast if isinstance(raw_mast, dict) else {}

        lufs = float(
            yaml_mast.get(
                "target_lufs",
                round(rng.uniform(profile.target_lufs_range[0], profile.target_lufs_range[1]), 1),
            )
        )
        lufs = round(max(-30.0, min(-6.0, lufs + rng.uniform(-0.8, 0.8))), 1)
        bright = float(
            yaml_mix.get(
                "brightness_target",
                round(rng.uniform(profile.brightness_range[0], profile.brightness_range[1]), 2),
            )
        )
        bright = round(max(0.05, min(0.95, bright + rng.uniform(-0.06, 0.06))), 2)
        warmth = float(
            yaml_mix.get(
                "warmth_target",
                round(rng.uniform(profile.warmth_range[0], profile.warmth_range[1]), 2),
            )
        )
        warmth = round(max(0.05, min(0.95, warmth + rng.uniform(-0.06, 0.06))), 2)

        mastering: Dict[str, Any] = {
            "production_preset": preset,
            "target_lufs": lufs,
            "master_ceiling_db": round(rng.choice((-1.5, -1.0, -0.8, -0.5)), 1),
            "stem_headroom_db": round(rng.choice((-6.0, -4.5, -3.0)), 1),
            "brightness_target": bright,
            "warmth_target": warmth,
        }

        melodic_names = [inst for (_r, inst, _o, _h) in chosen_melodic]
        prompt = _format_prompt_variant(
            profile=profile,
            bpm=bpm,
            key=key,
            mode=mode,
            time_sig=time_sig,
            mood=mood,
            preset=preset,
            sections=sections,
            melodic_names=melodic_names,
            drum_names=chosen_drums,
            avoid_names=avoid_insts + avoid_drums,
            rng=rng,
        )

        # Hard zero-leakage and deduplication gates
        if is_leaking_against_excluded(
            prompt,
            excluded_norm,
            excluded_tokens,
            jaccard_threshold=jaccard_threshold,
        ):
            continue

        norm_prompt = normalize_prompt_text(prompt)
        if norm_prompt in seen_prompts:
            continue

        struct_key = (
            profile.genre,
            bpm,
            key,
            mode,
            time_sig,
            tuple((s["type"], s["bars"]) for s in sections),
            tuple(sorted(melodic_names)),
            tuple(sorted(chosen_drums)),
        )
        if struct_key in seen_structures:
            continue

        cand_seed = int(seed) * 100003 + len(candidates)
        chord_map = _build_diatonic_chord_map(key, mode, total_bars, rng)
        influences = list(rng.sample(profile.influences, k=min(2, len(profile.influences))))

        plan_dict: Dict[str, Any] = {
            "schema_version": "score_plan_v1",
            "prompt": prompt,
            "bpm": float(bpm),
            "key": key,
            "mode": mode,
            "time_signature": [time_sig[0], time_sig[1]],
            "genre": profile.genre,
            "mood": mood,
            "influences": influences,
            "seed": cand_seed,
            "duration_bars": total_bars,
            "sections": sections,
            "chord_map": chord_map,
            "tension_curve": tension_curve,
            "cue_points": cue_points,
            "tracks": tracks,
            "constraints": constraints,
            "mastering": mastering,
        }

        # Strict Pydantic validation before accepting candidate
        ScorePlanV1.model_validate(plan_dict)

        seen_prompts.add(norm_prompt)
        seen_structures.add(struct_key)
        candidates.append(
            SweepCandidate(
                record_id=f"sweep_{profile.genre}_{len(candidates):05d}",
                prompt=prompt,
                genre=profile.genre,
                seed=cand_seed,
                score_plan=plan_dict,
                metadata={
                    "source": "prompt_sweep_v1",
                    "preset": preset,
                    "template_ref": profile.template_ref,
                },
            )
        )

    return candidates
