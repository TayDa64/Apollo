"""Deterministic, bounded ``[0.0, 1.0]`` GRPO reward functions for MUSE.

All rewards are pure Python, deterministic, finite, and strictly bounded in
``[0.0, 1.0]`` so they drop directly into TRL / Unsloth ``GRPOTrainer`` loops
without requiring JUCE, FluidSynth, or heavy audio/DSP extensions.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from pydantic import ValidationError

from ..schemas.score_plan import ScorePlanV1
from .critics import run_all_critics
from .genre_rules import GenreRulesEngine, RuleSeverity, get_genre_rules
from .instrument_ranges import get_range
from .midi_features import (
    extract_critic_features_from_midi,
    extract_note_tuples_from_midi,
)
from .quality import QualityValidator


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
_REQUIRED_SCORE_PLAN_KEYS = (
    "schema_version",
    "prompt",
    "bpm",
    "key",
    "mode",
    "sections",
    "tracks",
)

DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "schema": 0.30,
    "genre_rules": 0.20,
    "instrument_range": 0.15,
    "critics": 0.15,
    "midi_quality": 0.20,
}


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _try_parse_json_dict(candidate: Any) -> Optional[Dict[str, Any]]:
    if isinstance(candidate, ScorePlanV1):
        return candidate.to_schema_dict()
    if isinstance(candidate, Mapping):
        return dict(candidate)
    if not isinstance(candidate, str):
        return None
    text = candidate.strip()
    if not text:
        return None
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass(frozen=True)
class RewardBreakdown:
    """Detailed breakdown of deterministic GRPO verifier rewards."""

    total_reward: float
    schema_reward: float
    genre_rules_reward: float
    instrument_range_reward: float
    critics_reward: float
    midi_quality_reward: float
    weights: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_reward": self.total_reward,
            "schema_reward": self.schema_reward,
            "genre_rules_reward": self.genre_rules_reward,
            "instrument_range_reward": self.instrument_range_reward,
            "critics_reward": self.critics_reward,
            "midi_quality_reward": self.midi_quality_reward,
            "weights": dict(self.weights),
            "diagnostics": dict(self.diagnostics),
        }


def schema_reward(completion: Union[str, Mapping[str, Any], ScorePlanV1]) -> float:
    """Compute shaped schema validity reward in ``[0.0, 1.0]``.

    - ``0.0``: unparseable / non-object output
    - ``0.15 .. 0.50``: valid JSON object with partial required keys
    - ``1.0``: strict ``ScorePlanV1`` Pydantic validation passes with zero errors
    """
    if isinstance(completion, ScorePlanV1):
        return 1.0
    parsed = _try_parse_json_dict(completion)
    if parsed is None:
        return 0.0

    try:
        ScorePlanV1.model_validate(parsed)
        return 1.0
    except ValidationError:
        present = sum(1 for key in _REQUIRED_SCORE_PLAN_KEYS if key in parsed)
        fraction = present / float(len(_REQUIRED_SCORE_PLAN_KEYS))
        return round(_clamp01(0.15 + 0.35 * fraction), 6)


def genre_rules_reward(
    plan_or_elements: Union[str, Mapping[str, Any], ScorePlanV1, Sequence[str]],
    *,
    genre: Optional[str] = None,
    drum_elements: Optional[Sequence[str]] = None,
    patterns_used: Optional[Sequence[str]] = None,
    engine: Optional[GenreRulesEngine] = None,
) -> float:
    """Compute deterministic genre-rules conformance reward in ``[0.0, 1.0]``.

    Accepts a ``ScorePlanV1`` / JSON string / dict (extracting ``genre``,
    melodic instruments, and drum elements from ``tracks``) or an explicit
    sequence of instrument element names.
    """
    rules = engine or get_genre_rules()
    resolved_genre = (genre or "").strip().lower()
    instruments: List[str] = []
    drums: List[str] = list(drum_elements or [])

    if isinstance(plan_or_elements, (list, tuple, set)):
        instruments = [str(x).strip().lower() for x in plan_or_elements if str(x).strip()]
    else:
        parsed = _try_parse_json_dict(plan_or_elements)
        if parsed is None:
            return 0.0
        if not resolved_genre:
            resolved_genre = str(parsed.get("genre") or "").strip().lower()
        for track in parsed.get("tracks") or []:
            if not isinstance(track, Mapping):
                continue
            role = str(track.get("role") or "").strip().lower()
            inst = str(track.get("instrument") or "").strip().lower()
            if not inst:
                continue
            if role == "drums":
                drums.append(inst)
            else:
                instruments.append(inst)

    if not resolved_genre:
        return 0.5

    result = rules.validate_elements(
        resolved_genre,
        instruments=instruments,
        drum_elements=drums,
        patterns_used=list(patterns_used) if patterns_used is not None else None,
    )

    score = 1.0
    for violation in result.violations:
        if violation.severity == RuleSeverity.ERROR:
            score -= 0.40
        elif violation.severity == RuleSeverity.WARNING:
            score -= 0.10
        else:
            score -= 0.03
    return round(_clamp01(score), 6)


def _extract_pitch_list(notes: Sequence[Any]) -> List[int]:
    pitches: List[int] = []
    for item in notes:
        if isinstance(item, int) and not isinstance(item, bool):
            pitches.append(item)
        elif isinstance(item, (tuple, list)) and len(item) >= 3:
            pitches.append(int(item[2]))
        elif isinstance(item, Mapping) and "pitch" in item:
            pitches.append(int(item["pitch"]))
    return pitches


def instrument_range_reward(
    notes_or_tracks: Union[Sequence[Any], Mapping[str, Sequence[Any]]],
    *,
    instrument: str = "piano",
) -> float:
    """Compute playable-range & sweet-spot compliance reward in ``[0.0, 1.0]``.

    - Playable-range notes earn full credit (0.75 weight) + sweet-spot bonus (0.25 weight).
    - Accepts either a single note sequence + ``instrument`` or a mapping of
      ``{instrument_name: notes}``.
    """
    if isinstance(notes_or_tracks, Mapping):
        track_scores: List[float] = []
        for inst_name, track_notes in notes_or_tracks.items():
            pitches = _extract_pitch_list(track_notes)
            if not pitches:
                continue
            track_scores.append(instrument_range_reward(pitches, instrument=str(inst_name)))
        if not track_scores:
            return 0.0
        return round(_clamp01(sum(track_scores) / float(len(track_scores))), 6)

    pitches = _extract_pitch_list(notes_or_tracks)
    if not pitches:
        return 0.0

    inst_range = get_range(instrument)
    in_playable = sum(1 for p in pitches if inst_range.low <= p <= inst_range.high)
    in_sweet = sum(1 for p in pitches if inst_range.sweet_low <= p <= inst_range.sweet_high)
    playable_ratio = in_playable / float(len(pitches))
    sweet_ratio = in_sweet / float(len(pitches))
    return round(_clamp01(0.75 * playable_ratio + 0.25 * sweet_ratio), 6)


def critics_reward(
    *,
    midi_path: Optional[Union[str, Path, Any]] = None,
    voicings: Optional[Sequence[Sequence[int]]] = None,
    bass_notes: Optional[Sequence[Dict[str, int]]] = None,
    kick_ticks: Optional[Sequence[int]] = None,
    tolerance_ticks: int = 60,
    tension_curve: Optional[Sequence[float]] = None,
    note_density_per_beat: Optional[Sequence[float]] = None,
    time_signature: Tuple[int, int] = (4, 4),
) -> float:
    """Compute bounded ``[0.0, 1.0]`` reward from VLC, BKAS, and ADC critics."""
    if midi_path is not None:
        features = extract_critic_features_from_midi(
            midi_path,
            time_signature=time_signature,
            tension_curve=list(tension_curve) if tension_curve is not None else None,
        )
        voicings = features.voicings if voicings is None else voicings
        bass_notes = features.bass_notes if bass_notes is None else bass_notes
        kick_ticks = features.kick_ticks if kick_ticks is None else kick_ticks
        tension_curve = features.tension_curve if tension_curve is None else tension_curve
        note_density_per_beat = (
            features.note_density_per_beat
            if note_density_per_beat is None
            else note_density_per_beat
        )

    report = run_all_critics(
        voicings=voicings,
        bass_notes=bass_notes,
        kick_ticks=kick_ticks,
        tolerance_ticks=tolerance_ticks,
        tension_curve=tension_curve,
        note_density_per_beat=note_density_per_beat,
    )

    metric_map = {m.name: m for m in report.metrics}

    # VLC: 0.0 -> 1.0, <= 1.5 -> 1.0, smoothly decays above 1.5 (3.0 -> ~0.7, 9.0 -> 0.0)
    vlc_val = metric_map["VLC"].value
    if vlc_val <= 1.5:
        vlc_score = 1.0
    else:
        vlc_score = _clamp01(1.0 - (vlc_val - 1.5) / 7.5)

    # BKAS: already in [0.0, 1.0]
    bkas_score = _clamp01(metric_map["BKAS"].value)

    # ADC: Pearson r in [-1.0, 1.0] -> map [-1, 1] smoothly, prioritizing positive correlation
    adc_val = metric_map["ADC"].value
    adc_score = _clamp01(0.5 * (adc_val + 1.0))

    return round(_clamp01((vlc_score + bkas_score + adc_score) / 3.0), 6)


def midi_quality_reward(
    notes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    *,
    midi_path: Optional[Union[str, Path]] = None,
    genre: str = "pop",
    key: str = "C",
    scale: str = "major",
    tempo: float = 120.0,
    ticks_per_beat: int = 480,
    instrument: str = "piano",
    reference_notes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    tension_curve_data: Optional[Sequence[float]] = None,
    strict_mode: bool = False,
) -> float:
    """Compute bounded ``[0.0, 1.0]`` overall MIDI quality reward via ``QualityValidator``."""
    resolved_notes: List[Tuple[int, int, int, int]]
    resolved_ref: Optional[List[Tuple[int, int, int, int]]] = (
        list(reference_notes) if reference_notes is not None else None
    )
    resolved_tpb = ticks_per_beat

    if notes is not None:
        resolved_notes = list(notes)
    elif midi_path is not None:
        resolved_tpb, melodic_notes, drum_notes = extract_note_tuples_from_midi(midi_path)
        resolved_notes = melodic_notes if melodic_notes else drum_notes
        if resolved_ref is None and drum_notes:
            resolved_ref = drum_notes
    else:
        return 0.0

    if not resolved_notes:
        return 0.0

    validator = QualityValidator(strict_mode=strict_mode)
    report = validator.validate(
        resolved_notes,
        genre=genre,
        key=key,
        scale=scale,
        tempo=int(round(tempo)),
        ticks_per_beat=resolved_tpb,
        instrument=instrument,
        reference_notes=resolved_ref,
        tension_curve_data=list(tension_curve_data) if tension_curve_data is not None else None,
    )
    return round(_clamp01(report.overall_score), 6)


def composite_grpo_reward(
    completion: Union[str, Mapping[str, Any], ScorePlanV1],
    *,
    notes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    midi_path: Optional[Union[str, Path, Any]] = None,
    genre: Optional[str] = None,
    key: Optional[str] = None,
    scale: Optional[str] = None,
    tempo: Optional[float] = None,
    instrument: str = "piano",
    weights: Optional[Mapping[str, float]] = None,
) -> RewardBreakdown:
    """Compute composite deterministic GRPO reward in ``[0.0, 1.0]``.

    When neither ``notes`` nor ``midi_path`` is supplied (text-only ScorePlan
    GRPO rollout), only the plan-level verifiers (``schema``, ``genre_rules``)
    contribute to the weighted denominator so valid plans can still reach ``1.0``.
    """
    parsed = _try_parse_json_dict(completion)
    effective_genre = (genre or (parsed.get("genre") if parsed else None) or "pop")
    effective_key = str(key or (parsed.get("key") if parsed else None) or "C")
    effective_scale = str(scale or (parsed.get("mode") if parsed else None) or "minor")
    raw_tempo = tempo if tempo is not None else (parsed.get("bpm") if parsed else 120.0)
    try:
        effective_tempo = float(raw_tempo) if raw_tempo is not None and not isinstance(raw_tempo, bool) else 120.0
    except (TypeError, ValueError):
        effective_tempo = 120.0

    tension_curve = parsed.get("tension_curve") if isinstance(parsed, dict) else None
    if not isinstance(tension_curve, list):
        tension_curve = None

    r_schema = schema_reward(completion)
    r_genre = genre_rules_reward(completion, genre=str(effective_genre))

    has_midi_evidence = notes is not None or midi_path is not None
    if has_midi_evidence:
        if notes is not None:
            melodic_notes = list(notes)
            r_range = instrument_range_reward(melodic_notes, instrument=instrument)
            r_critics = critics_reward(
                midi_path=midi_path,
                tension_curve=tension_curve,
            ) if midi_path is not None else 0.5
            r_quality = midi_quality_reward(
                melodic_notes,
                genre=str(effective_genre),
                key=effective_key,
                scale=effective_scale,
                tempo=effective_tempo,
                instrument=instrument,
                tension_curve_data=tension_curve,
            )
        else:
            assert midi_path is not None
            tpb, melodic_notes, drum_notes = extract_note_tuples_from_midi(midi_path)
            eval_notes = melodic_notes if melodic_notes else drum_notes
            r_range = instrument_range_reward(eval_notes, instrument=instrument)
            r_critics = critics_reward(midi_path=midi_path, tension_curve=tension_curve)
            r_quality = midi_quality_reward(
                eval_notes,
                genre=str(effective_genre),
                key=effective_key,
                scale=effective_scale,
                tempo=effective_tempo,
                ticks_per_beat=tpb,
                instrument=instrument,
                reference_notes=drum_notes or None,
                tension_curve_data=tension_curve,
            )
    else:
        r_range = 1.0
        r_critics = 1.0
        r_quality = 1.0

    base_weights = dict(DEFAULT_REWARD_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k in base_weights and v >= 0.0:
                base_weights[k] = float(v)

    active_keys = (
        ("schema", "genre_rules", "instrument_range", "critics", "midi_quality")
        if has_midi_evidence
        else ("schema", "genre_rules")
    )
    scores = {
        "schema": r_schema,
        "genre_rules": r_genre,
        "instrument_range": r_range,
        "critics": r_critics,
        "midi_quality": r_quality,
    }
    denom = sum(base_weights[k] for k in active_keys)
    if denom <= 0.0:
        total = 0.0
    else:
        total = sum(base_weights[k] * scores[k] for k in active_keys) / denom

    return RewardBreakdown(
        total_reward=round(_clamp01(total), 6),
        schema_reward=r_schema,
        genre_rules_reward=r_genre,
        instrument_range_reward=r_range,
        critics_reward=r_critics,
        midi_quality_reward=r_quality,
        weights={k: base_weights[k] for k in active_keys},
        diagnostics={
            "has_midi_evidence": has_midi_evidence,
            "genre": str(effective_genre),
            "key": effective_key,
            "scale": effective_scale,
            "tempo": effective_tempo,
        },
    )


def grpo_batch_reward_func(
    completions: Sequence[Any],
    *,
    genres: Optional[Sequence[Optional[str]]] = None,
    weights: Optional[Mapping[str, float]] = None,
    **_kwargs: Any,
) -> List[float]:
    """Batch reward callable compatible with TRL / Unsloth ``GRPOTrainer``.

    Accepts either plain text/dict completions or conversational list-of-message
    completions (``[{"role": "assistant", "content": "..."}]``) and returns a
    list of deterministic floats strictly in ``[0.0, 1.0]``.
    """
    rewards: List[float] = []
    for idx, completion in enumerate(completions):
        raw_candidate: Any = completion
        if isinstance(raw_candidate, list) and raw_candidate and isinstance(raw_candidate[-1], Mapping):
            raw_candidate = raw_candidate[-1].get("content", "")
        candidate: Union[str, Mapping[str, Any], ScorePlanV1] = (
            raw_candidate
            if isinstance(raw_candidate, (str, Mapping, ScorePlanV1))
            else ""
        )
        genre_hint = genres[idx] if (genres is not None and idx < len(genres)) else None
        breakdown = composite_grpo_reward(candidate, genre=genre_hint, weights=weights)
        rewards.append(breakdown.total_reward)
    return rewards
