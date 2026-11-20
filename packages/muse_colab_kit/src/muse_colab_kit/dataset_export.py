"""Local CLI dataset exporter for MUSE -> Colab JSONL shards.

IMPORTANT BOUNDARY RULE:
- Importing this module NEVER imports ``multimodal_gen`` or any heavy audio/DSP
  packages so ``muse_colab_kit`` stays 100% pure-Python in Google Colab.
- ``multimodal_gen`` is lazy-imported ONLY inside ``export_prompt_record()``
  when running locally inside a MUSE workspace to generate validated
  ``(prompt [+ optional 16 kHz mono WAV]) -> score_plan / MIDI / patches``
  JSONL shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import sys
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .schemas.dataset_record import (
    DATASET_SCHEMA_VERSION,
    DatasetRecordV1,
    MidiArtifactV1,
    MidiNoteEventV1,
    VerifierScoreSnapshotV1,
)
from .schemas.instrument_patch import InstrumentPatchV1
from .schemas.neural_io import AudioConditioningRef
from .schemas.score_plan import ScorePlanV1
from .verifiers.midi_features import extract_note_events_from_midi
from .verifiers.rewards import composite_grpo_reward


_VALID_SCORE_PLAN_MODES = frozenset(
    {"major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "locrian"}
)


@dataclass(frozen=True)
class PromptSpec:
    """Single prompt entry loaded from a JSON or JSONL manifest."""

    record_id: str
    prompt: str
    genre_hint: Optional[str] = None
    duration_bars: int = 16
    seed: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class DatasetExportConfig:
    """Configuration for local MUSE -> Colab shard export."""

    output_dir: Path
    base_seed: int = 42
    default_duration_bars: int = 16
    shard_size: int = 250
    eval_ratio: float = 0.1
    min_reward: float = 0.0
    max_events_inline: int = 2048
    write_midi_files: bool = True
    include_audio_16k: bool = False
    soundfont_path: Optional[str] = None
    muse_root: Optional[Path] = None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prompt_specs(
    prompts_path: Union[str, Path],
    *,
    default_duration_bars: int = 16,
    limit: Optional[int] = None,
) -> List[PromptSpec]:
    """Load prompt specs from a JSON array (e.g. ``golden_prompts.json``) or JSONL file."""
    path = Path(prompts_path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    raw_items: List[Any]
    if path.suffix.lower() == ".jsonl":
        raw_items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            raw_items = parsed
        elif isinstance(parsed, dict) and isinstance(parsed.get("prompts"), list):
            raw_items = parsed["prompts"]
        else:
            raise ValueError(f"Unsupported prompt manifest format in {path}")

    specs: List[PromptSpec] = []
    for idx, item in enumerate(raw_items):
        if limit is not None and len(specs) >= limit:
            break
        if isinstance(item, str):
            specs.append(
                PromptSpec(
                    record_id=f"prompt_{idx:05d}",
                    prompt=item.strip(),
                    duration_bars=default_duration_bars,
                )
            )
            continue
        if not isinstance(item, dict):
            continue
        prompt_text = str(item.get("prompt") or "").strip()
        if not prompt_text:
            continue
        rec_id = str(item.get("id") or item.get("record_id") or f"prompt_{idx:05d}").strip()
        genre_hint = item.get("genre") or item.get("expected_genre")
        bars = int(item.get("duration_bars") or item.get("expected_target_bars") or default_duration_bars)
        seed_val = int(item["seed"]) if item.get("seed") is not None else None
        extra_meta = {
            k: v
            for k, v in item.items()
            if k not in ("id", "record_id", "prompt", "genre", "duration_bars", "seed")
        }
        specs.append(
            PromptSpec(
                record_id=rec_id,
                prompt=prompt_text,
                genre_hint=str(genre_hint).strip().lower() if genre_hint else None,
                duration_bars=max(4, bars),
                seed=seed_val,
                metadata=extra_meta or None,
            )
        )
    return specs


def resample_wav_to_16k_mono(
    src_wav: Union[str, Path],
    dst_wav: Union[str, Path],
) -> Tuple[float, str]:
    """Pure-Python stdlib WAV -> 16 kHz mono PCM-16 converter returning ``(duration_sec, sha256)``."""
    src_path = Path(src_wav)
    dst_path = Path(dst_wav)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(src_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        src_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if n_frames == 0 or src_rate <= 0:
        raise ValueError(f"Empty or invalid WAV file: {src_path}")

    # Decode frames to float mono in [-1.0, 1.0]
    mono: List[float] = []
    if sampwidth == 2:
        total_samples = n_frames * n_channels
        ints = struct.unpack(f"<{total_samples}h", raw[: total_samples * 2])
        if n_channels == 1:
            mono = [s / 32768.0 for s in ints]
        else:
            for i in range(0, total_samples, n_channels):
                mono.append(sum(ints[i : i + n_channels]) / (32768.0 * n_channels))
    elif sampwidth == 1:
        total_samples = n_frames * n_channels
        vals = raw[:total_samples]
        for i in range(0, total_samples, n_channels):
            avg = sum((b - 128) / 128.0 for b in vals[i : i + n_channels]) / n_channels
            mono.append(avg)
    elif sampwidth == 4:
        total_samples = n_frames * n_channels
        ints = struct.unpack(f"<{total_samples}i", raw[: total_samples * 4])
        for i in range(0, total_samples, n_channels):
            mono.append(sum(ints[i : i + n_channels]) / (2147483648.0 * n_channels))
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    target_rate = 16000
    if src_rate == target_rate:
        resampled = mono
    else:
        target_len = max(1, int(round(len(mono) * target_rate / float(src_rate))))
        ratio = (len(mono) - 1) / float(max(1, target_len - 1)) if target_len > 1 else 0.0
        resampled = []
        last_idx = len(mono) - 1
        for i in range(target_len):
            pos = i * ratio
            idx0 = int(pos)
            frac = pos - idx0
            idx1 = min(last_idx, idx0 + 1)
            resampled.append(mono[idx0] * (1.0 - frac) + mono[idx1] * frac)

    pcm16 = [max(-32768, min(32767, int(round(sample * 32767.0)))) for sample in resampled]
    with wave.open(str(dst_path), "wb") as out_wf:
        out_wf.setnchannels(1)
        out_wf.setsampwidth(2)
        out_wf.setframerate(target_rate)
        out_wf.writeframes(struct.pack(f"<{len(pcm16)}h", *pcm16))

    duration_sec = len(pcm16) / float(target_rate)
    return duration_sec, _sha256_file(dst_path)


def _ensure_local_muse_on_sys_path(muse_root: Optional[Path] = None) -> None:
    """Add local MUSE repo root to ``sys.path`` if present (no-op in Colab)."""
    candidates: List[Path] = []
    if muse_root is not None:
        candidates.append(Path(muse_root).resolve())
    candidates.append(Path.cwd().resolve())
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "multimodal_gen" / "__init__.py").is_file():
            candidates.append(parent)
            break

    for candidate in candidates:
        if (candidate / "multimodal_gen" / "__init__.py").is_file():
            c_str = str(candidate)
            if c_str not in sys.path:
                sys.path.insert(0, c_str)
            return


def _import_local_muse_pipeline(muse_root: Optional[Path] = None) -> Dict[str, Any]:
    """Lazy-import MUSE generation modules ONLY during local dataset export."""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    _ensure_local_muse_on_sys_path(muse_root)
    try:
        from multimodal_gen.arranger import Arrangement
        from multimodal_gen.instrument_patch import build_track_scoped_instrument_patches
        from multimodal_gen.intelligence.auto_producer import auto_score_plan_v1
        from multimodal_gen.midi_generator import MidiGenerator
        from multimodal_gen.prompt_parser import PromptParser
        from multimodal_gen.score_plan_adapter import (
            build_sections_from_plan,
            score_plan_to_parsed_prompt,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Local MUSE modules ('multimodal_gen') are required to run dataset export. "
            "Run 'muse-colab-export' inside the MUSE repository environment (or pass --muse-root) "
            "and upload the resulting JSONL shards to Google Colab."
        ) from exc

    return {
        "Arrangement": Arrangement,
        "MidiGenerator": MidiGenerator,
        "PromptParser": PromptParser,
        "auto_score_plan_v1": auto_score_plan_v1,
        "build_sections_from_plan": build_sections_from_plan,
        "build_track_scoped_instrument_patches": build_track_scoped_instrument_patches,
        "score_plan_to_parsed_prompt": score_plan_to_parsed_prompt,
    }


def export_prompt_record(
    spec: PromptSpec,
    *,
    config: DatasetExportConfig,
    split: str = "train",
    seed: Optional[int] = None,
) -> DatasetRecordV1:
    """Generate and validate a single ``DatasetRecordV1`` using local MUSE."""
    muse = _import_local_muse_pipeline(config.muse_root)
    effective_seed = int(seed if seed is not None else (spec.seed if spec.seed is not None else config.base_seed))
    random.seed(effective_seed)

    # 1) Build ScorePlan v1 deterministically and preserve exact mode when supported
    plan_dict: Dict[str, Any] = muse["auto_score_plan_v1"](
        spec.prompt,
        seed=effective_seed,
        duration_bars=spec.duration_bars,
        genre_hint=spec.genre_hint,
    )
    parser = muse["PromptParser"]()
    raw_parsed = parser.parse(spec.prompt)
    raw_mode = getattr(getattr(raw_parsed, "scale_type", None), "name", "").lower()
    if raw_mode in _VALID_SCORE_PLAN_MODES:
        plan_dict["mode"] = raw_mode

    score_plan = ScorePlanV1.model_validate(plan_dict)
    schema_plan_dict = score_plan.to_schema_dict()

    # 2) Adapt ScorePlan -> ParsedPrompt + Arrangement -> MIDI
    parsed = muse["score_plan_to_parsed_prompt"](schema_plan_dict, parser=parser)
    time_sig = score_plan.time_signature or (4, 4)
    sections = muse["build_sections_from_plan"](schema_plan_dict, time_signature=time_sig)
    total_bars = sum(int(s.bars) for s in sections)
    total_ticks = int(sections[-1].end_tick) if sections else 0
    arrangement = muse["Arrangement"](
        sections=sections,
        total_bars=total_bars,
        total_ticks=total_ticks,
        bpm=float(score_plan.bpm),
        time_signature=time_sig,
    )

    random.seed(effective_seed)
    midi_gen = muse["MidiGenerator"]()
    midi_file = midi_gen.generate(arrangement, parsed)

    midi_dir = config.output_dir / "artifacts" / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in spec.record_id)
    midi_path = midi_dir / f"{safe_id}.mid"
    midi_file.save(str(midi_path))

    midi_sha256 = _sha256_file(midi_path)
    tpb, note_events = extract_note_events_from_midi(midi_path)
    track_names = [str(getattr(tr, "name", "") or f"track_{i}") for i, tr in enumerate(midi_file.tracks)]
    inline_events = [
        MidiNoteEventV1(
            start_tick=int(st),
            duration_ticks=int(dur),
            pitch=int(pitch),
            velocity=int(vel),
            channel=int(ch),
            track=str(tname),
        )
        for (st, dur, pitch, vel, ch, tname) in note_events[: config.max_events_inline]
    ]

    # 3) Build strict InstrumentPatchV1 metadata
    instrument_names = [t.instrument for t in score_plan.tracks if t.role != "drums"]
    if not instrument_names:
        instrument_names = list(getattr(parsed, "instruments", []) or [])
    raw_patches = muse["build_track_scoped_instrument_patches"](
        instrument_names,
        genre=score_plan.genre or parsed.genre,
    )
    patches = [InstrumentPatchV1.model_validate(p.to_dict()) for p in raw_patches]

    # 4) Optional local 16 kHz mono WAV clip (OFF by default)
    audio_ref: Optional[AudioConditioningRef] = None
    if config.include_audio_16k:
        from multimodal_gen.audio_renderer import AudioRenderer

        audio_dir = config.output_dir / "artifacts" / "audio_16k"
        audio_dir.mkdir(parents=True, exist_ok=True)
        raw_wav = audio_dir / f"{safe_id}_raw.wav"
        wav_16k = audio_dir / f"{safe_id}_16k.wav"
        renderer = AudioRenderer(
            sample_rate=44100,
            use_fluidsynth=bool(config.soundfont_path),
            soundfont_path=config.soundfont_path,
            genre=score_plan.genre or parsed.genre or "pop",
            use_bwf=False,
        )
        ok = renderer.render_midi_file(str(midi_path), str(raw_wav), parsed)
        if not ok or not raw_wav.is_file():
            raise RuntimeError(f"AudioRenderer failed to render WAV for record '{spec.record_id}'")
        duration_sec, audio_sha = resample_wav_to_16k_mono(raw_wav, wav_16k)
        try:
            raw_wav.unlink()
        except OSError:
            pass
        audio_ref = AudioConditioningRef(
            path=str(wav_16k.relative_to(config.output_dir).as_posix()),
            sha256=audio_sha,
            sample_rate_hz=16000,
            channels=1,
            encoding="pcm_s16le",
            duration_seconds=round(duration_sec, 4),
        )

    # 5) Run deterministic GRPO verifiers over (ScorePlan + MIDI)
    breakdown = composite_grpo_reward(
        score_plan,
        midi_path=midi_path,
        genre=score_plan.genre or parsed.genre,
        key=score_plan.key,
        scale=score_plan.mode,
        tempo=score_plan.bpm,
    )

    rel_midi_path: Optional[str] = str(midi_path.relative_to(config.output_dir).as_posix())
    if not config.write_midi_files:
        try:
            midi_path.unlink()
        except OSError:
            pass
        rel_midi_path = None

    split_literal = "eval" if split == "eval" else "train"
    return DatasetRecordV1(
        record_id=spec.record_id,
        split=split_literal,
        seed=effective_seed,
        prompt=spec.prompt,
        genre=score_plan.genre or parsed.genre or "pop",
        audio_clip_16k=audio_ref,
        score_plan=score_plan,
        instrument_patches=patches,
        midi=MidiArtifactV1(
            relative_path=rel_midi_path,
            sha256=midi_sha256,
            ticks_per_beat=tpb,
            track_names=track_names,
            note_count=len(note_events),
            events=inline_events,
        ),
        verifier_scores=VerifierScoreSnapshotV1(
            total_reward=breakdown.total_reward,
            schema_reward=breakdown.schema_reward,
            genre_rules_reward=breakdown.genre_rules_reward,
            instrument_range_reward=breakdown.instrument_range_reward,
            critics_reward=breakdown.critics_reward,
            midi_quality_reward=breakdown.midi_quality_reward,
        ),
        metadata=dict(spec.metadata or {}),
    )


def write_sharded_jsonl(
    records: Sequence[DatasetRecordV1],
    *,
    output_dir: Path,
    shard_size: int = 250,
) -> Dict[str, Any]:
    """Write validated ``DatasetRecordV1`` rows into deterministic train/eval JSONL shards."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_shard_size = max(1, int(shard_size))

    by_split: Dict[str, List[DatasetRecordV1]] = {"train": [], "eval": []}
    for rec in records:
        by_split[rec.split].append(rec)

    shards_meta: List[Dict[str, Any]] = []
    for split_name in ("train", "eval"):
        split_records = by_split[split_name]
        for shard_idx in range(0, len(split_records), effective_shard_size):
            chunk = split_records[shard_idx : shard_idx + effective_shard_size]
            shard_num = shard_idx // effective_shard_size
            shard_file = out_dir / f"{split_name}-{shard_num:05d}.jsonl"
            lines = [r.to_jsonl_line() for r in chunk]
            shard_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            shards_meta.append(
                {
                    "split": split_name,
                    "path": shard_file.name,
                    "records": len(chunk),
                    "sha256": _sha256_file(shard_file),
                }
            )

    mean_reward = (
        round(sum(r.verifier_scores.total_reward for r in records) / float(len(records)), 6)
        if records
        else 0.0
    )
    manifest: Dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_records": len(records),
        "train_records": len(by_split["train"]),
        "eval_records": len(by_split["eval"]),
        "mean_total_reward": mean_reward,
        "shards": shards_meta,
    }
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def export_dataset(
    prompts: Sequence[PromptSpec],
    *,
    config: DatasetExportConfig,
) -> Dict[str, Any]:
    """Export a list of ``PromptSpec`` items into validated JSONL shards + manifest."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    total = len(prompts)
    eval_count = int(round(total * config.eval_ratio)) if total > 1 and config.eval_ratio > 0 else 0
    eval_start = total - eval_count

    records: List[DatasetRecordV1] = []
    filtered_out = 0
    for idx, spec in enumerate(prompts):
        split = "eval" if idx >= eval_start else "train"
        rec_seed = spec.seed if spec.seed is not None else (config.base_seed + idx)
        record = export_prompt_record(spec, config=config, split=split, seed=rec_seed)
        if record.verifier_scores.total_reward < config.min_reward:
            filtered_out += 1
            continue
        records.append(record)

    manifest = write_sharded_jsonl(
        records,
        output_dir=config.output_dir,
        shard_size=config.shard_size,
    )
    manifest["filtered_below_min_reward"] = filtered_out
    manifest["include_audio_16k"] = config.include_audio_16k
    (config.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="muse-colab-export",
        description="Export validated MUSE (prompt -> score_plan / MIDI / patch) JSONL shards for Colab.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to prompt manifest (.json array or .jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Destination directory for JSONL shards, artifacts, and dataset_manifest.json.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base deterministic RNG seed.")
    parser.add_argument("--duration-bars", type=int, default=16, help="Default arrangement bars.")
    parser.add_argument("--shard-size", type=int, default=250, help="Max records per JSONL shard.")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Fraction assigned to eval split.")
    parser.add_argument("--min-reward", type=float, default=0.0, help="Drop records below this reward.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on prompt count.")
    parser.add_argument(
        "--no-midi-files",
        action="store_true",
        help="Do not retain exported .mid files on disk (keep inline MIDI events only).",
    )
    parser.add_argument(
        "--include-audio-16k",
        action="store_true",
        help="Opt-in: render local WAV audio and export 16 kHz mono PCM-16 clips (default: False).",
    )
    parser.add_argument(
        "--soundfont",
        type=str,
        default=None,
        help="Optional SoundFont (.sf2/.sf3) path when --include-audio-16k is enabled.",
    )
    parser.add_argument(
        "--muse-root",
        type=str,
        default=None,
        help="Optional explicit path to the local MUSE repository root.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    specs = load_prompt_specs(
        args.prompts,
        default_duration_bars=args.duration_bars,
        limit=args.limit,
    )
    cfg = DatasetExportConfig(
        output_dir=Path(args.output_dir),
        base_seed=int(args.seed),
        default_duration_bars=int(args.duration_bars),
        shard_size=int(args.shard_size),
        eval_ratio=float(args.eval_ratio),
        min_reward=float(args.min_reward),
        write_midi_files=not bool(args.no_midi_files),
        include_audio_16k=bool(args.include_audio_16k),
        soundfont_path=args.soundfont,
        muse_root=Path(args.muse_root) if args.muse_root else None,
    )
    manifest = export_dataset(specs, config=cfg)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
