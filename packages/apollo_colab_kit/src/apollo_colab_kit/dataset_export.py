"""Local CLI dataset exporter for Apollo -> Colab JSONL bundles.

IMPORTANT BOUNDARY RULE:
- Importing this module NEVER imports ``multimodal_gen`` or any heavy audio/DSP
  packages so ``apollo_colab_kit`` stays 100% pure-Python in Google Colab.
- ``multimodal_gen`` is lazy-imported ONLY during local export execution when
  running inside an Apollo/MUSE workspace to generate validated
  ``(prompt [+ optional 16 kHz mono WAV]) -> score_plan / MIDI / patches``
  JSONL files for Colab SFT + GRPO training.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import struct
import sys
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .prompt_sweep import (
    SweepCandidate,
    generate_prompt_sweep,
    normalize_prompt_text,
)
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
    prebuilt_score_plan: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class DatasetExportConfig:
    """Configuration for local Apollo -> Colab dataset export."""

    output_dir: Path
    base_seed: int = 42
    default_duration_bars: int = 16
    shard_size: int = 250
    eval_ratio: float = 0.1
    min_reward: float = 0.0
    min_genre_reward: float = 0.0
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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _find_local_repo_root(muse_root: Optional[Path] = None) -> Optional[Path]:
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
            return candidate
    return None


def _ensure_local_muse_on_sys_path(muse_root: Optional[Path] = None) -> Optional[Path]:
    repo_root = _find_local_repo_root(muse_root)
    if repo_root is not None:
        c_str = str(repo_root)
        if c_str not in sys.path:
            sys.path.insert(0, c_str)
    return repo_root


def _import_local_muse_pipeline(muse_root: Optional[Path] = None) -> Dict[str, Any]:
    """Lazy-import Apollo/MUSE internal synthesis modules ONLY during local dataset export."""
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
            "Local Apollo/MUSE engine modules ('multimodal_gen') are required to run dataset export. "
            "Run 'apollo-colab-export' inside the Apollo repository environment (or pass --muse-root) "
            "and upload the resulting JSONL bundle to Google Colab."
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


def _sanitize_midi_delta_times(midi_file: Any) -> None:
    """Normalize negative/float MIDI delta ticks before serialization or feature extraction."""
    for track in getattr(midi_file, "tracks", ()):
        for msg in track:
            t = getattr(msg, "time", 0)
            if isinstance(t, (int, float)):
                if t < 0:
                    msg.time = 0
                elif isinstance(t, float):
                    msg.time = int(round(t))


def export_prompt_record(
    spec: PromptSpec,
    *,
    config: DatasetExportConfig,
    split: str = "train",
    seed: Optional[int] = None,
    _pipeline_cache: Optional[Dict[str, Any]] = None,
) -> DatasetRecordV1:
    """Generate and validate a single ``DatasetRecordV1`` using the local engine."""
    muse = _pipeline_cache if _pipeline_cache is not None else _import_local_muse_pipeline(config.muse_root)
    effective_seed = int(seed if seed is not None else (spec.seed if spec.seed is not None else config.base_seed))
    random.seed(effective_seed)

    parser = muse.get("_parser")
    if parser is None:
        parser = muse["PromptParser"]()
        muse["_parser"] = parser

    # 1) Build or validate ScorePlan v1
    if spec.prebuilt_score_plan is not None:
        plan_dict: Dict[str, Any] = dict(spec.prebuilt_score_plan)
    else:
        plan_dict = muse["auto_score_plan_v1"](
            spec.prompt,
            seed=effective_seed,
            duration_bars=spec.duration_bars,
            genre_hint=spec.genre_hint,
        )
        raw_parsed = parser.parse(spec.prompt)
        raw_mode = getattr(getattr(raw_parsed, "scale_type", None), "name", "").lower()
        if raw_mode in _VALID_SCORE_PLAN_MODES:
            plan_dict["mode"] = raw_mode

    score_plan = ScorePlanV1.model_validate(plan_dict)
    schema_plan_dict = score_plan.to_schema_dict()

    # 2) Adapt ScorePlan -> ParsedPrompt + Arrangement -> MIDI (stdout suppressed for clean JSON CLI)
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
    with contextlib.redirect_stdout(io.StringIO()):
        midi_gen = muse["MidiGenerator"]()
        midi_file = midi_gen.generate(arrangement, parsed)
    _sanitize_midi_delta_times(midi_file)

    safe_id = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in spec.record_id)
    rel_midi_path: Optional[str] = None
    midi_path_on_disk: Optional[Path] = None

    if config.write_midi_files or config.include_audio_16k:
        midi_dir = config.output_dir / "artifacts" / "midi"
        midi_dir.mkdir(parents=True, exist_ok=True)
        midi_path_on_disk = midi_dir / f"{safe_id}.mid"
        midi_file.save(str(midi_path_on_disk))
        midi_sha256 = _sha256_file(midi_path_on_disk)
        if config.write_midi_files:
            rel_midi_path = str(midi_path_on_disk.relative_to(config.output_dir).as_posix())
    else:
        buf = io.BytesIO()
        midi_file.save(file=buf)
        midi_sha256 = _sha256_bytes(buf.getvalue())

    tpb, note_events = extract_note_events_from_midi(midi_file)
    track_names = [str(getattr(tr, "name", "") or f"track_{i}") for i, tr in enumerate(midi_file.tracks)]
    inline_events = (
        [
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
        if config.max_events_inline > 0
        else []
    )

    # 3) Build strict InstrumentPatchV1 metadata
    instrument_names = [t.instrument for t in score_plan.tracks if t.role != "drums"]
    if not instrument_names:
        instrument_names = list(getattr(parsed, "instruments", []) or [])
    raw_patches = muse["build_track_scoped_instrument_patches"](
        instrument_names,
        genre=score_plan.genre or parsed.genre,
    )
    patches = [InstrumentPatchV1.model_validate(p.to_dict()) for p in raw_patches]

    # 4) Optional local 16 kHz mono WAV clip (OFF by default in v1)
    audio_ref: Optional[AudioConditioningRef] = None
    if config.include_audio_16k:
        assert midi_path_on_disk is not None
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
        ok = renderer.render_midi_file(str(midi_path_on_disk), str(raw_wav), parsed)
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
        if not config.write_midi_files:
            try:
                midi_path_on_disk.unlink()
            except OSError:
                pass

    # 5) Run deterministic GRPO verifiers over (ScorePlan + in-memory MIDI)
    breakdown = composite_grpo_reward(
        score_plan,
        midi_path=midi_file,
        genre=score_plan.genre or parsed.genre,
        key=score_plan.key,
        scale=score_plan.mode,
        tempo=score_plan.bpm,
    )

    if split == "eval":
        split_literal = "eval"
    elif split == "val":
        split_literal = "val"
    else:
        split_literal = "train"

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


def write_jsonl_file(records: Sequence[DatasetRecordV1], path: Path) -> Dict[str, Any]:
    """Write validated ``DatasetRecordV1`` rows to a single JSONL file and return metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r.to_jsonl_line() for r in records]
    content = ("\n".join(lines) + "\n") if lines else ""
    path.write_text(content, encoding="utf-8")
    mean_reward = (
        round(sum(r.verifier_scores.total_reward for r in records) / float(len(records)), 6)
        if records
        else 0.0
    )
    return {
        "path": path.name,
        "records": len(records),
        "mean_total_reward": mean_reward,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def write_sharded_jsonl(
    records: Sequence[DatasetRecordV1],
    *,
    output_dir: Path,
    shard_size: int = 250,
) -> Dict[str, Any]:
    """Write validated ``DatasetRecordV1`` rows into deterministic split JSONL shards."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_shard_size = max(1, int(shard_size))

    by_split: Dict[str, List[DatasetRecordV1]] = {"train": [], "val": [], "eval": []}
    for rec in records:
        by_split[rec.split].append(rec)

    shards_meta: List[Dict[str, Any]] = []
    for split_name in ("train", "val", "eval"):
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
        "val_records": len(by_split["val"]),
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
    pipeline_cache = _import_local_muse_pipeline(config.muse_root)
    total = len(prompts)
    eval_count = int(round(total * config.eval_ratio)) if total > 1 and config.eval_ratio > 0 else 0
    eval_start = total - eval_count

    records: List[DatasetRecordV1] = []
    filtered_out = 0
    for idx, spec in enumerate(prompts):
        split = "eval" if idx >= eval_start else "train"
        rec_seed = spec.seed if spec.seed is not None else (config.base_seed + idx)
        record = export_prompt_record(
            spec,
            config=config,
            split=split,
            seed=rec_seed,
            _pipeline_cache=pipeline_cache,
        )
        if (
            record.verifier_scores.total_reward < config.min_reward
            or record.verifier_scores.genre_rules_reward < config.min_genre_reward
        ):
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


def export_eval_golden(
    prompts_path: Union[str, Path],
    *,
    config: DatasetExportConfig,
    output_filename: str = "eval_golden.jsonl",
) -> Dict[str, Any]:
    """Export held-out golden prompts strictly to ``eval_golden.jsonl`` (never mixed into train)."""
    specs = load_prompt_specs(prompts_path, default_duration_bars=config.default_duration_bars)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_cache = _import_local_muse_pipeline(config.muse_root)

    records: List[DatasetRecordV1] = []
    for idx, spec in enumerate(specs):
        rec_seed = spec.seed if spec.seed is not None else (config.base_seed + idx)
        record = export_prompt_record(
            spec,
            config=config,
            split="eval",
            seed=rec_seed,
            _pipeline_cache=pipeline_cache,
        )
        records.append(record)

    out_path = config.output_dir / output_filename
    file_meta = write_jsonl_file(records, out_path)
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_prompts": str(Path(prompts_path)),
        "held_out_eval_only": True,
        "include_audio_16k": config.include_audio_16k,
        "eval_golden": file_meta,
    }
    (config.output_dir / "eval_golden_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def export_sweep_dataset(
    *,
    train_target: int,
    val_target: int,
    config: DatasetExportConfig,
    exclude_prompts_path: Optional[Union[str, Path]] = None,
    oversample_factor: float = 1.25,
    jaccard_threshold: float = 0.65,
) -> Dict[str, Any]:
    """Run deterministic prompt sweep -> local synthesis -> verifier filter -> ``train.jsonl`` & ``val.jsonl``.

    Guarantees strict deduplication and zero prompt leakage against *exclude_prompts_path*.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_cache = _import_local_muse_pipeline(config.muse_root)
    repo_root = _find_local_repo_root(config.muse_root)

    excluded_prompts: List[str] = []
    if exclude_prompts_path is not None:
        excluded_specs = load_prompt_specs(exclude_prompts_path)
        excluded_prompts = [s.prompt for s in excluded_specs]
    excluded_norm = {normalize_prompt_text(p) for p in excluded_prompts}

    templates_dir = (repo_root / "configs" / "arrangements" / "templates") if repo_root else None
    presets_dir = (repo_root / "configs" / "presets") if repo_root else None

    total_needed = max(0, int(train_target)) + max(0, int(val_target))
    candidate_budget = max(total_needed, int(round(total_needed * max(1.0, oversample_factor))) + 50)

    candidates: List[SweepCandidate] = generate_prompt_sweep(
        count=candidate_budget,
        seed=config.base_seed,
        exclude_prompts=excluded_prompts,
        templates_dir=templates_dir,
        presets_dir=presets_dir,
        jaccard_threshold=jaccard_threshold,
    )

    verified_records: List[DatasetRecordV1] = []
    filtered_out = 0
    errors_skipped = 0
    seen_norm: set[str] = set()

    for cand in candidates:
        if len(verified_records) >= total_needed:
            break
        norm_p = normalize_prompt_text(cand.prompt)
        if norm_p in excluded_norm or norm_p in seen_norm:
            continue

        spec = PromptSpec(
            record_id=cand.record_id,
            prompt=cand.prompt,
            genre_hint=cand.genre,
            duration_bars=int(cand.score_plan.get("duration_bars") or config.default_duration_bars),
            seed=cand.seed,
            metadata=cand.metadata,
            prebuilt_score_plan=cand.score_plan,
        )
        try:
            # Assign provisional split; final split assigned deterministically after filtering
            record = export_prompt_record(
                spec,
                config=config,
                split="train",
                seed=cand.seed,
                _pipeline_cache=pipeline_cache,
            )
        except Exception:
            errors_skipped += 1
            continue

        scores = record.verifier_scores
        if (
            scores.schema_reward < 1.0
            or scores.genre_rules_reward < config.min_genre_reward
            or scores.total_reward < config.min_reward
        ):
            filtered_out += 1
            continue

        seen_norm.add(norm_p)
        verified_records.append(record)

    # Deterministic stratified-friendly shuffle before train/val split (zero overlap)
    rng = random.Random(int(config.base_seed) + 9973)
    indices = list(range(len(verified_records)))
    rng.shuffle(indices)
    shuffled = [verified_records[i] for i in indices]

    actual_val = min(val_target, len(shuffled))
    val_raw = shuffled[:actual_val]
    train_raw = shuffled[actual_val : actual_val + train_target]

    train_records = [r.model_copy(update={"split": "train"}) for r in train_raw]
    val_records = [r.model_copy(update={"split": "val"}) for r in val_raw]

    # Paranoid leakage self-check
    train_norms = {normalize_prompt_text(r.prompt) for r in train_records}
    val_norms = {normalize_prompt_text(r.prompt) for r in val_records}
    assert train_norms.isdisjoint(val_norms), "Leakage detected between train and val splits"
    assert train_norms.isdisjoint(excluded_norm), "Leakage detected between train and excluded eval prompts"
    assert val_norms.isdisjoint(excluded_norm), "Leakage detected between val and excluded eval prompts"

    train_meta = write_jsonl_file(train_records, config.output_dir / "train.jsonl")
    val_meta = write_jsonl_file(val_records, config.output_dir / "val.jsonl")

    genre_counts: Dict[str, int] = {}
    mode_counts: Dict[str, int] = {}
    for rec in train_records + val_records:
        genre_counts[rec.genre] = genre_counts.get(rec.genre, 0) + 1
        m = rec.score_plan.mode
        mode_counts[m] = mode_counts.get(m, 0) + 1

    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": config.base_seed,
        "train": train_meta,
        "val": val_meta,
        "candidates_generated": len(candidates),
        "filtered_below_threshold": filtered_out,
        "generation_errors_skipped": errors_skipped,
        "excluded_eval_prompts": len(excluded_norm),
        "zero_leakage_verified": True,
        "thresholds": {
            "min_reward": config.min_reward,
            "min_genre_reward": config.min_genre_reward,
            "jaccard_threshold": jaccard_threshold,
        },
        "genre_distribution": dict(sorted(genre_counts.items())),
        "mode_distribution": dict(sorted(mode_counts.items())),
        "include_audio_16k": config.include_audio_16k,
    }
    (config.output_dir / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apollo-colab-export",
        description="Export validated Apollo (prompt -> score_plan / MIDI / patch) JSONL bundles for Colab.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Path to prompt manifest (.json array or .jsonl). Required unless --sweep is set.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Destination directory for JSONL files, artifacts, and manifests.",
    )
    parser.add_argument(
        "--eval-golden",
        action="store_true",
        help="Export --prompts strictly to eval_golden.jsonl (held-out benchmark, never mixed into train).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run deterministic prompt-sweep generator producing deduplicated train.jsonl and val.jsonl.",
    )
    parser.add_argument("--train-target", type=int, default=3500, help="Target train.jsonl examples in --sweep mode.")
    parser.add_argument("--val-target", type=int, default=300, help="Target val.jsonl examples in --sweep mode.")
    parser.add_argument(
        "--exclude-prompts",
        type=str,
        default=None,
        help="Path to held-out prompt manifest (e.g. golden_prompts.json) excluded from --sweep.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base deterministic RNG seed.")
    parser.add_argument("--duration-bars", type=int, default=16, help="Default arrangement bars.")
    parser.add_argument("--shard-size", type=int, default=250, help="Max records per JSONL shard (manifest mode).")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Fraction assigned to eval split (manifest mode).")
    parser.add_argument("--min-reward", type=float, default=0.0, help="Minimum composite verifier reward threshold.")
    parser.add_argument("--min-genre-reward", type=float, default=0.0, help="Minimum genre-rules verifier reward threshold.")
    parser.add_argument("--max-events-inline", type=int, default=64, help="Max inline MIDI events stored per JSONL record (0 = metadata only).")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on prompt count.")
    parser.add_argument(
        "--no-midi-files",
        action="store_true",
        help="Do not write .mid files to disk (verify MIDI in-memory and emit compact JSONL metadata).",
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
        help="Optional explicit path to the local Apollo/MUSE repository root.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = DatasetExportConfig(
        output_dir=Path(args.output_dir),
        base_seed=int(args.seed),
        default_duration_bars=int(args.duration_bars),
        shard_size=int(args.shard_size),
        eval_ratio=float(args.eval_ratio),
        min_reward=float(args.min_reward),
        min_genre_reward=float(args.min_genre_reward),
        max_events_inline=int(args.max_events_inline),
        write_midi_files=not bool(args.no_midi_files),
        include_audio_16k=bool(args.include_audio_16k),
        soundfont_path=args.soundfont,
        muse_root=Path(args.muse_root) if args.muse_root else None,
    )

    if args.sweep:
        manifest = export_sweep_dataset(
            train_target=int(args.train_target),
            val_target=int(args.val_target),
            config=cfg,
            exclude_prompts_path=args.exclude_prompts,
        )
        print(json.dumps(manifest, indent=2))
        return 0

    if not args.prompts:
        parser.error("--prompts is required unless --sweep is specified")

    if args.eval_golden:
        manifest = export_eval_golden(args.prompts, config=cfg)
        print(json.dumps(manifest, indent=2))
        return 0

    specs = load_prompt_specs(
        args.prompts,
        default_duration_bars=args.duration_bars,
        limit=args.limit,
    )
    manifest = export_dataset(specs, config=cfg)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
