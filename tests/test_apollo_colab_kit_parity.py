"""Cross-repo parity tests verifying apollo_colab_kit matches MUSE bit-for-bit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mido
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
COLAB_KIT_SRC = PROJECT_ROOT / "packages" / "apollo_colab_kit" / "src"
if str(COLAB_KIT_SRC) not in sys.path:
    sys.path.insert(0, str(COLAB_KIT_SRC))

from apollo_colab_kit.dataset_export import (
    DatasetExportConfig,
    export_dataset,
    export_eval_golden,
    export_sweep_dataset,
    load_prompt_specs,
)
from apollo_colab_kit.prompt_sweep import normalize_prompt_text
from apollo_colab_kit.schemas import (
    DatasetRecordV1,
    load_arrangement_template_yaml,
    load_bundled_schema,
)
from apollo_colab_kit.verifiers import (
    GenreRulesEngine as KitGenreRulesEngine,
    INSTRUMENT_RANGES as KIT_INSTRUMENT_RANGES,
    QualityValidator as KitQualityValidator,
    compute_adc as kit_compute_adc,
    compute_bkas as kit_compute_bkas,
    compute_vlc as kit_compute_vlc,
    extract_critic_features_from_midi as kit_extract_features,
    get_timpani_pitches as kit_get_timpani_pitches,
)

from multimodal_gen.genre_rules import GenreRulesEngine as MuseGenreRulesEngine
from multimodal_gen.instrument_ranges import (
    INSTRUMENT_RANGES as MUSE_INSTRUMENT_RANGES,
    get_timpani_pitches as muse_get_timpani_pitches,
)
from multimodal_gen.intelligence.critics import (
    compute_adc as muse_compute_adc,
    compute_bkas as muse_compute_bkas,
    compute_vlc as muse_compute_vlc,
)
from multimodal_gen.intelligence.midi_critic_features import (
    extract_critic_features_from_midi as muse_extract_features,
)
from multimodal_gen.quality_validator import QualityValidator as MuseQualityValidator


def test_bundled_schemas_match_canonical_muse_schemas_byte_for_byte():
    pairs = [
        ("score_plan.v1.schema.json", PROJECT_ROOT / "docs" / "muse-specs" / "schemas" / "score_plan.v1.schema.json"),
        ("arrangement_template.schema.json", PROJECT_ROOT / "configs" / "schemas" / "arrangement_template.schema.json"),
        ("neural_runtime.v1.schema.json", PROJECT_ROOT / "docs" / "muse-specs" / "schemas" / "neural_runtime.v1.schema.json"),
    ]
    for bundled_name, repo_path in pairs:
        assert repo_path.is_file()
        assert load_bundled_schema(bundled_name) == json.loads(repo_path.read_text(encoding="utf-8"))


def test_all_12_arrangement_templates_validate_strictly():
    templates_dir = PROJECT_ROOT / "configs" / "arrangements" / "templates"
    yaml_files = sorted(templates_dir.glob("*.yaml"))
    assert len(yaml_files) >= 12
    for yaml_path in yaml_files:
        tpl = load_arrangement_template_yaml(yaml_path)
        assert tpl.total_bars >= 4


def test_verifiers_parity_with_multimodal_gen(tmp_path: Path):
    # 1. Instrument ranges & timpani parity
    assert set(KIT_INSTRUMENT_RANGES.keys()) == set(MUSE_INSTRUMENT_RANGES.keys())
    for name, muse_r in MUSE_INSTRUMENT_RANGES.items():
        kit_r = KIT_INSTRUMENT_RANGES[name]
        assert (kit_r.low, kit_r.high, kit_r.sweet_low, kit_r.sweet_high) == (
            muse_r.low,
            muse_r.high,
            muse_r.sweet_low,
            muse_r.sweet_high,
        )
    for key in ("C", "D", "F#", "Bb", "A"):
        assert kit_get_timpani_pitches(key) == muse_get_timpani_pitches(key)

    # 2. Critics + MIDI features parity
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    for bar in range(4):
        base = bar * 1920
        for idx, p in enumerate((48, 60, 64, 67)):
            tr.append(mido.Message("note_on", channel=0, note=p + (bar % 2), velocity=85 + bar * 5, time=0 if (bar == 0 and idx == 0) else (1440 if idx == 0 else 0)))
        for idx, p in enumerate((48, 60, 64, 67)):
            tr.append(mido.Message("note_off", channel=0, note=p + (bar % 2), velocity=0, time=480 if idx == 0 else 0))
    dr = mido.MidiTrack()
    for bar in range(4):
        dr.append(mido.Message("note_on", channel=9, note=36, velocity=110, time=0 if bar == 0 else 1800))
        dr.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=120))
    mid.tracks.extend([tr, dr])
    midi_path = tmp_path / "parity.mid"
    mid.save(str(midi_path))

    kf = kit_extract_features(midi_path)
    mf = muse_extract_features(str(midi_path))
    assert kf.voicings == mf.voicings
    assert kf.bass_notes == mf.bass_notes
    assert kf.kick_ticks == mf.kick_ticks
    assert kf.note_density_per_beat == mf.note_density_per_beat
    assert kf.tension_curve == mf.tension_curve

    assert kit_compute_vlc(kf.voicings).value == muse_compute_vlc(mf.voicings).value
    assert kit_compute_bkas(kf.bass_notes, kf.kick_ticks).value == muse_compute_bkas(mf.bass_notes, mf.kick_ticks).value
    assert kit_compute_adc(kf.tension_curve, kf.note_density_per_beat).value == muse_compute_adc(mf.tension_curve, mf.note_density_per_beat).value

    # 3. GenreRulesEngine parity
    k_engine = KitGenreRulesEngine()
    m_engine = MuseGenreRulesEngine()
    assert set(k_engine.get_all_genres()) == set(m_engine.get_all_genres())
    for genre in ("trap", "g_funk", "boom_bap", "lofi", "ethiopian_traditional"):
        kr = k_engine.validate_elements(genre, ["808_distorted", "synth"], ["hihat_roll", "kick"])
        mr = m_engine.validate_elements(genre, ["808_distorted", "synth"], ["hihat_roll", "kick"])
        assert kr.valid == mr.valid
        assert sorted(kr.forbidden_used) == sorted(mr.forbidden_used)
        assert sorted(kr.mandatory_missing) == sorted(mr.mandatory_missing)

    # 4. QualityValidator parity
    notes = [(i * 240, 200, 60 + (i % 7), 70 + (i % 5) * 10) for i in range(28)]
    k_rep = KitQualityValidator().validate(notes, genre="hip_hop", key="A", scale="minor", tempo=92)
    m_rep = MuseQualityValidator().validate(notes, genre="hip_hop", key="A", scale="minor", tempo=92)
    assert k_rep.overall_score == pytest.approx(m_rep.overall_score, abs=1e-9)
    assert k_rep.passed == m_rep.passed
    assert len(k_rep.metrics) == len(m_rep.metrics)


def test_local_dataset_export_end_to_end_smoke(tmp_path: Path):
    golden_path = PROJECT_ROOT / "tests" / "fixtures" / "golden_prompts.json"
    specs = load_prompt_specs(golden_path, default_duration_bars=8, limit=2)
    assert len(specs) == 2

    out_dir = tmp_path / "colab_shards"
    cfg = DatasetExportConfig(
        output_dir=out_dir,
        base_seed=42,
        default_duration_bars=8,
        shard_size=10,
        eval_ratio=0.5,
        include_audio_16k=False,
    )
    manifest = export_dataset(specs, config=cfg)

    assert manifest["total_records"] == 2
    assert manifest["train_records"] == 1
    assert manifest["eval_records"] == 1
    assert manifest["include_audio_16k"] is False
    assert (out_dir / "dataset_manifest.json").is_file()

    for shard in manifest["shards"]:
        lines = (out_dir / shard["path"]).read_text(encoding="utf-8").splitlines()
        assert len(lines) == shard["records"]
        for line in lines:
            rec = DatasetRecordV1.model_validate_json(line)
            assert rec.audio_clip_16k is None
            assert rec.midi.note_count > 0
            assert 0.0 <= rec.verifier_scores.total_reward <= 1.0
            assert rec.midi.relative_path is not None
            assert (out_dir / rec.midi.relative_path).is_file()


def test_eval_golden_and_prompt_sweep_zero_leakage_smoke(tmp_path: Path):
    golden_path = PROJECT_ROOT / "tests" / "fixtures" / "golden_prompts.json"
    out_dir = tmp_path / "apollo_bundle"
    cfg = DatasetExportConfig(
        output_dir=out_dir,
        base_seed=2026,
        min_reward=0.60,
        min_genre_reward=0.80,
        max_events_inline=0,
        write_midi_files=False,
        include_audio_16k=False,
    )

    golden_manifest = export_eval_golden(golden_path, config=cfg)
    assert golden_manifest["eval_golden"]["records"] == 22
    assert (out_dir / "eval_golden.jsonl").is_file()

    sweep_manifest = export_sweep_dataset(
        train_target=10,
        val_target=4,
        config=cfg,
        exclude_prompts_path=golden_path,
    )
    assert sweep_manifest["train"]["records"] == 10
    assert sweep_manifest["val"]["records"] == 4
    assert sweep_manifest["zero_leakage_verified"] is True

    golden_norms = {
        normalize_prompt_text(DatasetRecordV1.model_validate_json(line).prompt)
        for line in (out_dir / "eval_golden.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    train_norms = {
        normalize_prompt_text(DatasetRecordV1.model_validate_json(line).prompt)
        for line in (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    val_norms = {
        normalize_prompt_text(DatasetRecordV1.model_validate_json(line).prompt)
        for line in (out_dir / "val.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    assert train_norms.isdisjoint(golden_norms)
    assert val_norms.isdisjoint(golden_norms)
    assert train_norms.isdisjoint(val_norms)

