"""Standalone tests for pure-Python dataset_export utilities (prompt loading, WAV 16k, sharding)."""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

from apollo_colab_kit.dataset_export import (
    load_prompt_specs,
    resample_wav_to_16k_mono,
    write_sharded_jsonl,
)
from apollo_colab_kit.schemas import (
    DatasetRecordV1,
    MidiArtifactV1,
    MidiNoteEventV1,
    ScorePlanV1,
    VerifierScoreSnapshotV1,
)


def _dummy_record(record_id: str, split: str) -> DatasetRecordV1:
    return DatasetRecordV1(
        record_id=record_id,
        split="eval" if split == "eval" else "train",
        seed=42,
        prompt="lofi chill beat at 82 BPM in C major",
        genre="lofi",
        score_plan=ScorePlanV1.model_validate(
            {
                "schema_version": "score_plan_v1",
                "prompt": "lofi chill beat at 82 BPM in C major",
                "bpm": 82,
                "key": "C",
                "mode": "major",
                "genre": "lofi",
                "sections": [{"name": "intro", "type": "intro", "bars": 4}],
                "tracks": [{"role": "keys", "instrument": "rhodes"}],
            }
        ),
        midi=MidiArtifactV1(
            note_count=1,
            events=[MidiNoteEventV1(start_tick=0, duration_ticks=480, pitch=60, velocity=80)],
        ),
        verifier_scores=VerifierScoreSnapshotV1(
            total_reward=0.9,
            schema_reward=1.0,
            genre_rules_reward=0.8,
            instrument_range_reward=1.0,
            critics_reward=0.85,
            midi_quality_reward=0.85,
        ),
    )


def test_load_prompt_specs_json_and_jsonl(tmp_path: Path):
    json_path = tmp_path / "prompts.json"
    json_path.write_text(
        json.dumps(
            [
                {"id": "p1", "prompt": "trap beat 140 bpm", "expected_genre": "trap"},
                {"id": "p2", "prompt": "lofi beat 80 bpm", "expected_genre": "lofi"},
            ]
        ),
        encoding="utf-8",
    )
    specs = load_prompt_specs(json_path)
    assert len(specs) == 2
    assert specs[0].record_id == "p1"
    assert specs[0].genre_hint == "trap"

    jsonl_path = tmp_path / "prompts.jsonl"
    jsonl_path.write_text(
        json.dumps({"prompt": "jazz trio"}) + "\n",
        encoding="utf-8",
    )
    specs_jsonl = load_prompt_specs(jsonl_path)
    assert len(specs_jsonl) == 1
    assert specs_jsonl[0].prompt == "jazz trio"


def test_resample_wav_to_16k_mono_pure_python(tmp_path: Path):
    src = tmp_path / "stereo_44k.wav"
    dst = tmp_path / "mono_16k.wav"

    sr = 44100
    n_frames = 4410  # 0.1s stereo
    samples = []
    for i in range(n_frames):
        val = int(16000 * math.sin(2.0 * math.pi * 440.0 * i / sr))
        samples.extend([val, val])

    with wave.open(str(src), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    duration_sec, sha256 = resample_wav_to_16k_mono(src, dst)
    assert dst.is_file()
    assert len(sha256) == 64
    assert 0.095 <= duration_sec <= 0.105

    with wave.open(str(dst), "rb") as out_wf:
        assert out_wf.getnchannels() == 1
        assert out_wf.getsampwidth() == 2
        assert out_wf.getframerate() == 16000


def test_write_sharded_jsonl_and_manifest(tmp_path: Path):
    records = [
        _dummy_record("r1", "train"),
        _dummy_record("r2", "train"),
        _dummy_record("r3", "train"),
        _dummy_record("r4", "eval"),
    ]
    manifest = write_sharded_jsonl(records, output_dir=tmp_path, shard_size=2)
    assert manifest["total_records"] == 4
    assert manifest["train_records"] == 3
    assert manifest["eval_records"] == 1
    assert (tmp_path / "train-00000.jsonl").is_file()
    assert (tmp_path / "train-00001.jsonl").is_file()
    assert (tmp_path / "eval-00000.jsonl").is_file()
    assert (tmp_path / "dataset_manifest.json").is_file()

    # Every emitted line round-trips strictly through DatasetRecordV1
    for shard in manifest["shards"]:
        lines = (tmp_path / shard["path"]).read_text(encoding="utf-8").splitlines()
        assert len(lines) == shard["records"]
        for line in lines:
            rec = DatasetRecordV1.model_validate_json(line)
            assert rec.split == shard["split"]
