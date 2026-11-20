"""Phase 2 contract tests for the frozen neural_io_v1 generation interface."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from multimodal_gen.instrument_patch import (
    get_instrument_patch,
    get_registered_patch_families,
)
from multimodal_gen.session_graph import SessionGraph, SessionGraphBuilder
from multimodal_gen.synthesizers import (
    NEURAL_CONTRACT_VERSION,
    AudioConditioningRef,
    InstrumentPatchV1,
    LocalGGUFNeuralAdapter,
    NeuralBackendConfig,
    NeuralGenerationRequest,
    NeuralGenerationResponse,
    NeuralRuntimeProtocol,
    OptionalNeuralRuntime,
    RemoteOpenAICompatibleNeuralAdapter,
    ScorePlanV1,
    create_neural_runtime,
    export_neural_runtime_json_schema,
)


PROJECT_ROOT = Path(__file__).parent.parent
NEURAL_SCHEMA_PATH = PROJECT_ROOT / "docs" / "muse-specs" / "schemas" / "neural_runtime.v1.schema.json"
SCORE_PLAN_SCHEMA_PATH = PROJECT_ROOT / "docs" / "muse-specs" / "schemas" / "score_plan.v1.schema.json"


def _valid_score_plan_dict() -> Dict[str, Any]:
    return {
        "schema_version": "score_plan_v1",
        "prompt": "G-Funk cruising beat at 92 BPM in A minor",
        "bpm": 92,
        "key": "A",
        "mode": "minor",
        "time_signature": [4, 4],
        "genre": "g_funk",
        "sections": [
            {"name": "intro", "type": "intro", "bars": 4, "energy": 0.35, "tension": 0.25},
            {"name": "verse", "type": "verse", "bars": 8, "energy": 0.72, "tension": 0.65},
        ],
        "tracks": [
            {"role": "drums", "instrument": "kick"},
            {"role": "bass", "instrument": "synth_bass"},
            {"role": "lead", "instrument": "synth"},
        ],
    }


def test_committed_neural_runtime_json_schema_matches_code_exactly():
    assert NEURAL_SCHEMA_PATH.is_file()
    committed = json.loads(NEURAL_SCHEMA_PATH.read_text(encoding="utf-8"))
    generated = export_neural_runtime_json_schema()
    assert committed == generated
    assert committed["$id"] == "muse://schemas/neural-runtime.v1.json"


def test_score_plan_v1_strict_validation_and_jsonschema_parity():
    jsonschema = pytest.importorskip("jsonschema")
    raw_schema = json.loads(SCORE_PLAN_SCHEMA_PATH.read_text(encoding="utf-8"))

    plan = ScorePlanV1.model_validate(_valid_score_plan_dict())
    dumped = plan.to_schema_dict()
    jsonschema.validate(dumped, raw_schema)

    # Extra unknown properties must be rejected
    bad_extra = dict(_valid_score_plan_dict(), unknown_field="nope")
    with pytest.raises(ValidationError):
        ScorePlanV1.model_validate(bad_extra)

    # Bool coerced as int/float must be rejected
    bad_bool_bpm = dict(_valid_score_plan_dict(), bpm=True)
    with pytest.raises(ValidationError):
        ScorePlanV1.model_validate(bad_bool_bpm)

    # Frozen immutability
    with pytest.raises(ValidationError):
        plan.bpm = 120.0  # type: ignore[misc]


def test_all_registered_instrument_patches_validate_against_frozen_contract():
    families = get_registered_patch_families()
    assert len(families) >= 13
    for family in families:
        patch = get_instrument_patch(family, genre="ethio_jazz")
        assert patch is not None
        validated = InstrumentPatchV1.model_validate(patch.to_dict())
        assert validated.patch_id == patch.patch_id


def test_neural_generation_response_rejects_vacuous_success():
    with pytest.raises(ValidationError):
        NeuralGenerationResponse(
            request_id="req-empty",
            success=True,
            attempted=True,
            score_plan=None,
            instrument_patches=[],
        )


def test_audio_conditioning_ref_requires_16k_mono_and_path_or_uri():
    ref = AudioConditioningRef(path="/tmp/clip.wav", duration_seconds=4.5)
    assert ref.sample_rate_hz == 16000
    assert ref.channels == 1

    with pytest.raises(ValidationError):
        AudioConditioningRef()

    with pytest.raises(ValidationError):
        AudioConditioningRef(path="/tmp/clip.wav", sample_rate_hz=44100)  # type: ignore[arg-type]


def test_optional_neural_runtime_plan_callback_validates_strictly():
    runtime = OptionalNeuralRuntime(
        enabled=True,
        required_dependencies=(),
        plan_callback=lambda req: _valid_score_plan_dict(),
    )
    assert isinstance(runtime, NeuralRuntimeProtocol)
    status = runtime.probe()
    assert status.details["can_generate_plan"] is True
    assert status.details["contract_version"] == NEURAL_CONTRACT_VERSION

    resp = runtime.generate_plan(
        NeuralGenerationRequest(request_id="cb-1", prompt="G-Funk cruising beat")
    )
    assert resp.success is True
    assert resp.attempted is True
    assert resp.score_plan is not None
    assert resp.score_plan.bpm == 92

    # Invalid callback output fails cleanly with contract_validation_failed
    bad_runtime = OptionalNeuralRuntime(
        enabled=True,
        required_dependencies=(),
        plan_callback=lambda req: {"schema_version": "score_plan_v1", "bogus": 1},
    )
    bad_resp = bad_runtime.generate_plan({"request_id": "cb-2", "prompt": "bad"})
    assert bad_resp.success is False
    assert bad_resp.attempted is True
    assert bad_resp.skip_reason == "contract_validation_failed"


def test_remote_openai_compatible_adapter_parses_fenced_gemma4_response():
    captured_requests = []

    class _FakeHTTPResponse:
        def __init__(self, body_bytes: bytes) -> None:
            self._buf = io.BytesIO(body_bytes)

        def read(self) -> bytes:
            return self._buf.read()

        def __enter__(self) -> "_FakeHTTPResponse":
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

    def fake_urlopen(req, timeout=60.0):
        captured_requests.append((req.full_url, json.loads(req.data.decode("utf-8")), timeout))
        envelope = {
            "model": "gemma-4-e4b-it",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"```json\n{json.dumps(_valid_score_plan_dict())}\n```",
                    }
                }
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 180, "total_tokens": 300},
        }
        return _FakeHTTPResponse(json.dumps(envelope).encode("utf-8"))

    cfg = NeuralBackendConfig(
        backend_name="gemma4_remote",
        transport="remote_http",
        enabled=True,
        model_id="gemma-4-e4b-it",
        endpoint_url="http://127.0.0.1:8000/v1/chat/completions",
        seed=42,
    )
    adapter = RemoteOpenAICompatibleNeuralAdapter(cfg, urlopen_fn=fake_urlopen)
    assert isinstance(adapter, NeuralRuntimeProtocol)
    assert adapter.probe().details["can_generate_plan"] is True

    resp = adapter.generate_plan(
        NeuralGenerationRequest(request_id="rem-1", prompt="G-Funk beat at 92 BPM")
    )
    assert resp.success is True
    assert resp.transport == "remote_http"
    assert resp.model_id == "gemma-4-e4b-it"
    assert resp.score_plan is not None
    assert resp.usage["total_tokens"] == 300
    assert len(captured_requests) == 1
    assert captured_requests[0][1]["seed"] == 42


def test_local_gguf_adapter_honest_probe_and_injected_llama(tmp_path):
    model_file = tmp_path / "gemma-4-e4b-it.Q4_K_M.gguf"
    model_file.write_bytes(b"GGUF")

    # Without llama_factory (and without llama_cpp installed), probe reports missing_dependencies honestly
    cfg = NeuralBackendConfig(
        backend_name="gemma4_gguf",
        transport="local_gguf",
        enabled=True,
        model_id="gemma-4-e4b-it-q4",
        model_path=str(model_file),
        required_dependencies=("nonexistent_llama_cpp_pkg_xyz",),
    )
    uninstalled = LocalGGUFNeuralAdapter(cfg)
    probe = uninstalled.probe()
    assert probe.available is False
    assert probe.model_exists is True
    assert probe.skip_reason == "missing_dependencies"
    assert probe.details["can_generate_plan"] is False

    # With injected llama_factory, generation succeeds and validates strictly
    class _FakeLlama:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

        def create_chat_completion(self, **kwargs: Any) -> Dict[str, Any]:
            return {
                "choices": [
                    {"message": {"content": json.dumps(_valid_score_plan_dict())}}
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 110},
            }

    adapter = LocalGGUFNeuralAdapter(cfg, llama_factory=_FakeLlama)
    assert adapter.probe().details["can_generate_plan"] is True
    resp = adapter.generate_plan(NeuralGenerationRequest(request_id="gguf-1", prompt="G-Funk"))
    assert resp.success is True
    assert resp.transport == "local_gguf"
    assert resp.score_plan is not None
    assert resp.score_plan.key == "A"


def test_create_neural_runtime_factory_dispatches_by_transport():
    remote = create_neural_runtime(
        NeuralBackendConfig(transport="remote_http", enabled=False)
    )
    gguf = create_neural_runtime(
        NeuralBackendConfig(transport="local_gguf", enabled=False)
    )
    default = create_neural_runtime(NeuralBackendConfig(transport="disabled"))
    assert isinstance(remote, RemoteOpenAICompatibleNeuralAdapter)
    assert isinstance(gguf, LocalGGUFNeuralAdapter)
    assert isinstance(default, OptionalNeuralRuntime)


def test_session_graph_from_dict_is_non_mutating_and_exports_neural_context():
    graph = SessionGraph(bpm=92.0, key="A", scale="minor", genre="g_funk", total_bars=12)
    graph.add_section("intro", "intro", 0, 1920 * 4, energy_level=0.35)
    graph.add_section("verse", "verse", 1920 * 4, 1920 * 12, energy_level=0.75)
    track = graph.add_track("Bass", "bass", channel=1)
    track.add_clip(0, 1920 * 12, seed=7)

    serialized = graph.to_dict()
    snapshot_before = json.dumps(serialized, sort_keys=True)
    restored = SessionGraph.from_dict(serialized)
    snapshot_after = json.dumps(serialized, sort_keys=True)

    assert snapshot_before == snapshot_after
    assert restored.session_id == graph.session_id
    assert len(restored.tracks) == 1

    ctx = restored.to_neural_context()
    req = NeuralGenerationRequest(
        request_id="ctx-1",
        prompt="Extend with chorus",
        session_context=ctx,
    )
    assert req.session_context is not None
    assert req.session_context["bpm"] == 92.0
    assert req.session_context["genre"] == "g_funk"
