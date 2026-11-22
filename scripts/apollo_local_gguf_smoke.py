"""End-to-end local GGUF smoke test for Apollo (llama-server + neural_runtime + MIDI/WAV).

Starts local ``llama-server.exe`` on a loopback port with GPU offloading,
queries Apollo's ``RemoteOpenAICompatibleNeuralAdapter`` (``neural_io_v1``),
validates the returned ``ScorePlanV1`` with ``apollo_colab_kit.verifiers``,
adapts it via ``score_plan_adapter``, and renders MIDI + WAV artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apollo_colab_kit.verifiers import composite_grpo_reward, genre_rules_reward, schema_reward
from multimodal_gen.arranger import Arranger
from multimodal_gen.audio_renderer import AudioRenderer
from multimodal_gen.midi_generator import MidiGenerator
from multimodal_gen.score_plan_adapter import (
    build_sections_from_plan,
    score_plan_to_parsed_prompt,
    score_plan_to_performance_score,
)
from multimodal_gen.synthesizers.neural_runtime import NeuralBackendConfig, create_neural_runtime


def _wait_for_health(base_url: str, timeout_s: float = 90.0) -> None:
    deadline = time.perf_counter() + timeout_s
    last_err = None
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2.0) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read().decode("utf-8"))
                    if body.get("status") == "ok":
                        return
        except Exception as exc:
            last_err = exc
        time.sleep(0.5)
    raise RuntimeError(f"llama-server did not become healthy within {timeout_s}s (last error: {last_err})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apollo local GGUF end-to-end smoke test")
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "models" / "gemma-4-E4B-it.Q4_K_M.gguf",
        help="Path to local GGUF model",
    )
    parser.add_argument(
        "--server-bin",
        type=Path,
        default=REPO_ROOT / "tools" / "llama-cpp" / "llama-server.exe",
        help="Path to llama-server executable",
    )
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument(
        "--prompt",
        type=str,
        default="smooth g-funk west coast beat at 92 bpm in G minor with funky bassline and talkbox lead",
    )
    parser.add_argument("--expected-genre", type=str, default="g_funk")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "apollo_gguf_smoke",
    )
    parser.add_argument("--no-audio", action="store_true", help="Skip WAV audio rendering")
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"GGUF model not found: {args.model}")
    if not args.server_bin.is_file():
        raise FileNotFoundError(f"llama-server binary not found: {args.server_bin}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "llama_server.log"

    cmd = [
        str(args.server_bin),
        "-m",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "-c",
        "4096",
        "-ngl",
        "99",
        "--reasoning-budget",
        "0",
    ]

    print(f"Starting llama-server on 127.0.0.1:{args.port} with {args.model.name} ...")
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        try:
            t_boot = time.perf_counter()
            base_url = f"http://127.0.0.1:{args.port}"
            _wait_for_health(base_url, timeout_s=90.0)
            print(f"llama-server healthy in {time.perf_counter() - t_boot:.2f}s")

            cfg = NeuralBackendConfig(
                backend_name="apollo_gemma4_local_server",
                transport="remote_http",
                enabled=True,
                model_id="apollo-gemma-4-e4b-q4km",
                endpoint_url=f"{base_url}/v1/chat/completions",
                temperature=0.2,
                max_tokens=2048,
                timeout_seconds=120.0,
            )
            runtime = create_neural_runtime(cfg)
            resp = runtime.generate_plan({"request_id": "smoke-1", "prompt": args.prompt})

            print(f"Neural response: success={resp.success} latency={resp.latency_ms:.1f}ms usage={resp.usage}")
            if not resp.success or resp.score_plan is None:
                print(f"FAILED: skip_reason={resp.skip_reason} error={resp.error_message}")
                return 1

            plan_dict = resp.score_plan.to_schema_dict()
            plan_path = args.output_dir / "score_plan.json"
            plan_path.write_text(json.dumps(plan_dict, indent=2), encoding="utf-8")

            r_schema = schema_reward(plan_dict)
            r_genre = genre_rules_reward(plan_dict, genre=args.expected_genre)
            breakdown = composite_grpo_reward(plan_dict, genre=args.expected_genre)
            print(
                f"ScorePlanV1 verified: bpm={plan_dict.get('bpm')} key={plan_dict.get('key')} "
                f"mode={plan_dict.get('mode')} genre={plan_dict.get('genre')} "
                f"sections={len(plan_dict.get('sections', []))} tracks={len(plan_dict.get('tracks', []))}"
            )
            print(
                f"Verifier rewards: schema={r_schema:.4f} genre_rules={r_genre:.4f} "
                f"total={breakdown.total_reward:.4f}"
            )

            parsed = score_plan_to_parsed_prompt(plan_dict)
            perf_score = score_plan_to_performance_score(plan_dict)
            arrangement = Arranger().create_arrangement(parsed)
            arrangement.sections = build_sections_from_plan(plan_dict, time_signature=parsed.time_signature)
            print(f"Adapted PerformanceScore: {len(perf_score.sections)} sections, tempo_map={perf_score.tempo_map}")
            midi_gen = MidiGenerator()
            midi_file = midi_gen.generate(arrangement, parsed)
            midi_path = args.output_dir / "apollo_smoke.mid"
            midi_file.save(str(midi_path))
            print(f"Rendered MIDI: {midi_path} ({midi_path.stat().st_size:,} bytes)")

            if not args.no_audio:
                renderer = AudioRenderer()
                wav_path = args.output_dir / "apollo_smoke.wav"
                renderer.render_midi_file(str(midi_path), str(wav_path), parsed)
                print(f"Rendered WAV : {wav_path} ({wav_path.stat().st_size:,} bytes)")

            return 0
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
