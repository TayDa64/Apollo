# AGENTS.md — Apollo Repository Rules

**Apollo** is the standalone, neural-first music generation successor forked from **MUSE** (`multimodal-ai-music-gen`). Internal synthesis and desktop engine paths (`multimodal_gen/` and `juce/Source/`) are intentionally preserved for v1 zero-regression stability while external contracts, CLI tooling, and Colab training packages are branded under Apollo.

This file aligns Antigravity and coding agents with [.github/copilot-instructions.md](.github/copilot-instructions.md), [PROTOCOL.md](PROTOCOL.md), and the architectural contracts in `docs/muse-specs/`.

## Core Principles

1. **Work strictly from verified code and command output** — never guess behavior or claim tests pass without executing them.
2. **Preserve existing functionality** — maintain backward compatibility across OSC, JSON-RPC, CLI, and JUCE boundaries.
3. **Minimal, surgical diffs** — keep changes scoped to the active phase and verified by unit/contract tests.

## Protected Monoliths (Do Not Refactor Without Explicit Gate Clearance)

Do **not** refactor or broadly edit these files unless baseline tests and `packages/apollo_colab_kit/` contract tests are green and the task explicitly authorizes it:

- `multimodal_gen/midi_generator.py`
- `multimodal_gen/audio_renderer.py`
- `multimodal_gen/assets_gen.py`
- `main.py`
- `juce/Source/MainComponent.cpp`

## Canonical Environment & Diagnostics

- **Python interpreter**: Use the repo-local `.venv` on Python 3.13 (`./.venv/Scripts/python.exe` on Windows; `./.venv/bin/python` on POSIX). Do **not** run global Python 3.10 `pytest`/`pyright` shims.
- **Test suite**: `.venv/Scripts/python.exe -m pytest tests packages/apollo_colab_kit/tests --tb=short -q`
- **Typecheck**: `.venv/Scripts/pyright.exe` (configured via `pyrightconfig.json`)
- **JUCE CMake build**: `node scripts/cmake.js -S juce -B juce/build` and `node scripts/cmake.js --build juce/build --config Release`
- **FluidSynth preflight**: `.venv/Scripts/python.exe scripts/fluidsynth_env_proof.py --check-only`

## Directory-Scoped Contracts

- `docs/muse-specs/schemas/` — Canonical frozen JSON Schemas (`score_plan.v1.schema.json`, `neural_runtime.v1.schema.json`). Any schema evolution must be versioned (`v2`) rather than breaking `v1`.
- `multimodal_gen/server/` & `juce/Source/Communication/Messages.h` — Must remain synchronized with `PROTOCOL.md` (`SCHEMA_VERSION = 1`).
- `multimodal_gen/synthesizers/neural_runtime.py` — Strict Pydantic v2 / JSON Schema I/O boundary (`neural_io_v1`) decoupling local GGUF / remote HTTP neural models (e.g., Gemma 4) from `worker.py` and JUCE.
- `packages/apollo_colab_kit/` — **Strictly pure-Python standalone package** (`apollo-colab-kit`, depending on `pydantic`, `mido`, `PyYAML` only). Must **never** import `multimodal_gen` at package import time and must have **zero** JUCE, FluidSynth, `pygame`, `librosa`, `numba`, `llvmlite`, `scipy`, `numpy`, `soundfile`, or `torch` dependencies in `schemas`, `verifiers`, or `prompt_sweep`.
- `tests/fixtures/` — Checked-in test fixtures (e.g., `golden_prompts.json`) must always remain tracked and protected in `.gitignore`, and must **never** leak into synthetic training shards.
- `.github/state/*.json` and `output/` — Mutable runtime/orchestration/bundle artifacts; kept ignored in `.gitignore` so routine runs do not pollute Git diffs.
