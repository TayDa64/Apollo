# Apollo v2 Handoff Implementation Plan — Instrument Realism, Genre Authenticity & Desktop Integration

## 1. Verified v1 Baseline (What Already Works)

All work below is verified in `C:\dev\apollo` (`origin = https://github.com/TayDa64/Apollo.git`, branch `main`, zero edits to the 5 protected monoliths):

- **Pure-Python Colab Kit (`packages/apollo_colab_kit/`)**: Standalone wheel (`apollo_colab_kit-0.1.0-py3-none-any.whl`), 3,900-row 17-genre balanced dataset sweep (`train.jsonl` / `val.jsonl`) with zero leakage against 22 held-out golden prompts (`eval_golden.jsonl`).
- **Trained Neural Planner (Gemma 4 E4B-it SFT + GRPO)**:
  - Held-out 22-prompt `eval_golden.jsonl` benchmark (verified with Gemma 4 stop-token list `eos_token_id=[1, 50, 106]`):
    - `strict_schema_pass_rate`: `0.0000 -> 0.9545` (21/22)
    - `mean_schema_reward`: `0.5000 -> 0.9773`
    - `mean_genre_rules_reward`: `0.8682 -> 0.9364`
    - `mean_total_reward`: `0.6473 -> 0.9609`
    - `genre_match_rate`: `0.0000 -> 0.8636`
- **Local Offline RTX 3080 Inference**:
  - Model weights: `C:\dev\apollo\models\gemma-4-E4B-it.Q4_K_M.gguf` (5.34 GB, gitignored)
  - Local server binary: `C:\dev\apollo\tools\llama-cpp\llama-server.exe` (`b12631` Vulkan, gitignored)
  - End-to-end smoke runner: `scripts/apollo_local_gguf_smoke.py`
  - Measured local speed: `llama-server` boots in `4.16s`, generates ~89 tokens/sec (`18.5s` for 1,660 tokens), passes strict `ScorePlanV1` (`schema=1.0000`, `total=0.9200`), and renders MIDI + WAV.

---

## 2. Root-Cause Diagnosis of Current Audio Quality Gap

Listening to `output/apollo_gguf_smoke/apollo_smoke.wav` and inspecting `score_plan.json` + `AudioRenderer` output analysis revealed the exact gap:

- **What Gemma 4 got right**: Correct `bpm=92`, `key=G minor`, `genre=g_funk`, `Gm -> Cm -> Fm -> Cm` chord map, `funky_syncopated_bass`, `talkbox`, `clean_whine_lead`, `drums`, and `avoid_instruments=["808_distorted"]`.
- **Why the rendered WAV did NOT sound authentic**:
  1. **Procedural fallback timbres**: `AudioRenderer()` rendered without a loaded SoundFont (`*.sf2`) or sample expansion library, causing harsh procedural synthesis (`Spectral centroid 7943 Hz` vs expected `1000–3500 Hz`, `Spectral flatness 0.5706` vs `< 0.15`, `Drum presence 0.149` vs `>= 0.15`).
  2. **ScorePlan → MIDI/Audio binding gap**: `score_plan_adapter.py` currently transfers `bpm`, `key`, `mode`, `genre`, and `sections`, but does not yet bind `tracks` (`instrument`, `pattern_hint`, `octave`, `density`), `chord_map`, `cue_points`, or `InstrumentPatchV1` directly into MIDI generation and instrument selection.
  3. **No audio-level timbre/mixing verification loop**: v1 only verifies symbolic JSON/MIDI rules; it does not yet gate or optimize actual rendered audio realism.

---

## 3. Phased Apollo v2 Plan (For the `apollo` Workspace Agent)

> **Rule Reminder**: Read `AGENTS.md` first. Run `.venv/Scripts/python.exe -m pytest tests packages/apollo_colab_kit/tests --tb=short -q` and `.venv/Scripts/pyright.exe` before any monolith edits. Get explicit user clearance before refactoring any of the 5 protected monoliths (`midi_generator.py`, `audio_renderer.py`, `assets_gen.py`, `main.py`, `MainComponent.cpp`).

### Phase A — Authentic Instrument Timbres & SoundFont/Sample Binding (Highest Impact)
1. **Audit & wire real instrument assets**:
   - Inspect `multimodal_gen/instrument_manager.py`, `multimodal_gen/instrument_resolution.py`, `multimodal_gen/soundfont_extract.py`, and `scripts/fluidsynth_env_proof.py` to determine why `AudioRenderer` fell back to raw procedural oscillators.
   - Configure deterministic SoundFont (`.sf2`) + `instruments/` sample library resolution (including genre-specific drum kits, bass, Rhodes, brass, talkbox/lead, and Ethiopian instruments like krar/washint/kebero/masenqo).
2. **Activate `InstrumentPatchV1` synthesis & fallback binding**:
   - Wire `InstrumentPatchV1` (`multimodal_gen/instrument_patch.py` + `neural_runtime.py`) so each track in `ScorePlanV1` resolves to an authentic SF2 program, sample layer, or sculpted synth patch (proper low-pass filtering, envelope shaping, and velocity mapping — eliminating 7.9 kHz harshness).
3. **Add an automated audio-quality benchmark gate**:
   - Run a 5-genre local benchmark (`g_funk`, `boom_bap`, `ethio_jazz`, `trap_soul`, `house`) through `apollo_local_gguf_smoke.py` and verify `AudioRenderer`'s spectral centroid, rolloff, flatness, and drum presence pass genre targets.

### Phase B — Deep ScorePlan-to-MIDI & Genre Authenticity
1. **Honor the full `ScorePlanV1` in generation**:
   - Bind `plan.chord_map` directly into harmonic accompaniment instead of ignoring it in favor of default progressions.
   - Bind `plan.tracks` (`role`, `instrument`, `pattern_hint`, `octave`, `density`) and `plan.constraints` (`avoid_instruments`) through `score_plan_adapter.py` into pattern selection and track voicing.
   - Bind `plan.cue_points` (`fill`, `drop`, `breakdown`, `riser`) and per-section `energy`/`tension` into drum fills, transitions, and mix automation.
2. **Genre-authentic rhythm & articulation templates**:
   - Audit and tighten swing/groove, syncopation, and drum-kit mapping for core genres (`g_funk` laid-back swing + Moog bass + high sine/portamento lead; `ethio_jazz` / `eskista` qenet scales + kebero rhythms; `boom_bap` dusty swing; `trap_soul` warm Rhodes + controlled 808).

### Phase C — Apollo Desktop App & Neural Audio Options
1. **Wire local `llama-server` / `LocalGGUFNeuralAdapter` into `worker.py` + JUCE**:
   - Expose a clean toggle/config in `multimodal_gen/server/worker.py` so the desktop app can generate via local Apollo Gemma 4 GGUF with automatic fallback to deterministic planning if offline/disabled.
   - Optional isolated CMake pass to rebrand `AI Music Generator.exe` -> `Apollo.exe`.
2. **Evaluate Neural Audio / Timbre Rendering (v2)**:
   - Decide whether to keep hybrid **Gemma 4 Symbolic + High-Quality SF2/Sample/DSP Engine** (fast, editable MIDI/stems, DAW-ready) or add an opt-in neural audio renderer / timbre transfer stage via `NeuralAudioRenderRequest` (`neural_io_v1`).
   - Optionally train a **Gemma 4 12B** (`unsloth/gemma-4-12b-it`) or **26B-A4B MoE** checkpoint on Colab A100 80 GB with `InstrumentPatchV1` outputs and `--include-audio-16k` multimodal conditioning.
