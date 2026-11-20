# `muse-colab-kit`

Standalone, pure-Python training & verification boundary extracted from **MUSE** (`multimodal-ai-music-gen`) for Google Colab / Unsloth SFT & GRPO workflows.

## Hard Isolation Guarantee

`muse-colab-kit` depends **only** on lightweight pure-Python packages:
- `pydantic>=2.0.0`
- `mido>=1.3.0`
- `PyYAML>=6.0.0`

It has **zero** dependencies on JUCE, FluidSynth, `pygame`, `librosa`, `numba`, `llvmlite`, `scipy`, `numpy`, `soundfile`, `torch`, or `multimodal_gen` when importing `muse_colab_kit.schemas` or `muse_colab_kit.verifiers`.

## Modules

- **`muse_colab_kit.schemas`**: Frozen Pydantic v2 validators and bundled JSON schemas for `ScorePlanV1`, `ArrangementTemplateV1`, `InstrumentPatchV1`, `NeuralRuntimeContractV1` (`neural_io_v1`), and `DatasetRecordV1`.
- **`muse_colab_kit.verifiers`**: Pure-Python deterministic verifiers (`critics`, `midi_features`, `instrument_ranges`, `genre_rules`, `quality`) and bounded `[0.0, 1.0]` GRPO reward callables (`rewards`, including `grpo_batch_reward_func` for TRL/Unsloth `GRPOTrainer`).
- **`muse_colab_kit.dataset_export`**: Local CLI (`muse-colab-export`) run inside a MUSE checkout to export validated `(prompt [+ optional 16 kHz mono WAV]) -> score_plan / MIDI / patches` JSONL shards for upload to Colab.

## Quick Usage

### In Google Colab (zero MUSE install needed)

```python
from muse_colab_kit.schemas import ScorePlanV1, DatasetRecordV1
from muse_colab_kit.verifiers import composite_grpo_reward, grpo_batch_reward_func

breakdown = composite_grpo_reward(candidate_json_string)
print(breakdown.total_reward, breakdown.components)
```

### Locally inside MUSE (export shards for Colab)

```bash
muse-colab-export \
  --prompts tests/fixtures/golden_prompts.json \
  --output-dir output/colab_dataset \
  --seed 42
```

Add `--include-audio-16k` only if you want local audio rendering and 16 kHz mono PCM-16 WAV clips for multimodal audio-conditioned training (defaults to `False`).
