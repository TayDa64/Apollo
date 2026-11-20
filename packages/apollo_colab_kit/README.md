# `apollo-colab-kit`

Standalone, pure-Python training & verification boundary for **Apollo** (forked from MUSE) powering Google Colab / Unsloth SFT & GRPO workflows for Gemma 4.

## Hard Isolation Guarantee

`apollo-colab-kit` depends **only** on lightweight pure-Python packages:
- `pydantic>=2.0.0`
- `mido>=1.3.0`
- `PyYAML>=6.0.0`

It has **zero** dependencies on JUCE, FluidSynth, `pygame`, `librosa`, `numba`, `llvmlite`, `scipy`, `numpy`, `soundfile`, `torch`, or `multimodal_gen` when importing `apollo_colab_kit.schemas`, `apollo_colab_kit.verifiers`, `apollo_colab_kit.prompt_sweep`, or `apollo_colab_kit.dataset_export`.

## Modules

- **`apollo_colab_kit.schemas`**: Frozen Pydantic v2 validators and bundled JSON schemas for `ScorePlanV1`, `ArrangementTemplateV1`, `InstrumentPatchV1`, `NeuralRuntimeContractV1` (`neural_io_v1`), and `DatasetRecordV1` (`apollo_dataset_v1`).
- **`apollo_colab_kit.verifiers`**: Pure-Python deterministic verifiers (`critics`, `midi_features`, `instrument_ranges`, `genre_rules`, `quality`) and bounded `[0.0, 1.0]` GRPO reward callables (`rewards`, including `grpo_batch_reward_func` for TRL/Unsloth `GRPOTrainer`).
- **`apollo_colab_kit.prompt_sweep`**: Pure-Python deterministic prompt-sweep generator permuting presets, arrangement templates, genres (including Ethiopian modes), tempos, keys/all 7 modes, section counts, and genre-compliant instrumentation with strict zero-leakage filtering against held-out golden prompts.
- **`apollo_colab_kit.dataset_export`**: Local CLI (`apollo-colab-export`) run inside an Apollo checkout to export held-out `eval_golden.jsonl` and verified synthetic `train.jsonl` / `val.jsonl` shards for upload to Colab.

## Quick Usage

### In Google Colab (zero Apollo/MUSE install needed)

```python
from apollo_colab_kit.schemas import ScorePlanV1, DatasetRecordV1
from apollo_colab_kit.verifiers import composite_grpo_reward, grpo_batch_reward_func

breakdown = composite_grpo_reward(candidate_json_string)
print(breakdown.total_reward, breakdown.schema_reward, breakdown.genre_rules_reward)
```

### Locally inside Apollo (export bundle for Colab)

```bash
# 1. Held-out golden benchmark (never mixed into training)
apollo-colab-export \
  --prompts tests/fixtures/golden_prompts.json \
  --eval-golden \
  --output-dir output/apollo_colab_bundle

# 2. Deduplicated synthetic train/val sweep (zero overlap with golden prompts)
apollo-colab-export \
  --sweep \
  --train-target 3500 \
  --val-target 300 \
  --exclude-prompts tests/fixtures/golden_prompts.json \
  --output-dir output/apollo_colab_bundle \
  --seed 2026
```
