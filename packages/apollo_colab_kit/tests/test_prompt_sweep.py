"""Pure-Python tests for apollo_colab_kit.prompt_sweep determinism and zero-leakage guarantees."""

from __future__ import annotations

from apollo_colab_kit.prompt_sweep import (
    GENRE_SWEEP_PROFILES,
    generate_prompt_sweep,
    is_leaking_against_excluded,
    normalize_prompt_text,
    token_jaccard_similarity,
)
from apollo_colab_kit.schemas import ScorePlanV1
from apollo_colab_kit.verifiers import genre_rules_reward


def test_prompt_sweep_is_deterministic_diverse_and_genre_rule_clean():
    c1 = generate_prompt_sweep(count=120, seed=2026)
    c2 = generate_prompt_sweep(count=120, seed=2026)

    assert len(c1) == 120
    assert [x.prompt for x in c1] == [x.prompt for x in c2]
    assert [x.score_plan for x in c1] == [x.score_plan for x in c2]

    genres_seen = {c.genre for c in c1}
    expected_genres = {p.genre for p in GENRE_SWEEP_PROFILES}
    assert genres_seen == expected_genres
    assert {"ethio_jazz", "ethiopian_traditional", "eskista"}.issubset(genres_seen)

    modes_seen = {c.score_plan["mode"] for c in c1}
    assert len(modes_seen) >= 5

    # Every candidate validates strictly and has clean genre-rules reward
    for cand in c1:
        plan = ScorePlanV1.model_validate(cand.score_plan)
        assert plan.schema_version == "score_plan_v1"
        assert genre_rules_reward(plan) >= 0.90


def test_prompt_sweep_enforces_zero_leakage_and_deduplication():
    sample_excluded = [
        "smooth g-funk west coast beat at 92 bpm in G minor with funky bassline and talkbox lead",
        "traditional ethiopian folk in C minor with krar melody, masenqo answers, and kebero drums",
    ]
    cands = generate_prompt_sweep(
        count=80,
        seed=99,
        exclude_prompts=sample_excluded,
        jaccard_threshold=0.65,
    )
    assert len(cands) == 80

    norms = [normalize_prompt_text(c.prompt) for c in cands]
    assert len(norms) == len(set(norms))

    excluded_norm = {normalize_prompt_text(p) for p in sample_excluded}
    for cand in cands:
        assert normalize_prompt_text(cand.prompt) not in excluded_norm
        for ex in sample_excluded:
            assert token_jaccard_similarity(cand.prompt, ex) < 0.65

    assert is_leaking_against_excluded(
        "Smooth G-Funk West Coast Beat at 92 BPM in G Minor with funky bassline and talkbox lead!",
        excluded_norm,
        [set(normalize_prompt_text(p).split()) for p in sample_excluded],
    )
