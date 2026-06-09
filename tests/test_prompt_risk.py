"""Direction tests for Stage 1 pre-generation prompt-risk signals (s1..s4)."""

from __future__ import annotations

from backend.app.metrics.prompt_risk import (
    entity_rarity_signal,
    time_risk,
    prompt_perplexity_signal,
    quoted_title_risk,
    compute_prompt_risk,
)


# s1 — entity rarity
def test_entity_rarity_higher_for_obscure_prompt():
    obscure = entity_rarity_signal("Who succeeded Pope Hormisdas at Aquileia?")
    common = entity_rarity_signal("What is the colour of the sky?")
    assert obscure > common


# s2 — time risk
def test_time_risk_higher_for_post_cutoff_year():
    risky = time_risk("Who won the 2027 world championship?")
    safe = time_risk("What is the boiling point of water?")
    assert risky > safe
    assert 0.0 <= risky <= 1.0


def test_time_risk_flags_relative_markers():
    assert time_risk("What happened in the news today?") > 0.0


# s3 — prompt perplexity
def test_prompt_perplexity_higher_for_more_negative_logprobs():
    confident = prompt_perplexity_signal([-0.05, -0.1, -0.05])  # low ppl
    uncertain = prompt_perplexity_signal([-4.0, -5.0, -4.5])  # high ppl
    assert uncertain > confident
    assert 0.0 <= uncertain <= 1.0


def test_prompt_perplexity_none_is_neutral_zero():
    assert prompt_perplexity_signal(None) == 0.0
    assert prompt_perplexity_signal([]) == 0.0


# s4 — quoted-title detection
def test_quoted_title_detected():
    quoted = quoted_title_risk('Who wrote "The Great Gatsby"?')
    plain = quoted_title_risk("Who wrote that book?")
    assert quoted > plain
    assert 0.0 <= quoted <= 1.0


# aggregate
def test_compute_prompt_risk_aggregates_and_bounds():
    out = compute_prompt_risk('Who won the 2027 "Tour de France" in Vinkovci?')
    assert set(
        [
            "s1_entity_rarity",
            "s2_time_risk",
            "s3_prompt_perplexity",
            "s4_quoted_title",
            "prompt_risk",
        ]
    ).issubset(out.keys())
    assert 0.0 <= out["prompt_risk"] <= 1.0


def test_compute_prompt_risk_monotonic_in_signals():
    risky = compute_prompt_risk('Who won the 2027 "Tour de France" in Vinkovci?')
    calm = compute_prompt_risk("What is the colour of the sky?")
    assert risky["prompt_risk"] > calm["prompt_risk"]
