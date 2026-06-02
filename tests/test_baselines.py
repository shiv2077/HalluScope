"""Tests for the pure pieces of the baseline scorers. The LLM-driven scoring
paths take injected callables, so these run without any model."""

from __future__ import annotations

import numpy as np

from eval.baselines.semantic_entropy import cluster_answers, semantic_entropy
from eval.baselines.p_true import parse_p_true
from eval.baselines.verbalized import parse_confidence
from eval.baselines.text_similarity import TextSimilarityArtifactScorer


def _exact_sim(a, b):
    return 1.0 if a.strip().lower() == b.strip().lower() else 0.0


# ── Semantic entropy ─────────────────────────────────────────────────
def test_cluster_collapses_identical_answers():
    clusters = cluster_answers(["Paris", "paris", "PARIS"], _exact_sim, threshold=0.5)
    assert len(clusters) == 1


def test_cluster_separates_distinct_answers():
    clusters = cluster_answers(["Paris", "London", "Berlin"], _exact_sim, 0.5)
    assert len(clusters) == 3


def test_semantic_entropy_zero_when_all_agree():
    ent = semantic_entropy(["Paris", "Paris", "Paris"], _exact_sim, 0.5)
    assert abs(ent - 0.0) < 1e-9


def test_semantic_entropy_max_when_all_distinct():
    answers = ["Paris", "London", "Berlin", "Rome"]
    ent = semantic_entropy(answers, _exact_sim, 0.5)
    assert abs(ent - 1.0) < 1e-9  # normalized to [0,1]: 4 equiprobable clusters


def test_semantic_entropy_higher_for_more_disagreement():
    agree = semantic_entropy(["Paris", "Paris", "Paris", "London"], _exact_sim, 0.5)
    disagree = semantic_entropy(["Paris", "London", "Berlin", "Rome"], _exact_sim, 0.5)
    assert disagree > agree


# ── P(True) parsing ──────────────────────────────────────────────────
def test_parse_p_true_reads_explicit_probability():
    assert abs(parse_p_true("Probability true: 0.85") - 0.85) < 1e-9


def test_parse_p_true_reads_true_false_words():
    assert parse_p_true("True") == 1.0
    assert parse_p_true("False, that is incorrect.") == 0.0


# ── Verbalized confidence parsing ────────────────────────────────────
def test_parse_confidence_percentage_and_decimal():
    assert abs(parse_confidence("Confidence: 90%") - 0.90) < 1e-9
    assert abs(parse_confidence("confidence 0.4 overall") - 0.40) < 1e-9


# ── Text-similarity artifact ─────────────────────────────────────────
def test_text_similarity_artifact_direction():
    def fake_embed(texts):
        table = {"q": np.array([1.0, 0.0]), "same": np.array([1.0, 0.0]),
                 "diff": np.array([0.0, 1.0])}
        return np.stack([table[t] for t in texts])

    scorer = TextSimilarityArtifactScorer(embed_fn=fake_embed)
    # risk = 1 - cos(question, answer): unrelated answer -> higher artifact risk
    assert scorer.score("q", "diff") > scorer.score("q", "same")
