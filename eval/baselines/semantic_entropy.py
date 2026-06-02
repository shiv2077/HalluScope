"""Semantic Entropy baseline (Kuhn et al., 2023), simplified.

Sample K answers for a question, cluster them by meaning, and compute the
(normalized) entropy over cluster probabilities. High entropy => the model's
answer distribution is semantically unstable => higher hallucination risk.

The clustering + entropy are pure and testable. The sampling path takes an
injected ``sampler`` (a callable question -> answer) and an injected ``sim_fn``
(answer, answer -> [0,1] similarity, e.g. SBERT cosine or an NLI agreement),
so the scorer runs with any backend and the tests need no model.
"""

from __future__ import annotations

import math
from typing import Callable, List

SimFn = Callable[[str, str], float]


def cluster_answers(answers: List[str], sim_fn: SimFn, threshold: float) -> List[List[int]]:
    """Greedy single-link clustering: an answer joins the first cluster whose
    representative it is >= threshold similar to, else starts a new cluster."""
    clusters: List[List[int]] = []
    reps: List[str] = []
    for i, ans in enumerate(answers):
        placed = False
        for c, rep in enumerate(reps):
            if sim_fn(ans, rep) >= threshold:
                clusters[c].append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
            reps.append(ans)
    return clusters


def semantic_entropy(answers: List[str], sim_fn: SimFn, threshold: float = 0.7) -> float:
    """Normalized semantic entropy in [0, 1] over meaning clusters.

    0 => all samples mean the same thing; 1 => all distinct (max entropy for
    the given number of samples).
    """
    n = len(answers)
    if n <= 1:
        return 0.0
    clusters = cluster_answers(answers, sim_fn, threshold)
    probs = [len(c) / n for c in clusters]
    ent = -sum(p * math.log(p) for p in probs if p > 0)
    max_ent = math.log(n)  # all-distinct upper bound
    return ent / max_ent if max_ent > 0 else 0.0


class SemanticEntropyScorer:
    """Risk = normalized semantic entropy of K sampled answers."""

    name = "baseline_semantic_entropy"

    def __init__(self, sampler: Callable[[str], str], sim_fn: SimFn,
                 k: int = 5, threshold: float = 0.7):
        self.sampler = sampler
        self.sim_fn = sim_fn
        self.k = k
        self.threshold = threshold

    def score(self, question: str) -> float:
        answers = [self.sampler(question) for _ in range(self.k)]
        return semantic_entropy(answers, self.sim_fn, self.threshold)
