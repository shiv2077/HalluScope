"""Text-similarity ARTIFACT baseline (confound control).

This is deliberately a *weak* baseline that uses only surface text — the
embedding (dis)similarity between the question and the model's answer — and
nothing about the model's internal uncertainty. Its purpose is to show that a
trivial text artifact does NOT separate hallucinations on HalluEval QA (i.e.
the dataset's length/fluency confound, which sank the v1 methodology, is
controlled). If HalluScope's HRI does not beat this baseline by DeLong, the
result is suspect.

risk = 1 - cos(SBERT(question), SBERT(answer))   (higher => answer drifts from
the question; a confound proxy, not a real uncertainty signal).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

EmbedFn = Callable[[List[str]], np.ndarray]


class TextSimilarityArtifactScorer:
    name = "baseline_text_similarity_artifact"

    def __init__(self, embed_fn: Optional[EmbedFn] = None):
        self._embed = embed_fn

    def _embedder(self) -> EmbedFn:
        if self._embed is None:
            from backend.app import config_store
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(config_store.grader()["sbert_model"])

            def _e(texts: List[str]) -> np.ndarray:
                return model.encode(
                    list(texts), convert_to_numpy=True, normalize_embeddings=True
                )

            self._embed = _e
        return self._embed

    def score(self, question: str, answer: str) -> float:
        emb = self._embedder()([question, answer])
        a = np.asarray(emb[0], dtype=np.float64)
        b = np.asarray(emb[1], dtype=np.float64)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        cos = float(np.dot(a, b) / denom)
        return float(min(1.0, max(0.0, 1.0 - cos)))
