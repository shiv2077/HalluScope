"""Stage 3 — post-generation semantic checks.

Computed from the (prompt, response) pair AFTER generation. Each signal is
bounded to [0, 1]; higher = riskier.

  semantic_novelty  — 1 - cos(SBERT(prompt), SBERT(response)). A response that
                      drifts far from the question is more likely to be
                      confabulated. Embedding is injectable for testing; the
                      default lazy-loads the SBERT model named in
                      config/grader.json.
  entity_expansion  — |entities(response) \\ entities(prompt)| /
                      (|entities(response)| + eps). The share of named entities
                      the model *introduced* that were not in the prompt — a
                      proxy for unsupported specifics.
  claim_density     — claim-bearing content per token. The real definition is
                      SVO dependency triples / tokens with quantitative claims
                      up-weighted; that requires a dependency parser
                      (parser="spacy", optional). The dependency-free default
                      (parser="heuristic") approximates it as
                      (n_sentences + w_quant * n_quantitative_mentions) /
                      n_tokens. DOCUMENTED as an approximation.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from backend.app import config_store
from backend.app.metrics.nlp_backend import entity_set

_SENT_SPLIT = re.compile(r"[.!?]+")
_WORD = re.compile(r"[A-Za-z0-9][\w'’\-]*")
# Quantitative mentions: bare numbers, percentages, years, ordinals, units.
_QUANT = re.compile(
    r"\b\d[\d,\.]*\s*(?:%|percent|million|billion|thousand|km|kg|m|cm|mm|"
    r"years?|months?|days?|hours?|minutes?|dollars?|usd|°c|°f)?\b",
    re.IGNORECASE,
)

EmbedFn = Callable[[List[str]], np.ndarray]

_default_embed: Optional[EmbedFn] = None


def _get_default_embed() -> EmbedFn:
    """Lazy SBERT embedder using the configured grader model."""
    global _default_embed
    if _default_embed is None:
        from sentence_transformers import SentenceTransformer

        model_name = config_store.grader()["sbert_model"]
        model = SentenceTransformer(model_name)

        def _embed(texts: List[str]) -> np.ndarray:
            return model.encode(
                list(texts), convert_to_numpy=True, normalize_embeddings=True
            )

        _default_embed = _embed
    return _default_embed


# ── semantic novelty ─────────────────────────────────────────────────
def semantic_novelty(
    prompt: str,
    response: str,
    embed_fn: Optional[EmbedFn] = None,
) -> float:
    """1 - cosine(prompt, response), clipped to [0, 1]."""
    if not prompt.strip() or not response.strip():
        return 0.0
    embed = embed_fn or _get_default_embed()
    emb = embed([prompt, response])
    a, b = np.asarray(emb[0], dtype=np.float64), np.asarray(emb[1], dtype=np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    cos = float(np.dot(a, b) / denom)
    return float(min(1.0, max(0.0, 1.0 - cos)))


# ── entity expansion ─────────────────────────────────────────────────
def entity_expansion(prompt: str, response: str, backend: str = "regex") -> float:
    """Share of response entities not present in the prompt (∈ [0, 1])."""
    p_ents = entity_set(prompt, backend=backend)
    r_ents = entity_set(response, backend=backend)
    if not r_ents:
        return 0.0
    new = r_ents - p_ents
    return len(new) / (len(r_ents) + 1e-12)


# ── claim density ────────────────────────────────────────────────────
def _claim_density_heuristic(response: str, w_quant: float) -> float:
    tokens = _WORD.findall(response)
    n_tokens = len(tokens)
    if n_tokens == 0:
        return 0.0
    sentences = [s for s in _SENT_SPLIT.split(response) if s.strip()]
    n_quant = len(_QUANT.findall(response))
    weighted = len(sentences) + w_quant * n_quant
    return float(min(1.0, weighted / n_tokens))


def _claim_density_spacy(response: str, w_quant: float) -> float:
    import spacy

    from backend.app.metrics.nlp_backend import _load_spacy  # reuse cache

    nlp = _load_spacy("en_core_web_sm")
    doc = nlp(response)
    n_tokens = len([t for t in doc if not t.is_space])
    if n_tokens == 0:
        return 0.0
    # SVO triples: a verb with both an nsubj and a (dobj|obj|attr|acomp) child.
    svo = 0
    for tok in doc:
        if tok.pos_ in ("VERB", "AUX"):
            deps = {c.dep_ for c in tok.children}
            if ("nsubj" in deps or "nsubjpass" in deps) and \
               deps & {"dobj", "obj", "attr", "acomp", "dative", "oprd"}:
                svo += 1
    n_quant = sum(1 for t in doc if t.like_num or t.ent_type_ in ("DATE", "QUANTITY", "PERCENT", "MONEY", "CARDINAL"))
    weighted = svo + w_quant * n_quant
    return float(min(1.0, weighted / n_tokens))


def claim_density(response: str, parser: str = "heuristic") -> float:
    """Claim-bearing content per token (∈ [0, 1]).

    parser="heuristic" (default, dependency-free) or "spacy" (true SVO triples;
    requires spaCy en_core_web_sm).
    """
    w_quant = float(config_store.norm()["claim_quant_weight"])
    if parser == "spacy":
        return _claim_density_spacy(response, w_quant)
    return _claim_density_heuristic(response, w_quant)


# ── aggregate ────────────────────────────────────────────────────────
def compute_semantic_checks(
    prompt: str,
    response: str,
    embed_fn: Optional[EmbedFn] = None,
    entity_backend: str = "regex",
    claim_parser: str = "heuristic",
    include_novelty: bool = True,
) -> Dict[str, Any]:
    """Compute the Stage 3 signals for a (prompt, response) pair.

    entity_expansion and claim_density are cheap (no model). semantic_novelty
    needs an embedder; set ``include_novelty=False`` (or rely on the caller's
    try/except) to skip the SBERT load — its value is then None.
    """
    novelty: Optional[float] = None
    if include_novelty:
        novelty = round(semantic_novelty(prompt, response, embed_fn), 4)
    return {
        "semantic_novelty": novelty,
        "entity_expansion": round(entity_expansion(prompt, response, entity_backend), 4),
        "claim_density": round(claim_density(response, claim_parser), 4),
    }
