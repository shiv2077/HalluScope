"""Shared NLP backend: entity extraction + lexical rarity.

Two paper signals depend on named-entity extraction — Stage 1 *entity rarity*
and Stage 3 *entity expansion* — and rarity additionally needs a word-frequency
source. To keep the package importable and the tests runnable WITHOUT a heavy
model download, both have a dependency-free default backend and an optional
upgraded backend:

  Entity extractor
    * default ``"regex"``  — Title-Case proper-noun runs, common-word-trimmed.
    * optional ``"spacy"`` — spaCy ``en_core_web_sm`` NER (if installed).

  Frequency source (rarity)
    * default ``"builtin"``  — a bundled common-words list
      (config/common_words_en.txt); a token is "rare" if its lowercase form is
      not in the list. This is a CRUDE binary proxy.
    * optional ``"wordfreq"`` — graded zipf frequency (if the `wordfreq`
      package is installed); a token is "rare" if zipf < ``rare_zipf_max``.

The backend names are configurable (see ``extract_entities`` / ``Rarity``
parameters). DOCUMENTED LIMITATION: with the builtin source, well-known proper
nouns (e.g. "Paris") count as rare because they are not in the small common
list — install `wordfreq` for true frequency-graded rarity. The direction of
the signal (more obscure prompt => higher rarity) holds either way.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Set

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
COMMON_WORDS_PATH = ROOT_DIR / "config" / "common_words_en.txt"

# A maximal run of capitalised, possibly hyphenated/apostrophised tokens.
_TITLE_RUN = re.compile(r"\b([A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*)*)\b")
_TOKEN = re.compile(r"[A-Za-z][\w'’\-]*")


@lru_cache(maxsize=1)
def _common_words() -> Set[str]:
    words: Set[str] = set()
    if COMMON_WORDS_PATH.exists():
        for line in COMMON_WORDS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                words.add(line.lower())
    return words


def _is_common(token: str) -> bool:
    return token.lower() in _common_words()


# ── Entity extraction ────────────────────────────────────────────────
def _extract_regex(text: str) -> List[str]:
    entities: List[str] = []
    for match in _TITLE_RUN.finditer(text):
        run_tokens = match.group(1).split()
        # Trim leading/trailing common words (e.g. sentence-initial "What",
        # determiners like "The") so "The Diocese" -> "Diocese".
        while run_tokens and _is_common(run_tokens[0]):
            run_tokens.pop(0)
        while run_tokens and _is_common(run_tokens[-1]):
            run_tokens.pop()
        if run_tokens:
            entities.append(" ".join(run_tokens))
    return entities


def _extract_spacy(text: str, model: str) -> List[str]:
    import spacy

    nlp = _load_spacy(model)
    return [ent.text for ent in nlp(text).ents]


@lru_cache(maxsize=4)
def _load_spacy(model: str):
    import spacy

    return spacy.load(model)


def extract_entities(
    text: str,
    backend: str = "regex",
    spacy_model: str = "en_core_web_sm",
) -> List[str]:
    """Extract named entities from ``text``.

    Args:
        backend: ``"regex"`` (default, dependency-free) or ``"spacy"``.
        spacy_model: spaCy model name when ``backend="spacy"``.
    """
    if not text or not text.strip():
        return []
    if backend == "spacy":
        return _extract_spacy(text, spacy_model)
    return _extract_regex(text)


# ── Lexical rarity ───────────────────────────────────────────────────
def _token_is_rare(token: str, source: str, rare_zipf_max: float) -> bool:
    if source == "wordfreq":
        from wordfreq import zipf_frequency

        return zipf_frequency(token.lower(), "en") < rare_zipf_max
    # builtin: rare iff not in the bundled common-words list
    return not _is_common(token)


def rare_entity_fraction(
    text: str,
    entity_backend: str = "regex",
    frequency_source: str = "builtin",
    rare_zipf_max: float = 3.0,
) -> float:
    """Fraction of entity TOKENS judged rare. Bounded to [0, 1].

    Returns 0.0 when the text has no extracted named entities (a prompt of
    only common words carries no entity rarity). Higher => the prompt names
    more obscure entities => higher pre-generation hallucination risk.
    """
    entities = extract_entities(text, backend=entity_backend)
    tokens = [t for ent in entities for t in _TOKEN.findall(ent)]
    if not tokens:
        return 0.0
    rare = sum(1 for t in tokens if _token_is_rare(t, frequency_source, rare_zipf_max))
    return rare / len(tokens)


def entity_set(text: str, backend: str = "regex") -> Set[str]:
    """Lowercased set of entity surface forms (for Stage 3 entity expansion)."""
    return {e.lower() for e in extract_entities(text, backend=backend)}
