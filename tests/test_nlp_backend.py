"""Tests for the shared NLP backend (entity extraction + rarity).

Default backend is dependency-free (regex proper-noun heuristic + a bundled
common-words frequency source). spaCy is an optional backend; these tests
exercise the default so they run without any model download.
"""

from __future__ import annotations

from backend.app.metrics.nlp_backend import (
    extract_entities,
    rare_entity_fraction,
)


def test_extracts_proper_nouns():
    ents = extract_entities("Who is Zbigniew Brzezinski?")
    joined = " ".join(ents)
    assert "Zbigniew" in joined and "Brzezinski" in joined


def test_ignores_sentence_initial_common_words():
    # "What"/"The" are capitalized only because they start a sentence; they
    # are common words and must NOT be treated as named entities.
    ents = extract_entities("What is the dog called?")
    assert ents == []


def test_rare_fraction_higher_for_obscure_entities():
    obscure = rare_entity_fraction(
        "Who succeeded Pope Hormisdas in the Diocese of Aquileia?"
    )
    common = rare_entity_fraction("What is the colour of the sky?")
    assert obscure > common
    assert common == 0.0  # no rare named entities in a plain common-word prompt


def test_rare_fraction_bounded_unit_interval():
    for text in ["", "Xqzy Vronk Plimth", "the the the", "Paris London Tokyo"]:
        f = rare_entity_fraction(text)
        assert 0.0 <= f <= 1.0
