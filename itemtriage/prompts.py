"""Prompt rendering for the panel, built from items.parquet only, and the prompt leak guard.

Two prompts per item: a direct-letter multiple-choice prompt (no reasoning, a deliberate low-cost regime)
and a P(True) prompt that shows the keyed answer. Local models also see one cyclic shift of the options.
"""

from __future__ import annotations

import re

from . import paths

CATEGORY_RE = re.compile(r"\b(" + "|".join(c for c in paths.CATEGORY_IDS if "_" in c) + r")\b")


def subject_name(subject: str) -> str:
    return subject.replace("_", " ")


def permute(options: list, shift: int) -> tuple[list, list]:
    """Cyclic shift. Returns (shown options, original index of each shown position)."""
    n = len(options)
    order = [(i + shift) % n for i in range(n)]
    return [options[j] for j in order], order


def _options_block(options: list) -> str:
    return "\n".join(f"{paths.LETTERS[i]}. {o}" for i, o in enumerate(options))


def mc_prompt(item: dict, shift: int = 0) -> tuple[str, list]:
    shown, order = permute(list(item["options"]), shift)
    text = (f"The following is a multiple choice question about {subject_name(item['subject'])}. "
            "Reply with only the letter of the correct answer.\n\n"
            f"{item['question']}\n{_options_block(shown)}\nAnswer:")
    return text, order


def ptrue_prompt(item: dict) -> str:
    opts = list(item["options"])
    k = int(item["key_index"])
    return (f"The following is a multiple choice question about {subject_name(item['subject'])} with a proposed answer.\n\n"
            f"{item['question']}\n{_options_block(opts)}\n\n"
            f"Proposed answer: {paths.LETTERS[k]}. {opts[k]}\n"
            "Is the proposed answer correct? Answer with exactly one word, Yes or No.")


def check_prompt(prompt: str, item: dict, order: list | None = None, forbidden_texts: tuple = ()) -> None:
    """Raise if the prompt could leak label information.

    1. Rendered options must equal the item's options exactly, in original order or the declared permutation.
       A correct answer that appears as a non-key option is item text and is allowed.
    2. No error-category identifier may appear (identifiers containing an underscore, word-bounded).
    3. No annotation text in forbidden_texts may appear unless the item's own text already contains it.
    """
    opts = list(item["options"])
    expected = [opts[j] for j in order] if order is not None else opts
    if _options_block(expected) not in prompt:
        raise AssertionError(f"{item['item_id']}: rendered options differ from items.parquet")
    hit = CATEGORY_RE.search(prompt)
    if hit:
        raise AssertionError(f"{item['item_id']}: prompt contains category identifier {hit.group(0)!r}")
    own = (item["question"] + "\n" + "\n".join(opts)).lower()
    low = prompt.lower()
    for t in forbidden_texts:
        t = str(t).strip().lower()
        if len(t) >= 20 and t in low and t not in own:
            raise AssertionError(f"{item['item_id']}: prompt contains annotation text")
