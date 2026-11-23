"""Every number in the README must appear, as written, in a generated report or a committed results file."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?")


def _readme_numbers() -> set[str]:
    text = ROOT.joinpath("README.md").read_text()
    text = re.sub(r"```.*?```", "", text, flags=re.S)          # commands
    text = re.sub(r"\]\([^)]*\)|https?://\S+", "", text)       # links and badges
    out = set()
    for tok in NUMBER.findall(text):
        if "." in tok or tok.endswith("%"):
            out.add(tok)
    return out


def _sources() -> str:
    files = [*ROOT.glob("reports/*.md"), *ROOT.glob("results/*.json"), *ROOT.glob("results/*.md"),
             *ROOT.glob("results/itemtriage/*.json"), ROOT / "HANDOFF.md"]
    return "\n".join(p.read_text() for p in files if p.exists())


def test_every_readme_number_is_in_a_generated_source():
    src = _sources()
    missing = sorted(n for n in _readme_numbers() if not re.search(r"(?<![\d.])" + re.escape(n) + r"(?!\d)", src))
    assert not missing, f"README numbers not found in reports/ or results/: {missing}"
