"""Compute baseline risk scores for a results CSV (capability only).

Adds a ``baseline_<name>`` column per selected baseline to a NEW CSV so
eval/analyze.py can DeLong-test HalluScope's HRI against each. Produces NO
headline numbers — the human runs this and evaluates the output.

Baselines:
  text_similarity        cheap; SBERT cosine artifact (confound control).
  semantic_entropy       K stochastic samples per question, SBERT-clustered.
  p_true                 a second LLM call: "is your answer true?".
  verbalized             prompt the model to state its confidence.

Usage (human runs this; the assistant never fabricates outputs):
    # cheap artifact baseline only (needs SBERT, no generation):
    python3 eval/run_baselines.py --csv eval/results_api_test.csv \
        --which text_similarity --out eval/results_api_test_bl.csv

    # LLM baselines on the local model:
    python3 eval/run_baselines.py --csv eval/results_local_test.csv \
        --mode local --which p_true verbalized semantic_entropy \
        --out eval/results_local_test_bl.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.seeding import set_global_seed  # noqa: E402

ALL_BASELINES = ["text_similarity", "semantic_entropy", "p_true", "verbalized"]


def _llm_fn(mode: str) -> Callable[[str], str]:
    """Return a prompt->text callable backed by the chosen adapter."""
    if mode == "local":
        from backend.app.adapters.local_adapter import LocalAdapter

        adapter = LocalAdapter()
    else:
        from backend.app.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter()

    def _call(prompt: str) -> str:
        return adapter.generate(prompt, max_tokens=80, temperature=0.0).text

    return _call


def _sampler(mode: str, temperature: float) -> Callable[[str], str]:
    if mode == "local":
        from backend.app.adapters.local_adapter import LocalAdapter

        adapter = LocalAdapter()
    else:
        from backend.app.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter()

    def _s(question: str) -> str:
        return adapter.generate(question, max_tokens=80, temperature=temperature).text

    return _s


def _sbert_sim():
    from eval.match import cosine_similarity

    return lambda a, b: cosine_similarity(a, b)


def build_scorers(which: List[str], mode: str, k: int, temperature: float) -> Dict:
    from eval.baselines.text_similarity import TextSimilarityArtifactScorer
    from eval.baselines.semantic_entropy import SemanticEntropyScorer
    from eval.baselines.p_true import PTrueScorer
    from eval.baselines.verbalized import VerbalizedConfidenceScorer

    scorers: Dict = {}
    if "text_similarity" in which:
        scorers["text_similarity"] = TextSimilarityArtifactScorer()
    if "semantic_entropy" in which:
        scorers["semantic_entropy"] = SemanticEntropyScorer(
            _sampler(mode, temperature), _sbert_sim(), k=k
        )
    if "p_true" in which:
        scorers["p_true"] = PTrueScorer(_llm_fn(mode))
    if "verbalized" in which:
        scorers["verbalized"] = VerbalizedConfidenceScorer(_llm_fn(mode))
    return scorers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["api", "local"], default="local")
    ap.add_argument(
        "--which", nargs="+", choices=ALL_BASELINES, default=["text_similarity"]
    )
    ap.add_argument("--k", type=int, default=5, help="Samples for semantic entropy.")
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for semantic entropy.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seed(args.seed)
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if not rows:
        print("ERROR: empty CSV", file=sys.stderr)
        return 2

    scorers = build_scorers(args.which, args.mode, args.k, args.temperature)
    out_cols = list(rows[0].keys()) + [s.name for s in scorers.values()]

    for i, r in enumerate(rows, 1):
        q, ans = r.get("question", ""), r.get("model_answer", "")
        for key, scorer in scorers.items():
            if key == "text_similarity":
                val = scorer.score(q, ans)
            elif key == "p_true":
                val = scorer.score(q, ans)
            elif key == "verbalized":
                val = scorer.score(q)
            else:  # semantic_entropy
                val = scorer.score(q)
            r[scorer.name] = round(float(val), 4)
        print(f"\r[{i}/{len(rows)}] {r.get('id','')}        ", end="", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows with baselines {list(scorers)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
