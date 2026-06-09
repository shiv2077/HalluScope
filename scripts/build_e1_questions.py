"""Build the E1 question pool + frozen splits (CPU, network to HF Hub).

Samples 4000 TriviaQA (rc.nocontext/validation) + 2000 NQ-open (validation) at a
fixed seed, writes data/e1_questions.parquet (+ .meta.json), and freezes the
question-disjoint calib/val/test split to data/e1_splits.json.

Run: .venv/bin/python scripts/build_e1_questions.py [--n-triviaqa N] [--n-nq M]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hv2.config import REPO_ROOT, set_all_seeds  # noqa: E402
from src.hv2.datasets_hv2 import build_questions  # noqa: E402
from src.hv2.splits import freeze_splits  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-triviaqa", type=int, default=4000)
    ap.add_argument("--n-nq", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    q_path = REPO_ROOT / "data" / "e1_questions.parquet"
    s_path = REPO_ROOT / "data" / "e1_splits.json"

    print(f"[e1] sampling {args.n_triviaqa} TriviaQA + {args.n_nq} NQ-open (seed={args.seed}) ...")
    df = build_questions(args.n_triviaqa, args.n_nq, seed=args.seed, out_path=q_path)
    print(f"[e1] wrote {len(df)} questions -> {q_path}")
    print(df.source.value_counts().to_string())

    payload = freeze_splits(df["qid"].tolist(), s_path)
    print(f"[e1] froze splits -> {s_path}  counts={payload['counts']}  hash={payload['content_hash'][:12]}")

    # sanity: show 3 example rows
    import json

    for _, r in df.head(3).iterrows():
        print(f"  {r.qid} [{r.source}] {r.question[:70]!r} aliases={json.loads(r.gold_aliases)[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
