"""HalluEval QA benchmark runner for HalluScope (v2 methodology).

For each unique question, send ONLY the question to HalluScope. The
LLM generates its own free-form answer; we capture HRI on that answer.
The dataset's reference right_answer is recorded alongside, but only
for downstream SBERT-based grading (eval/match.py) — it is never
shown to HalluScope.

Each /chat call is preceded by /chat/reset so the singleton risk
service does not accumulate context across questions.

Usage:
    python3 eval/run_benchmark.py --mode api
    python3 eval/run_benchmark.py --mode local
    python3 eval/run_benchmark.py --mode local --limit 50    # smaller run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "eval" / "datasets" / "halluveval_qa_questions_100.json"

CSV_FIELDS = [
    "id", "hri", "risk_level", "hri_mode",
    "mean_logprob", "perplexity",
    "mean_semantic_energy", "mean_rcd",
    "intervened",
    "question", "reference_answer", "model_answer", "model_match",
]


def build_eval_prompt(question: str) -> str:
    # v2: just the question. HalluScope receives no candidate answer.
    return question


def open_csv_appender(out_path: Path):
    """Open the CSV; write header only if the file is fresh / empty."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fresh = (not out_path.exists()) or out_path.stat().st_size == 0
    f = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if fresh:
        w.writeheader()
        f.flush()
    return f, w


def already_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()
    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["id"] for row in reader}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["api", "local"], required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of samples (otherwise all 100).")
    ap.add_argument("--rps-sleep", type=float, default=0.1,
                    help="Seconds to sleep between API calls.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    samples = json.loads(DATASET_PATH.read_text())
    if args.limit:
        samples = samples[: args.limit]

    out_path = Path(args.out) if args.out else (
        ROOT / "eval" / f"results_{args.mode}.csv"
    )
    done = already_done_ids(out_path)
    if done:
        print(f"[resume] {len(done)} rows already in {out_path.name}, skipping those")

    # Defer FastAPI/TestClient import until after argparse
    from fastapi.testclient import TestClient
    from backend.app.main import app

    print(f"Booting in-process app for mode={args.mode}, n={len(samples)}...")

    f, writer = open_csv_appender(out_path)
    n_total = len(samples)
    n_done = 0
    t_start = time.monotonic()

    with TestClient(app) as client:
        # If local mode, wait for the warmup thread to finish
        if args.mode == "local":
            print("Waiting for local model warmup ...", end=" ", flush=True)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if client.get("/health").json().get("local_model_loaded"):
                    break
                time.sleep(2)
            else:
                print("FAILED")
                return 2
            print("ready.")

        for i, row in enumerate(samples, 1):
            if row["id"] in done:
                continue

            # Reset singleton conversation history before each evaluation
            client.post("/chat/reset")

            prompt = build_eval_prompt(row["question"])
            t0 = time.monotonic()
            try:
                r = client.post("/chat", json={
                    "message": prompt,
                    "evaluation_mode": args.mode,
                })
                if r.status_code != 200:
                    print(f"\n[{row['id']}] HTTP {r.status_code}: {r.text[:200]}")
                    continue
                body = r.json()
            except Exception as exc:
                print(f"\n[{row['id']}] EXC: {exc}")
                continue
            dt = time.monotonic() - t0

            metrics = body.get("metrics", {})
            writer.writerow({
                "id":                   row["id"],
                "hri":                  body.get("hri"),
                "risk_level":           body.get("risk_level"),
                "hri_mode":             metrics.get("hri_mode", ""),
                "mean_logprob":         metrics.get("mean_logprob"),
                "perplexity":           metrics.get("perplexity"),
                "mean_semantic_energy": metrics.get("mean_semantic_energy"),
                "mean_rcd":             metrics.get("mean_rcd"),
                "intervened":           body.get("intervention", False),
                "question":             row["question"],
                "reference_answer":     row["reference_answer"],
                "model_answer":         body.get("answer", ""),
                "model_match":          "",  # filled in by eval/match.py post-run
            })
            f.flush()
            n_done += 1

            elapsed = time.monotonic() - t_start
            rate = n_done / max(elapsed, 1e-3)
            eta = (n_total - n_done) / max(rate, 1e-3)
            print(
                f"\r[{i}/{n_total}] {row['id']}  "
                f"hri={body.get('hri'):.3f} {body.get('risk_level'):>6}  "
                f"({dt:.2f}s/call · ETA {eta/60:.1f}min)        ",
                end="", flush=True,
            )

            if args.mode == "api" and args.rps_sleep > 0:
                time.sleep(args.rps_sleep)

    f.close()
    elapsed = time.monotonic() - t_start
    print()
    print(f"Done. {n_done} new rows in {out_path}  ({elapsed/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
