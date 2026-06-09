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

Reproducibility:
    Add --eval-deterministic to force greedy decoding (temperature=0) and a
    fixed per-request seed. Without it, decoding is stochastic (temperature
    0.7) and the resulting CSV is NOT reproducible. Use it for any reported
    benchmark:
    python3 eval/run_benchmark.py --mode local --eval-deterministic --seed 42
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

from backend.app import config_store  # noqa: E402

# Per-row schema. Beyond the original summary columns we now log every RAW
# per-signal FEATURE so eval/fit_weights.py can learn the HRI combination and
# eval/ablation.py can leave signals out. 'seed' supports multi-seed runs.
CSV_FIELDS = [
    "id",
    "seed",
    "hri",
    "risk_level",
    "hri_mode",
    # logprob features
    "mean_logprob",
    "abs_mean_logprob",
    "logprob_variance",
    "confidence_decay",
    "perplexity",
    # Stage 2 internal features
    "mean_semantic_energy",
    "mean_rcd",
    "rcd_inv",
    "mean_lld",
    "mean_csg",
    # Stage 1 prompt-risk features
    "s1_entity_rarity",
    "s2_time_risk",
    "s3_prompt_perplexity",
    "s4_quoted_title",
    "prompt_risk",
    # Stage 3 semantic checks
    "semantic_novelty",
    "entity_expansion",
    "claim_density",
    "intervened",
    "question",
    "reference_answer",
    "model_answer",
    "model_match",
]


def build_eval_prompt(question: str) -> str:
    # v2: just the question. HalluScope receives no candidate answer.
    return question


def _row_features(body: dict, rcd_scale: float) -> dict:
    """Flatten the /chat response into the feature columns above."""
    metrics = body.get("metrics", {}) or {}
    pr = body.get("prompt_risk", {}) or {}
    sc = body.get("semantic_checks", {}) or {}
    mean_lp = metrics.get("mean_logprob")
    mean_rcd = metrics.get("mean_rcd")
    rcd_inv = None
    if mean_rcd is not None:
        rcd_inv = 1.0 - min(float(mean_rcd) * rcd_scale, 1.0)
    return {
        "mean_logprob": mean_lp,
        "abs_mean_logprob": abs(mean_lp) if mean_lp is not None else None,
        "logprob_variance": metrics.get("logprob_variance"),
        "confidence_decay": metrics.get("confidence_decay"),
        "perplexity": metrics.get("perplexity"),
        "mean_semantic_energy": metrics.get("mean_semantic_energy"),
        "mean_rcd": mean_rcd,
        "rcd_inv": rcd_inv,
        "mean_lld": metrics.get("mean_lld"),
        "mean_csg": metrics.get("mean_csg"),
        "s1_entity_rarity": pr.get("s1_entity_rarity"),
        "s2_time_risk": pr.get("s2_time_risk"),
        "s3_prompt_perplexity": pr.get("s3_prompt_perplexity"),
        "s4_quoted_title": pr.get("s4_quoted_title"),
        "prompt_risk": pr.get("prompt_risk"),
        "semantic_novelty": sc.get("semantic_novelty"),
        "entity_expansion": sc.get("entity_expansion"),
        "claim_density": sc.get("claim_density"),
    }


def open_csv_appender(out_path: Path):
    """Open the CSV; write header only if the file is fresh / empty.

    Refuses to append to an existing file whose header does not match the
    current schema (e.g. a legacy results CSV without the feature columns) —
    that would silently corrupt it. The human picks a fresh --out instead.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fresh = (not out_path.exists()) or out_path.stat().st_size == 0
    if not fresh:
        with open(out_path, newline="", encoding="utf-8") as existing:
            header = next(csv.reader(existing), [])
        if header != CSV_FIELDS:
            raise SystemExit(
                f"Refusing to append: {out_path} has an incompatible header "
                f"(legacy schema?). Pass --out <new_path> to write a fresh "
                f"feature-rich CSV and leave the existing file untouched."
            )
    f = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if fresh:
        w.writeheader()
        f.flush()
    return f, w


def already_done_ids(out_path: Path) -> set:
    """Return the set of (id, seed) pairs already present (for resume)."""
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()
    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {(row["id"], row.get("seed", "")) for row in reader}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["api", "local"], required=True)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of samples (otherwise all in the file).",
    )
    ap.add_argument(
        "--dataset-file",
        type=str,
        default=str(DATASET_PATH),
        help="Path to a questions JSON ([{id,question,reference_answer}]). "
        "Use the train/test splits from build_subset.py --split.",
    )
    ap.add_argument(
        "--rps-sleep",
        type=float,
        default=0.1,
        help="Seconds to sleep between API calls.",
    )
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--eval-deterministic",
        action="store_true",
        help="Greedy decoding (temperature=0) + fixed seed per "
        "request so the run is reproducible. STRONGLY "
        "recommended for any reported benchmark.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Single seed (default).")
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Multiple seeds -> one pass each, rows tagged with the "
        "seed column (for multi-seed variance). With "
        "--eval-deterministic + greedy decoding, seeds yield "
        "identical outputs; use without it (sampling) for "
        "genuine variance.",
    )
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]

    samples = json.loads(Path(args.dataset_file).read_text())
    if args.limit:
        samples = samples[: args.limit]

    out_path = (
        Path(args.out) if args.out else (ROOT / "eval" / f"results_{args.mode}.csv")
    )
    done = already_done_ids(out_path)
    if done:
        print(f"[resume] {len(done)} rows already in {out_path.name}, skipping those")

    # Defer FastAPI/TestClient import until after argparse
    from fastapi.testclient import TestClient
    from backend.app.main import app

    print(f"Booting in-process app for mode={args.mode}, n={len(samples)}...")

    f, writer = open_csv_appender(out_path)
    rcd_scale = float(config_store.norm()["rcd_scale"])
    n_total = len(samples) * len(seeds)
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

        i = 0
        for seed in seeds:
            for row in samples:
                i += 1
                if (row["id"], str(seed)) in done:
                    continue

                # Reset singleton conversation history before each evaluation
                client.post("/chat/reset")

                prompt = build_eval_prompt(row["question"])
                payload = {
                    "message": prompt,
                    "evaluation_mode": args.mode,
                    "seed": seed,  # always seed RNGs for reproducibility
                    # Capture the model's REAL answer for grading. The
                    # intervention decision is still reported in `intervention`
                    # (logged as the `intervened` column); it is just not
                    # allowed to replace the answer the grader needs to see.
                    "apply_intervention": False,
                }
                if args.eval_deterministic:
                    payload["temperature"] = 0.0  # greedy => deterministic
                t0 = time.monotonic()
                try:
                    r = client.post("/chat", json=payload)
                    if r.status_code != 200:
                        print(f"\n[{row['id']}] HTTP {r.status_code}: {r.text[:200]}")
                        continue
                    body = r.json()
                except Exception as exc:
                    print(f"\n[{row['id']}] EXC: {exc}")
                    continue
                dt = time.monotonic() - t0

                rowout = {
                    "id": row["id"],
                    "seed": seed,
                    "hri": body.get("hri"),
                    "risk_level": body.get("risk_level"),
                    "hri_mode": (body.get("metrics", {}) or {}).get("hri_mode", ""),
                    "intervened": body.get("intervention", False),
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "model_answer": body.get("answer", ""),
                    "model_match": "",  # filled by eval/match.py post-run
                }
                rowout.update(_row_features(body, rcd_scale))
                writer.writerow(rowout)
                f.flush()
                n_done += 1

                elapsed = time.monotonic() - t_start
                rate = n_done / max(elapsed, 1e-3)
                eta = (n_total - n_done) / max(rate, 1e-3)
                print(
                    f"\r[{i}/{n_total}] {row['id']} s{seed}  "
                    f"hri={body.get('hri'):.3f} {body.get('risk_level'):>6}  "
                    f"({dt:.2f}s/call · ETA {eta/60:.1f}min)        ",
                    end="",
                    flush=True,
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
