"""Command line for the routing layer.

  python -m routing.cli verify-models
  python -m routing.cli build-questions
  python -m routing.cli generate --split calibration|test
  python -m routing.cli judge --split calibration|test [--judge judge_a|judge_b]
  python -m routing.cli score --split calibration|test
  python -m routing.cli gates
  python -m routing.cli calibrate
  python -m routing.cli evaluate
  python -m routing.cli route --scorer mean_logprob --budget 0.20 --in answers.csv --out routed.csv
  python -m routing.cli run-all        (phases 1 to 3 in order; needs OPENAI_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

load_dotenv()

from . import data  # noqa: E402


def cmd_build_questions(_):
    from .generate import build_questions

    build_questions()


def cmd_generate(a):
    from .generate import generate_split

    generate_split(a.split)


def cmd_judge(a):
    from .judge import judge_split

    judge_split(a.split, (a.judge,) if a.judge else data.JUDGES)


def cmd_score(a):
    from . import scorers

    coef = None if a.split == "calibration" else scorers.load_coef()
    df = scorers.score_split(a.split, coef)
    print(f"wrote {data.path(a.split, 'scores')} ({len(df)} rows)")


def cmd_gates(_):
    from . import gates

    g = gates.run("test")
    print((data.RESULTS / "gates_test.md").read_text())
    return g


def cmd_calibrate(_):
    from . import calibrate

    c = calibrate.run()
    print(f"wrote {data.RESULTS / 'calibration.json'} (n={c['n']})")


def cmd_evaluate(_):
    from . import evaluate

    evaluate.run()
    print(f"wrote {data.REPORTS / 'routing_report.md'} and figures/*.png")


def cmd_route(a):
    """Apply a calibrated cutoff to any CSV with qid, answer_text, token_logprobs. Adds score and lane."""
    import pandas as pd

    from . import calibrate, scorers

    cal = calibrate.load()
    if a.scorer not in ("mean_logprob", "min_k_logprob", "length_residual"):
        raise SystemExit("route supports the zero-extra-call scorers: mean_logprob, min_k_logprob, length_residual")
    df = pd.read_csv(a.infile, dtype=str, keep_default_na=False)
    for col in ("qid", "answer_text", "token_logprobs"):
        if col not in df.columns:
            raise SystemExit(f"{a.infile} lacks column {col}")
    lps = df["token_logprobs"].map(json.loads)
    if a.scorer == "mean_logprob":
        s = lps.map(scorers.mean_logprob)
    elif a.scorer == "min_k_logprob":
        s = lps.map(scorers.min_k_logprob)
    else:
        coef = cal["length_residual"]
        s = lps.map(scorers.mean_logprob) - (coef["slope"] * lps.map(len) + coef["intercept"])
    review_cut = cal["budgets"][a.scorer][str(a.budget)]
    df["score"] = s
    df["lane"] = "return"
    df.loc[df["score"] >= review_cut, "lane"] = "review"
    if a.escalate_budget is not None:
        esc_cut = cal["budgets"][a.scorer][str(a.escalate_budget)]
        df.loc[(df["score"] >= esc_cut) & (df["lane"] == "return"), "lane"] = "escalate"
    df.to_csv(a.outfile, index=False)
    print(df["lane"].value_counts().to_string())
    print(f"cutoff review>={review_cut:.4f} from calibration n={cal['n']} hash={cal['calibration_hash']}")


def cmd_verify_models(_):
    """Every model id in config/models.json must exist on the OpenAI models endpoint and have a price."""
    import openai

    from .cost import require_keys, cost_usd

    require_keys("OPENAI_API_KEY")
    cfg = data.models_config()
    ids = {cfg[k] for k in ("cheap", "strong", "judge_a", "judge_b")}
    available = {m.id for m in openai.OpenAI().models.list().data}
    missing = sorted(ids - available)
    if missing:
        raise SystemExit(f"model ids not on the models endpoint: {missing}")
    if cfg["judge_a"] == cfg["judge_b"]:
        raise SystemExit("judge_a and judge_b must differ")
    for m in ids:
        cost_usd(m, 1, 1)
    print(f"verified {sorted(ids)} against the models endpoint")


def cmd_run_all(_):
    from .cost import require_keys

    require_keys("OPENAI_API_KEY")
    cmd_verify_models(None)
    from . import calibrate, evaluate, gates, scorers
    from .generate import build_questions, generate_split
    from .judge import judge_split

    if not data.path("test", "questions").exists():
        build_questions()
    for split in data.SPLITS:
        generate_split(split)
        judge_split(split)
    from .notes import write_phase1_notes

    write_phase1_notes()
    coef = None
    for split in data.SPLITS:
        scorers.score_split(split, coef)
        coef = scorers.load_coef()
    gates.run("test")
    calibrate.run()
    evaluate.run()
    print("done: reports/routing_report.md")


def main(argv=None):
    p = argparse.ArgumentParser(prog="routing.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-questions").set_defaults(fn=cmd_build_questions)
    s = sub.add_parser("generate")
    s.add_argument("--split", required=True, choices=data.SPLITS)
    s.set_defaults(fn=cmd_generate)
    s = sub.add_parser("judge")
    s.add_argument("--split", required=True, choices=data.SPLITS)
    s.add_argument("--judge", choices=data.JUDGES)
    s.set_defaults(fn=cmd_judge)
    s = sub.add_parser("score")
    s.add_argument("--split", required=True, choices=data.SPLITS)
    s.set_defaults(fn=cmd_score)
    sub.add_parser("gates").set_defaults(fn=cmd_gates)
    sub.add_parser("calibrate").set_defaults(fn=cmd_calibrate)
    sub.add_parser("evaluate").set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("route")
    s.add_argument("--scorer", default="mean_logprob")
    s.add_argument("--budget", type=float, default=0.20, help="review lane budget (0.1, 0.2, 0.3)")
    s.add_argument("--escalate-budget", type=float, default=None, help="optional larger budget for the escalate lane")
    s.add_argument("--in", dest="infile", required=True)
    s.add_argument("--out", dest="outfile", required=True)
    s.set_defaults(fn=cmd_route)
    s = sub.add_parser("notes")
    s.set_defaults(fn=lambda _: __import__("routing.notes", fromlist=["x"]).write_phase1_notes())
    sub.add_parser("verify-models").set_defaults(fn=cmd_verify_models)
    sub.add_parser("run-all").set_defaults(fn=cmd_run_all)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
