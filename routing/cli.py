"""Command line for the routing layer.

  python -m routing.cli verify-models
  python -m routing.cli build-questions
  python -m routing.cli generate --split calibration|test
  python -m routing.cli judge --split calibration|test [--judge judge_a|judge_b]
  python -m routing.cli score --split calibration|test
  python -m routing.cli gates
  python -m routing.cli calibrate
  python -m routing.cli evaluate
  python -m routing.cli label --split test --n 60 --seed 7 --annotator <name>
  python -m routing.cli agreement --split test
  python -m routing.cli judge-change   (before and after replacing judge A, ADR-033)
  python -m routing.cli route --policy budget --scorer mean_logprob --budget 0.20 --in answers.csv --out routed.csv
  python -m routing.cli route --policy risk --scorer p_true --alpha 0.10 --in answers.csv --out routed.csv
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
    """Route any CSV with qid, answer_text, token_logprobs (plus optional precomputed scorer columns).

    --policy budget: flag the top ceil(budget * n) rows of this batch by batch quantile. Never reads
    results/calibration.json. --policy risk: apply the calibrated cutoff for --scorer and --alpha from
    results/calibration.json (or an explicit --cutoff); the flag rate floats.
    """
    import pandas as pd

    from . import scorers
    from .policy import batch_cutoff, flag_by_budget, flag_by_risk

    df = pd.read_csv(a.infile, dtype=str, keep_default_na=False)
    for col in ("qid", "answer_text", "token_logprobs"):
        if col not in df.columns:
            raise SystemExit(f"{a.infile} lacks column {col}")
    if a.scorer in df.columns:
        s = pd.to_numeric(df[a.scorer], errors="coerce")
    elif a.scorer in ("mean_logprob", "min_k_logprob"):
        fn = scorers.mean_logprob if a.scorer == "mean_logprob" else scorers.min_k_logprob
        s = df["token_logprobs"].map(json.loads).map(fn)
    else:
        raise SystemExit(f"{a.infile} has no column {a.scorer}; only mean_logprob and min_k_logprob "
                         "can be computed from token_logprobs")
    s.index = df["qid"]
    df["score"] = s.to_numpy()
    if a.policy == "budget":
        if a.budget is None:
            raise SystemExit("--policy budget requires --budget")
        if a.cutoff is not None or a.alpha is not None:
            raise SystemExit("--policy budget forbids --cutoff and --alpha")
        flagged = flag_by_budget(s, a.budget).to_numpy()
        cutoff = batch_cutoff(s, a.budget)
        esc = flag_by_budget(s, a.escalate_budget).to_numpy() if a.escalate_budget is not None else None
        source = f"batch quantile at budget {a.budget}"
    else:
        if a.cutoff is not None:
            cutoff, source = a.cutoff, "explicit --cutoff"
        else:
            if a.alpha is None:
                raise SystemExit("--policy risk requires --alpha (or --cutoff)")
            from . import calibrate

            cal = calibrate.load()
            r = cal["risk"].get(a.scorer, {}).get(str(a.alpha))
            if not r or r.get("threshold") is None:
                raise SystemExit(f"no calibrated risk cutoff for {a.scorer} at alpha {a.alpha} in "
                                 f"results/calibration.json: {(r or {}).get('reason', 'absent')}")
            cutoff = r["threshold"]
            source = f"calibration n={cal['n']} hash={cal['calibration_hash']} alpha={a.alpha}"
        flagged = flag_by_risk(s, cutoff).to_numpy()
        esc = None
    df["policy"] = a.policy
    df["cutoff_applied"] = cutoff
    df["lane"] = "return"
    if esc is not None:
        df.loc[esc, "lane"] = "escalate"
    df.loc[flagged, "lane"] = "review"
    df.to_csv(a.outfile, index=False)
    print(df["lane"].value_counts().to_string())
    print(f"policy={a.policy} scorer={a.scorer} cutoff_applied={cutoff:.4f} ({source}); flagged {int(flagged.sum())}/{len(df)}")


def cmd_label(a):
    from .label_tool import run

    run(a.split, a.n, a.seed, a.annotator)


def cmd_agreement(a):
    from .agreement import run

    run(a.split)


def cmd_verify_models(_):
    """Every model id in config/models.json must exist on the OpenAI models endpoint and have a price."""
    import openai

    from .cost import cost_usd, require_keys

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
    s.add_argument("--policy", required=True, choices=("budget", "risk"))
    s.add_argument("--scorer", required=True)
    s.add_argument("--budget", type=float, default=None, help="budget policy: fraction of the batch to flag")
    s.add_argument("--escalate-budget", type=float, default=None, help="budget policy: larger budget for the escalate lane")
    s.add_argument("--alpha", type=float, default=None, help="risk policy: alpha of the calibrated cutoff")
    s.add_argument("--cutoff", type=float, default=None, help="risk policy: explicit cutoff instead of calibration.json")
    s.add_argument("--in", dest="infile", required=True)
    s.add_argument("--out", dest="outfile", required=True)
    s.set_defaults(fn=cmd_route)
    s = sub.add_parser("notes")
    s.set_defaults(fn=lambda _: __import__("routing.notes", fromlist=["x"]).write_phase1_notes())
    s = sub.add_parser("label", help="blind human labeling in the terminal")
    s.add_argument("--split", default="test", choices=data.SPLITS)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--annotator", required=True)
    s.set_defaults(fn=cmd_label)
    s = sub.add_parser("agreement", help="human vs machine label agreement report")
    s.add_argument("--split", default="test", choices=data.SPLITS)
    s.set_defaults(fn=cmd_agreement)
    s = sub.add_parser("judge-change", help="before and after replacing judge A (ADR-033)")
    s.set_defaults(fn=lambda _: print(json.dumps(__import__("routing.judge_change", fromlist=["x"]).run()["rows_with_consensus"])))
    sub.add_parser("verify-models").set_defaults(fn=cmd_verify_models)
    sub.add_parser("run-all").set_defaults(fn=cmd_run_all)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
