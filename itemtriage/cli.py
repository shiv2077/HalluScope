"""itemtriage command line.

  python3 -m itemtriage.cli ingest          download MMLU-Redux 2.0, write items and labels
  python3 -m itemtriage.cli split           freeze the subject split (refuses to change it)
  python3 -m itemtriage.cli audit-prompts   render every prompt and run the full leak guard
  python3 -m itemtriage.cli findings        write reports/phase1_findings.md
  python3 -m itemtriage.cli panel-api --pilot | --full   API panel pass (cost gate first)
  .venv/bin/python -m itemtriage.cli panel-local --pilot | --full [--models a,b]   local GPU pass
  python3 -m itemtriage.cli collect         raw responses -> panel_responses.parquet, ptrue_responses.parquet
  python3 -m itemtriage.cli pilot           apply the preregistered panel selection rule
  python3 -m itemtriage.cli score           scorers and IRT -> results/itemtriage/scores.parquet
  python3 -m itemtriage.cli evaluate        H1 on test (refuses unless PREREG is committed and clean)
  python3 -m itemtriage.cli calibrate       H3: cutoff transfer and the risk lane
  .venv/bin/python -m itemtriage.cli hidden-capture   H2 capture on the GPU
  python3 -m itemtriage.cli h2              H2 probe and contrast
  python3 -m itemtriage.cli label --annotator NAME    blind 150-item spot check
  python3 -m itemtriage.cli matrix [--download]   Study 2 response matrix (network only with --download)
  python3 -m itemtriage.cli explore         exploratory subject versus item analysis (post-unblinding)
  python3 -m itemtriage.cli report          reports/itemtriage_report.md from results JSON
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="itemtriage.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("ingest", "split", "audit-prompts", "findings", "collect", "pilot", "score", "evaluate", "calibrate",
                 "hidden-capture", "h2", "explore", "report"):
        sub.add_parser(name)
    mx = sub.add_parser("matrix")
    mx.add_argument("--download", action="store_true", help="fetch lb.pickle, detail files, platinum and parameter counts")
    lb = sub.add_parser("label")
    lb.add_argument("--annotator", required=True)
    pa = sub.add_parser("panel-api")
    g = pa.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--full", action="store_true")
    pl = sub.add_parser("panel-local")
    g2 = pl.add_mutually_exclusive_group(required=True)
    g2.add_argument("--pilot", action="store_true")
    g2.add_argument("--full", action="store_true")
    pl.add_argument("--models", default=None, help="comma-separated slugs; default is the five local panel models")
    a = p.parse_args(argv)
    if a.cmd == "ingest":
        from . import ingest
        print(json.dumps(ingest.run(), indent=1))
    elif a.cmd == "split":
        from . import splits
        s = splits.freeze()
        print(f"split {s['sha256'][:16]}: " + ", ".join(f"{k} {len(v)} items" for k, v in s["item_ids"].items()))
    elif a.cmd == "audit-prompts":
        from . import audit
        print(json.dumps(audit.run()))
    elif a.cmd == "findings":
        from . import findings
        o = findings.run()
        print(f"wrote reports/phase1_findings.md (test primary positives {o['folds']['test']['primary']})")
    elif a.cmd == "panel-api":
        from dotenv import load_dotenv

        from . import panel_api
        load_dotenv()
        if a.pilot:
            out = panel_api.run(panel_api.pilot_ids(), "pilot")
        else:
            import pandas as pd

            from . import paths
            out = panel_api.run(sorted(pd.read_parquet(paths.ITEMS)["item_id"]), "full")
        print(json.dumps(out, indent=1, default=float))
    elif a.cmd == "collect":
        from . import collect
        print(json.dumps(collect.run()))
    elif a.cmd == "pilot":
        from . import pilot
        b = pilot.select()
        print(json.dumps({"panel": b["panel"], "replaced": b["replaced"], "stop": b["stop"]}))
    elif a.cmd == "score":
        from . import scorers
        print(json.dumps(scorers.run()))
    elif a.cmd == "evaluate":
        from . import evaluate
        o = evaluate.run()
        print(json.dumps({"h1a_holds": o["h1a"]["holds"], "deviation": o["prereg"]["deviation"]}))
    elif a.cmd == "calibrate":
        from . import calibrate
        calibrate.run()
        print("wrote results/itemtriage/calibration.json")
    elif a.cmd == "hidden-capture":
        from . import hidden
        print(json.dumps(hidden.capture()))
        hidden.features()
    elif a.cmd == "h2":
        from . import evaluate
        o = evaluate.run_h2()
        print(json.dumps({"h2_holds": o["h2"]["holds"]}))
    elif a.cmd == "label":
        from . import human
        human.run(a.annotator)
    elif a.cmd == "matrix":
        from . import matrix
        if a.download:
            print(json.dumps(matrix.download()))
            print(json.dumps(matrix.fetch_params()))
        m = matrix.build()
        print(json.dumps({k: m[k] for k in ("models", "representatives", "lineages_among_representatives", "gsm8k")}))
    elif a.cmd == "explore":
        from . import explore
        o = explore.run()
        print(f"wrote {explore.OUT.relative_to(explore.paths.ROOT)} ({o['all']['n']} test items, {o['all']['positives']} primary positives)")
    elif a.cmd == "report":
        from . import report
        report.write_final()
        report.update_readme()
        print("wrote reports/itemtriage_report.md and the README itemtriage block")
    elif a.cmd == "panel-local":
        from . import panel_api, panel_local
        slugs = a.models.split(",") if a.models else list(panel_local.PANEL_LOCAL)
        if a.pilot:
            ids, tag = panel_api.pilot_ids(), "pilot"
        else:
            import pandas as pd

            from . import paths
            ids, tag = sorted(pd.read_parquet(paths.ITEMS)["item_id"]), "full"
        if len(slugs) > 1:  # one process per model, so GPU memory is fully released between models
            import subprocess
            flag = "--pilot" if a.pilot else "--full"
            for slug in slugs:
                rc = subprocess.run([sys.executable, "-m", "itemtriage.cli", "panel-local", flag, "--models", slug]).returncode
                if rc:
                    return rc
        else:
            print(json.dumps(panel_local.run(slugs[0], ids, tag)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
