"""itemtriage command line.

  python3 -m itemtriage.cli ingest          download MMLU-Redux 2.0, write items and labels
  python3 -m itemtriage.cli split           freeze the subject split (refuses to change it)
  python3 -m itemtriage.cli audit-prompts   render every prompt and run the full leak guard
  python3 -m itemtriage.cli findings        write reports/phase1_findings.md
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="itemtriage.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("ingest", "split", "audit-prompts", "findings"):
        sub.add_parser(name)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
