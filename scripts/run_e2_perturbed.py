"""E2 — perturbed main generations + strata table (GPU job). RESUMABLE.

For every question: 1 neutral main generation + 3 perturbed (P1/P2/P3), full
observability captured to the HDF5 cache, each graded by alias-EM. Combined with
the E1 know-score band, each generation is assigned a stratum (KH/KC/UH/UC/MID)
and a per-perturbation strata table is written.

Perturbation context (distractor for P1, the 4 exemplars + 2 corruptions for P2)
is sampled deterministically from OTHER questions' gold answers (seeded per qid).

Same GPU policy as E1: refuses the full run on CPU; ``--smoke-n N`` validates the
loop on CPU with the 0.5B model (output suffixed ``_smoke``, not a result).

Run (GPU):  .venv/bin/python scripts/run_e2_perturbed.py --config configs/e1.yaml
Run (CPU validation): .venv/bin/python scripts/run_e2_perturbed.py --smoke-n 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm  # noqa: E402

from src.hv2.config import REPO_ROOT, SMOKE_MODEL, HV2Config, cuda_available, load_config, set_all_seeds  # noqa: E402
from src.hv2.datasets_hv2 import load_questions  # noqa: E402
from src.hv2.gpulock import GpuLock  # noqa: E402
from src.hv2.grading import GraderAudit, grade  # noqa: E402
from src.hv2.harness import chat_prompt, generate, load_model  # noqa: E402
from src.hv2.hdf5_store import H5Store  # noqa: E402
from src.hv2.perturb import build_perturbations  # noqa: E402
from src.hv2.strata import strata_table  # noqa: E402


def _model_tag(name: str) -> str:
    return name.split("/")[-1]


def _build_context(qid: str, idx: int, df, pool_idx: list[int], seed: int):
    """Deterministically pick a distractor + 4 exemplars + 2 corruptions."""
    rng = random.Random(f"{seed}:{qid}")
    own = set(df.iloc[idx]["gold_aliases"])
    others = [p for p in pool_idx if p != idx]
    rng.shuffle(others)

    def alias_of(j):
        al = df.iloc[j]["gold_aliases"]
        return al[0] if al else "unknown"

    distractor = next((alias_of(p) for p in others if alias_of(p) not in own), "the other option")
    # need 4 exemplars + 2 corruption distractors; cycle if the pool is tiny (smoke)
    pick = (others * 6)[:6] if others else [idx] * 6
    exemplars = [(df.iloc[p]["question"], alias_of(p)) for p in pick[:4]]
    exemplar_distractors = [alias_of(p) for p in pick[4:6]] or ["something else", "another thing"]
    return distractor, exemplars, exemplar_distractors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke-n", type=int, default=0)
    ap.add_argument("--questions", default="data/e1_questions.parquet")
    args = ap.parse_args()

    smoke = args.smoke_n > 0
    if smoke:
        cfg = HV2Config(model_name=SMOKE_MODEL, quantization="none", max_new_tokens=24)
    else:
        cfg = load_config(args.config or REPO_ROOT / "configs" / "e1.yaml")
    set_all_seeds(cfg.seed)

    if cfg.device == "cpu" and not smoke:
        print(
            "[e2] REFUSING full perturbed-generation run on CPU — GPU job. "
            "No CUDA (BLOCKERS.md B1). Use --smoke-n 3 to validate on CPU."
        )
        return 2

    tag = _model_tag(cfg.model_name) + ("_smoke" if smoke else "")
    cache_path = REPO_ROOT / "data" / "cache" / f"e2_{tag}.h5"
    jsonl_path = REPO_ROOT / "data" / f"strata_records_{tag}.jsonl"
    parquet_path = REPO_ROOT / "data" / f"strata_{tag}.parquet"
    table_path = REPO_ROOT / "results" / f"e2_{tag}" / "strata_table.json"

    # E1 know-score bands (per question). Required to assign strata.
    e1_parquet = REPO_ROOT / "data" / f"knowledge_labels_{tag}.parquet"
    if not e1_parquet.exists():
        print(f"[e2] missing {e1_parquet} — run E1 (run_e1_knowledge.py) first.")
        return 3
    import pandas as pd

    e1 = pd.read_parquet(e1_parquet).set_index("qid")["n_correct_probes"].to_dict()

    df = load_questions(REPO_ROOT / args.questions).reset_index(drop=True)
    if smoke:
        df = df.head(args.smoke_n).reset_index(drop=True)
    pool_idx = list(range(len(df)))

    done = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["qid"])
            except Exception:  # noqa: BLE001
                pass
    todo = df[~df["qid"].isin(done)]
    print(f"[e2] model={cfg.model_name} device={cfg.device} smoke={smoke}; {len(todo)}/{len(df)} to process")

    audit = GraderAudit()
    t0 = time.time()
    with GpuLock(), H5Store(cache_path) as store, open(jsonl_path, "a") as jf:
        model, tok = load_model(cfg)
        for idx, row in tqdm(list(todo.iterrows()), total=len(todo), desc="e2-perturbed"):
            qid, question, aliases = row["qid"], row["question"], row["gold_aliases"]
            n_correct = e1.get(qid)
            if n_correct is None:
                continue  # no E1 band -> cannot stratify; skip honestly
            distractor, exemplars, ex_distract = _build_context(qid, idx, df, pool_idx, cfg.seed)

            gens = [("neutral", question, True)] + [
                (p.perturb_type, p.text, p.greedy)
                for p in build_perturbations(question, distractor, exemplars, ex_distract)
            ]
            for gen_type, text, greedy in gens:
                key = f"{tag}/{qid}/{gen_type}"
                if store.has(key):
                    rec = store.read(key)
                else:
                    rec = generate(model, tok, chat_prompt(tok, text), cfg, key=key, qid=qid,
                                   kind=gen_type, gold_aliases=aliases, greedy=greedy)
                    store.write(rec)
                correct = grade(rec.text, aliases, qid=qid, audit=audit)
                jf.write(json.dumps({
                    "qid": qid, "perturb_type": gen_type, "correct": correct,
                    "n_correct_probes": int(n_correct), "source": row["source"],
                }) + "\n")
            jf.flush()

    records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    import pandas as pd

    pd.DataFrame(records).to_parquet(parquet_path)
    table = strata_table([r for r in records if r["perturb_type"] != "neutral"])
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps(table, indent=2))
    audit.dump_sample(REPO_ROOT / "results" / f"e2_{tag}" / "grading_audit.csv", n=50)

    kh = table["overall"]["KH"]
    print(f"[e2] DONE {len(records)} records in {time.time()-t0:.1f}s; strata={table['overall']}")
    print(f"[e2] KH (knows-but-hallucinates) overall = {kh}; table -> {table_path}")
    if not smoke and kh < 300:
        print(f"[e2] NOTE: KH={kh} < 300 — see REPORT.md for widening options (do not silently change protocol).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
