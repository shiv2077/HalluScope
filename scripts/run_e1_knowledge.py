"""E1 — knowledge labeling (the long GPU job). RESUMABLE.

For every question: run the 8 neutral probes (probes.py), grade by alias-EM,
record n_correct/know_score. GenRecords stream to an HDF5 cache (resume = skip
keys already present); per-question labels append to a JSONL checkpoint and are
consolidated to data/knowledge_labels_{model}.parquet at the end.

GPU policy (CLAUDE.md): this is a GPU job. On a CPU-only box it REFUSES the full
run (6000*8 fwd passes is intractable on CPU and we won't fake it). Use
``--smoke-n N`` to validate the loop on N questions with the 0.5B model on CPU;
output is suffixed ``_smoke`` and is NOT a scientific result.

Run (GPU):  .venv/bin/python scripts/run_e1_knowledge.py --config configs/e1.yaml
Run (CPU validation): .venv/bin/python scripts/run_e1_knowledge.py --smoke-n 3
"""

from __future__ import annotations

import argparse
import json
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
from src.hv2.probes import build_probes, know_band, know_score  # noqa: E402


def _model_tag(name: str) -> str:
    return name.split("/")[-1]


def _load_done(jsonl: Path) -> set[str]:
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                done.add(json.loads(line)["qid"])
            except Exception:  # noqa: BLE001
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke-n", type=int, default=0, help="run N questions on CPU and suffix output _smoke")
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
            "[e1] REFUSING full knowledge-labeling run on CPU — this is a GPU job "
            "(6000 questions * 8 probes). No CUDA device (see BLOCKERS.md B1).\n"
            "     Restore the NVIDIA driver, then re-run this exact command.\n"
            "     To validate the pipeline on CPU instead: --smoke-n 3"
        )
        return 2

    tag = _model_tag(cfg.model_name) + ("_smoke" if smoke else "")
    cache_path = REPO_ROOT / "data" / "cache" / f"e1_{tag}.h5"
    jsonl_path = REPO_ROOT / "data" / f"knowledge_labels_{tag}.jsonl"
    parquet_path = REPO_ROOT / "data" / f"knowledge_labels_{tag}.parquet"

    df = load_questions(REPO_ROOT / args.questions)
    if smoke:
        df = df.head(args.smoke_n)
    done = _load_done(jsonl_path)
    todo = df[~df["qid"].isin(done)]
    print(f"[e1] model={cfg.model_name} device={cfg.device} cuda={cuda_available()} smoke={smoke}")
    print(f"[e1] {len(df)} questions, {len(done)} already done, {len(todo)} to label -> {parquet_path}")

    audit = GraderAudit()
    t0 = time.time()
    with GpuLock(), H5Store(cache_path) as store, open(jsonl_path, "a") as jf:
        model, tok = load_model(cfg)
        for _, row in tqdm(todo.iterrows(), total=len(todo), desc="e1-knowledge"):
            qid, question, aliases = row["qid"], row["question"], row["gold_aliases"]
            flags = []
            for probe in build_probes(question):
                key = f"{tag}/{qid}/{probe.probe_id}"
                if store.has(key):
                    rec = store.read(key)
                else:
                    prompt = chat_prompt(tok, probe.text)
                    rec = generate(
                        model, tok, prompt, cfg, key=key, qid=qid, kind=probe.probe_id,
                        gold_aliases=aliases, greedy=probe.greedy, temperature=probe.temperature, seed=probe.seed,
                    )
                    store.write(rec)
                flags.append(grade(rec.text, aliases, qid=qid, audit=audit))
            n_correct = sum(flags)
            jf.write(json.dumps({
                "qid": qid, "source": row["source"], "model": cfg.model_name,
                "n_correct_probes": n_correct, "know_score": know_score(flags),
                "band": know_band(n_correct), "probe_correct": flags,
            }) + "\n")
            jf.flush()

    # consolidate JSONL -> parquet (full set, incl. previously-done)
    import pandas as pd

    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    out = pd.DataFrame(rows)
    out.to_parquet(parquet_path)
    audit.dump_sample(REPO_ROOT / "results" / f"e1_{tag}" / "grading_audit.csv", n=50)

    bands = out["band"].value_counts().to_dict()
    print(f"[e1] DONE {len(out)} labels in {time.time()-t0:.1f}s; bands={bands}; parquet={parquet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
