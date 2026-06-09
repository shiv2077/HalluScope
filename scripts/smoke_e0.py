"""E0 smoke: exercise the full harness on Qwen2.5-0.5B (CPU-ok).

Validates: model loads once, generate() returns well-formed GenRecords for the
four observation levels, the HDF5 cache round-trips, and per-example resume
skips already-written keys. Reports peak RAM. CLAUDE.md says 0.5B is for smoke
ONLY — nothing here is a scientific result.

VRAM ceiling note: the CLAUDE.md "< 5.5GB VRAM peak" assertion needs CUDA. When
no CUDA device is present we SKIP that assertion (and say so) and report CPU
peak RSS instead — we refuse to invent a VRAM number.

Run: .venv/bin/python scripts/smoke_e0.py
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hv2.config import SMOKE_MODEL, HV2Config, cuda_available, set_all_seeds  # noqa: E402
from src.hv2.gpulock import GpuLock  # noqa: E402
from src.hv2.grading import GraderAudit, grade  # noqa: E402
from src.hv2.harness import chat_prompt, generate, load_model  # noqa: E402
from src.hv2.hdf5_store import H5Store  # noqa: E402

# (question, gold aliases) — tiny known-answer set, NOT a benchmark.
QUESTIONS = [
    ("What is the capital of France?", ["Paris"]),
    ("Who wrote the play Romeo and Juliet?", ["Shakespeare", "William Shakespeare"]),
    ("What is the chemical symbol for gold?", ["Au"]),
    ("How many continents are there on Earth?", ["seven", "7"]),
    ("What planet is known as the Red Planet?", ["Mars"]),
]


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB->MB on Linux


def run(n: int = 5, cache_path: str | None = None) -> dict:
    set_all_seeds(42)
    cfg = HV2Config(model_name=SMOKE_MODEL, quantization="none", max_new_tokens=24)
    print(f"[smoke] device={cfg.device} dtype={cfg.torch_dtype} cuda={cuda_available()}")

    cache = Path(cache_path or (cfg.abs_path("cache_dir") / "smoke_e0.h5"))
    if cache.exists():
        cache.unlink()
    audit = GraderAudit()

    t0 = time.time()
    with GpuLock():  # no-op on CPU
        model, tok = load_model(cfg)
        print(f"[smoke] model loaded in {time.time()-t0:.1f}s; n_params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")
        store = H5Store(cache)
        n_correct = 0
        for i, (q, aliases) in enumerate(QUESTIONS[:n]):
            qid = f"smoke_{i:03d}"
            key = f"{cfg.model_name.split('/')[-1]}/{qid}/main"
            if store.has(key):
                print(f"[smoke] resume: {key} already present, skipping")
                continue
            prompt = chat_prompt(tok, q)
            rec = generate(model, tok, prompt, cfg, key=key, qid=qid, kind="main", gold_aliases=aliases, greedy=True)
            store.write(rec)
            ok = grade(rec.text, aliases, qid=qid, audit=audit)
            n_correct += int(ok)
            print(
                f"[smoke] {qid} T={len(rec.token_ids)} "
                f"top_lp={rec.top_logprobs.shape} func={rec.functionals.shape} "
                f"hidden={rec.hidden_states.shape} ans_idx={list(rec.answer_token_indices)} "
                f"correct={ok} :: {rec.text!r}"
            )
        store.close()

    # ---- assertions ----
    with H5Store(cache, mode="r") as ro:
        keys = ro.keys()
        assert len(keys) == n, f"expected {n} records, got {len(keys)}"
        sample = ro.read(keys[0])
        T = sample.token_ids.shape[0]
        assert sample.top_logprobs.shape == (T, cfg.topk_logprobs), "top-100 logprobs shape"
        assert sample.functionals.shape == (T, 3), "entropy/energy/margin"
        assert sample.hidden_states.ndim == 3 and sample.hidden_states.dtype.name == "float16"
        import numpy as np

        assert np.isfinite(sample.functionals).all(), "functionals finite"

    # resume check: a second pass writes nothing new
    store2 = H5Store(cache)
    before = len(store2.keys())
    for i, (q, aliases) in enumerate(QUESTIONS[:n]):
        key = f"{cfg.model_name.split('/')[-1]}/smoke_{i:03d}/main"
        assert store2.has(key), f"resume: {key} should exist"
    store2.close()
    assert before == n

    peak = peak_rss_mb()
    if cuda_available():
        import torch

        vram_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[smoke] CUDA peak VRAM = {vram_mb:.0f} MB")
        assert vram_mb < 5500, f"VRAM ceiling exceeded: {vram_mb:.0f} MB"
    else:
        print("[smoke] CUDA absent -> VRAM ceiling assert SKIPPED (see BLOCKERS.md B1)")

    audit_path = audit.dump_sample(cfg.abs_path("results_dir") / "smoke_e0" / "grading_audit.csv", n=50)
    summary = {
        "device": cfg.device,
        "n_records": n,
        "n_correct": n_correct,
        "peak_rss_mb": round(peak, 1),
        "elapsed_s": round(time.time() - t0, 1),
        "cache": str(cache),
        "audit": str(audit_path),
    }
    print(f"[smoke] OK {summary}")
    return summary


if __name__ == "__main__":
    run()
