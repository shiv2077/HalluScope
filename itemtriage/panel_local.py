"""Local panel inference on the GPU: one forward pass per item and option order. Label-free.

Runs in .venv under hv2's GpuLock. For each item and each order (original and one cyclic shift), the model
reads the direct-letter prompt through its chat template, and the log-softmax at the answer position is
pooled over each letter's single-token variants ("A", " A"). Raw rows go to
data/itemtriage/raw/local_<slug>.jsonl with the top 100 tokens; a pass resumes per (item_id, shift).
Batch size is 1 so padding never shifts position ids. No full-vocabulary logits are stored.
"""

from __future__ import annotations

import json
import math
import sys
import time

import pandas as pd

from . import paths, prompts

sys.path.insert(0, str(paths.ROOT / "src"))

MODELS = {
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct", "nf4", "Qwen"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "fp16", "Qwen"),
    "phi-3.5-mini": ("microsoft/Phi-3.5-mini-instruct", "nf4", "Microsoft"),
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "fp16", "Hugging Face"),
    "granite-3.1-2b": ("ibm-granite/granite-3.1-2b-instruct", "nf4", "IBM"),
    "olmo-2-1b": ("allenai/OLMo-2-0425-1B-Instruct", "fp16", "AI2"),  # reserve
}
PANEL_LOCAL = ("qwen2.5-3b", "qwen2.5-1.5b", "phi-3.5-mini", "smollm2-1.7b", "granite-3.1-2b")
SHIFTS = (0, 1)
TOPK = 100


def raw_path(slug: str):
    return paths.RAW / f"local_{slug}.jsonl"


def done_keys(slug: str) -> set:
    p = raw_path(slug)
    if not p.exists():
        return set()
    out = set()
    for x in p.read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            out.add((r["item_id"], r["shift"]))
    return out


def _chat_ids(tok, text: str) -> list[int]:
    enc = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
    if isinstance(enc, dict) or hasattr(enc, "keys"):
        enc = enc["input_ids"]
    return list(enc[0] if enc and isinstance(enc[0], list) else enc)


def letter_token_ids(tok, n: int) -> dict[str, list[int]]:
    out = {}
    for L in paths.LETTERS[:n]:
        ids = set()
        for v in (L, " " + L):
            e = tok.encode(v, add_special_tokens=False)
            if len(e) == 1:
                ids.add(e[0])
        if not ids:
            raise RuntimeError(f"no single-token encoding for letter {L}")
        out[L] = sorted(ids)
    return out


def run(slug: str, ids: list[str], tag: str) -> dict:
    import torch

    from hv2.config import HV2Config
    from hv2.gpulock import GpuLock
    from hv2.harness import load_model

    name, quant, _family = MODELS[slug]
    items = pd.read_parquet(paths.ITEMS).set_index("item_id")
    todo = [(i, s) for i in sorted(ids) for s in SHIFTS if (i, s) not in done_keys(slug)]
    if not todo:
        return {"slug": slug, "tag": tag, "new_rows": 0}
    torch.manual_seed(42)
    t0 = time.time()
    with GpuLock(wait=True):
        model, tok = load_model(HV2Config(model_name=name, quantization=quant))
        lids = letter_token_ids(tok, 4)
        paths.RAW.mkdir(parents=True, exist_ok=True)
        with open(raw_path(slug), "a") as fh, torch.inference_mode():
            for k, (iid, shift) in enumerate(todo, 1):
                it = items.loc[iid].to_dict() | {"item_id": iid}
                text, order = prompts.mc_prompt(it, shift)
                prompts.check_prompt(text, it, order)
                x = torch.tensor([_chat_ids(tok, text)], device=model.device)
                lp = torch.log_softmax(model(input_ids=x).logits[0, -1].float(), dim=-1)
                letter = [float(torch.logsumexp(lp[lids[L]], dim=0)) for L in paths.LETTERS[: it["n_options"]]]
                top_lp, top_id = torch.topk(lp, TOPK)
                top1 = int(top_id[0])
                row = {"item_id": iid, "shift": shift, "order": order, "letter_logprobs": [round(v, 6) for v in letter],
                       "letter_mass": round(sum(math.exp(v) for v in letter), 6),
                       "top1_is_letter": any(top1 in v for v in lids.values()),
                       "top_ids": top_id.tolist(), "top_logprobs": [round(float(v), 5) for v in top_lp],
                       "n_prompt_tokens": int(x.shape[1])}
                fh.write(json.dumps(row) + "\n")
                if k % 500 == 0:
                    fh.flush()
                    print(f"{slug} {tag}: {k}/{len(todo)} ({k / (time.time() - t0):.1f}/s)", file=sys.stderr, flush=True)
        del model
        torch.cuda.empty_cache()
    return {"slug": slug, "tag": tag, "new_rows": len(todo), "seconds": round(time.time() - t0, 1)}
