"""H2 capture: hidden states of one local model reading the item with its keyed answer. Label-free.

Qwen2.5-1.5B-Instruct (fp16) reads the direct-letter prompt with the keyed letter appended as the assistant
turn, in one teacher-forced pass. Every layer is cached at two positions: the keyed letter token and the last
prompt token. Cache: data/itemtriage/cache/hidden_qwen2.5-1.5b.h5, dataset H[item, position, layer, dim] float16,
resumable through a `done` mask. Runs in .venv under GpuLock.

Preregistered features (docs/PREREG.md), derived on CPU from the cache: at the key token, the mean over the
middle third of layers and the layers at relative depth 0.5 and 0.75; at the last prompt token, the mean over
the middle third. No other layer is used.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import paths, prompts, thermal

MODEL = ("Qwen/Qwen2.5-1.5B-Instruct", "fp16")
SLUG = "qwen2.5-1.5b"
CACHE = paths.DATA / "cache" / f"hidden_{SLUG}.h5"
FEATURES = paths.DATA / "cache" / f"hidden_features_{SLUG}.npy"
FEATURES_META = paths.RESULTS / f"hidden_features_{SLUG}.json"
POSITIONS = ("key_token", "last_prompt_token")


def _ids(tok, messages, gen: bool) -> list[int]:
    enc = tok.apply_chat_template(messages, add_generation_prompt=gen, tokenize=True)
    if hasattr(enc, "keys"):
        enc = enc["input_ids"]
    return list(enc[0] if enc and isinstance(enc[0], list) else enc)


def capture() -> dict:
    import h5py
    import torch

    from .gpu import GpuLock, load_model

    items = pd.read_parquet(paths.ITEMS).sort_values("item_id").reset_index(drop=True)
    n = len(items)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with GpuLock(wait=True):
        model, tok = load_model(*MODEL)
        L, D = model.config.num_hidden_layers, model.config.hidden_size
        with h5py.File(CACHE, "a") as h5:
            if "H" not in h5:
                h5.create_dataset("H", shape=(n, 2, L, D), dtype="float16", chunks=(1, 2, L, D))
                h5.create_dataset("done", data=np.zeros(n, bool))
                h5.create_dataset("item_id", data=np.array(items["item_id"].tolist(), dtype="S64"))
            assert [x.decode() for x in h5["item_id"][:]] == items["item_id"].tolist()
            done = h5["done"][:]
            todo = np.flatnonzero(~done)
            with torch.inference_mode():
                for k, i in enumerate(todo, 1):
                    it = items.iloc[i].to_dict()
                    text, order = prompts.mc_prompt(it, 0)
                    prompts.check_prompt(text, it, order)
                    user = [{"role": "user", "content": text}]
                    p_ids = _ids(tok, user, True)
                    f_ids = _ids(tok, user + [{"role": "assistant", "content": paths.LETTERS[it["key_index"]]}], False)
                    assert f_ids[: len(p_ids)] == p_ids, f"{it['item_id']}: chat template prefix mismatch"
                    key_pos, last_pos = len(p_ids), len(p_ids) - 1
                    letter = tok.decode([f_ids[key_pos]]).strip()
                    assert letter == paths.LETTERS[it["key_index"]], f"{it['item_id']}: key token is {letter!r}"
                    x = torch.tensor([f_ids[: key_pos + 1]], device=model.device)
                    hs = model(input_ids=x, output_hidden_states=True).hidden_states[1:]  # drop the embedding layer
                    H = torch.stack([torch.stack([h[0, key_pos], h[0, last_pos]]) for h in hs], dim=1)  # [2, L, D]
                    h5["H"][i] = H.to(torch.float16).cpu().numpy()
                    h5["done"][i] = True
                    thermal.pace(k, "hidden")
                    if k % 500 == 0:
                        h5.flush()
                        print(f"hidden: {k}/{len(todo)}", file=sys.stderr, flush=True)
            n_done = int(h5["done"][:].sum())
        del model
        torch.cuda.empty_cache()
    return {"items": n, "done": n_done, "layers": L, "dim": D}


def layer_plan(L: int) -> dict:
    third = list(range(L // 3, 2 * L // 3))
    return {"middle_third": third, "depth_0.5": round(0.5 * L) - 1, "depth_0.75": round(0.75 * L) - 1}


def features() -> np.ndarray:
    """[n_items, 4 * dim] float32 in item_id order: key mid-third mean, key depth 0.5, key depth 0.75, last mid-third mean."""
    import h5py

    with h5py.File(CACHE, "r") as h5:
        if not h5["done"][:].all():
            raise RuntimeError("hidden-state capture is incomplete")
        H = h5["H"]
        n, _, L, D = H.shape
        lp = layer_plan(L)
        out = np.empty((n, 4 * D), np.float32)
        for s in range(0, n, 256):
            blk = H[s:s + 256].astype(np.float32)
            out[s:s + 256] = np.concatenate([blk[:, 0, lp["middle_third"]].mean(1), blk[:, 0, lp["depth_0.5"]],
                                             blk[:, 0, lp["depth_0.75"]], blk[:, 1, lp["middle_third"]].mean(1)], axis=1)
    np.save(FEATURES, out)
    import json

    FEATURES_META.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_META.write_text(json.dumps({"model": MODEL[0], "layers": int(L), "dim": int(D), "layer_plan": lp,
                                         "positions": list(POSITIONS), "n_items": int(n)}, indent=1))
    return out
