"""Public response matrix for Study 2 (docs/PREREG_2.md): 395 Open LLM Leaderboard models on MMLU and GSM8K.

Correctness comes from the tinyBenchmarks release (`lb.pickle`, binary correct-against-key per model and row).
Its rows follow the Open LLM Leaderboard v1 detail files, whose `example` text is the join key to MMLU-Redux.
Platinum GSM8K supplies a second ground truth. Raw downloads go to data/itemtriage/raw/matrix/ (not committed).
`build()` writes small committed tables to data/itemtriage/matrix/, so everything downstream runs on CPU offline.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pickle
import re
import urllib.request

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from . import paths, splits
from .ingest import normalize, text_hash

RAW = paths.RAW / "matrix"
OUT = paths.DATA / "matrix"
CORRECT = OUT / "correctness.parquet"
GSM_CORRECT = OUT / "gsm8k_correctness.parquet"
GSM_ITEMS = OUT / "gsm8k_items.parquet"
GSM_LABELS = OUT / "gsm8k_labels.parquet"
MODELS = OUT / "models.parquet"
MANIFEST = OUT / "manifest.json"

TB_COMMIT = "ce9f7024248685ff1a5b507b955051e9a0e6e074"
TB_URL = f"https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/{TB_COMMIT}/tutorials/data/lb.pickle"
TB_SHA256 = "34f44d6a819512ef74d00a95288d252fa679288a10ca167cd97fdbc3aae66437"
DETAILS_REPO = "open-llm-leaderboard-old/details_alignment-handbook__zephyr-7b-sft-full"
DETAILS_REV = "1325769d0f484a510bb435fc900bba469efbc804"
PLATINUM_REPO = "madrylab/platinum-bench"
PLATINUM_REV = "51920a33bfb4620c789729ace14141e87a14969b"
PLATINUM_POSITIVE = ("rejected", "revised")
CLUSTER_AGREE = 0.95   # one model per cluster at this pairwise agreement (the model filter)
LINEAGE_AGREE = 0.90   # coarser clusters for the lineage-diverse stratum
MC_RE = re.compile(r"(?s)^(.*)\nA\. (.*)\nB\. (.*)\nC\. (.*)\nD\. (.*?)\nAnswer:\s*$")


# ------------------------------------------------------------------ download (network, run once)
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_load(path) -> dict:
    """Unpickle lb.pickle allowing only numpy's array reconstruction. Anything else raises."""
    allowed = {"_frombuffer": "numpy._core.numeric", "dtype": "numpy"}

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if name in allowed and module.split(".")[0] == "numpy":
                return getattr(importlib.import_module(allowed[name]), name)
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    with open(path, "rb") as fh:
        return Restricted(fh).load()


def _detail_file(files: list[str], task: str) -> list[str]:
    return sorted(f for f in files if f"|{task}|" in f and f.endswith(".parquet"))


def download() -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    RAW.mkdir(parents=True, exist_ok=True)
    tb = RAW / "lb.pickle"
    if not tb.exists():
        urllib.request.urlretrieve(TB_URL, tb)
    if sha256(tb) != TB_SHA256:
        raise RuntimeError(f"{tb} sha256 mismatch; refusing to use it")
    files = [s.rfilename for s in HfApi().dataset_info(DETAILS_REPO, revision=DETAILS_REV).siblings]
    lb = safe_load(tb)
    tasks = {k.split("harness_")[1].rsplit("_", 1)[0].replace("hendrycksTest_", "hendrycksTest-"): k for k in lb["data"]
             if k.startswith("harness_hendrycksTest_") or k == "harness_gsm8k_5"}
    got = {}
    for task, key in sorted(tasks.items()):
        n = lb["data"][key]["correctness"].shape[0]
        for f in reversed(_detail_file(files, task)):
            p = hf_hub_download(DETAILS_REPO, f, repo_type="dataset", revision=DETAILS_REV, local_dir=RAW / "details")
            if len(pd.read_parquet(p, columns=["example"])) == n:
                got[key] = f
                break
        else:
            raise RuntimeError(f"no complete detail file for {task}")
    plat = hf_hub_download(PLATINUM_REPO, "gsm8k/test-00000-of-00001.parquet", repo_type="dataset", revision=PLATINUM_REV,
                           local_dir=RAW / "platinum")
    (RAW / "detail_files.json").write_text(json.dumps(got, indent=1, sort_keys=True))
    return {"lb_sha256": TB_SHA256, "detail_files": len(got), "platinum": str(plat)}


# ------------------------------------------------------------------ build (offline)
def model_names(lb: dict) -> list[str]:
    return [m.split("details_", 1)[1].replace("__", "/", 1) for m in lb["models"]]


def parse_mc(example: str) -> tuple[str, list[str]] | None:
    m = MC_RE.match(example)
    return (m.group(1).strip(), [g.strip() for g in m.groups()[1:]]) if m else None


def gsm_question(example: str) -> str:
    return normalize(str(example).split("Question:")[-1].split("Answer:")[0])


def join_mmlu(items: pd.DataFrame, rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """items: item_id, subject, text_hash. rows: subject, pos, th (one per leaderboard row).
    A duplicate MMLU text maps to its first leaderboard row. Returns item_id, subject, pos and join counts."""
    first = rows.dropna(subset=["th"]).sort_values(["subject", "pos"]).drop_duplicates(["subject", "th"])
    j = items.merge(first, left_on=["subject", "text_hash"], right_on=["subject", "th"], how="left")
    dup = rows.dropna(subset=["th"]).groupby(["subject", "th"]).size()
    n_dup = int(items.set_index(["subject", "text_hash"]).index.isin(dup[dup > 1].index).sum())
    matched = j[j["pos"].notna()][["item_id", "subject", "pos"]].astype({"pos": int})
    return matched, {"items": len(items), "matched": len(matched), "unmatched": sorted(j[j["pos"].isna()]["item_id"]),
                     "matched_to_duplicate_text": n_dup}


def clusters(X: np.ndarray, agree: float) -> np.ndarray:
    """Average-linkage clusters of models (columns of X) whose pairwise agreement is at least `agree`."""
    A = (X[:, :, None] == X[:, None, :]).mean(axis=0)
    D = squareform(1.0 - A, checks=False)
    return fcluster(linkage(D, "average"), 1.0 - agree, "distance")


def build() -> dict:
    lb = safe_load(RAW / "lb.pickle")
    names = model_names(lb)
    detail = json.loads((RAW / "detail_files.json").read_text())
    items = pd.read_parquet(paths.ITEMS)[["item_id", "subject", "text_hash", "question", "options"]]
    rows, Ys = [], {}
    for subj in sorted(items["subject"].unique()):
        key = f"harness_hendrycksTest_{subj}_5"
        ex = pd.read_parquet(RAW / "details" / detail[key], columns=["example"])["example"]
        for pos, e in enumerate(ex):
            q = parse_mc(e)
            rows.append((subj, pos, text_hash(*q) if q else None))
        Ys[subj] = lb["data"][key]["correctness"]
    matched, join = join_mmlu(items, pd.DataFrame(rows, columns=["subject", "pos", "th"]))
    Y = np.vstack([Ys[s][p] for s, p in zip(matched["subject"], matched["pos"])]).astype(np.int8)
    corr = pd.concat([matched[["item_id"]].reset_index(drop=True), pd.DataFrame(Y, columns=names)], axis=1)

    # ability and clusters on calibration-fit subjects, all their MMLU rows (label-free: correctness against the key)
    fold = splits.fold_of(splits.load())
    cal_subj = sorted(set(items.loc[items["item_id"].map(fold) == "calibration_fit", "subject"]))
    Xc = np.vstack([lb["data"][f"harness_hendrycksTest_{s}_5"]["correctness"] for s in cal_subj])
    ability = Xc.mean(axis=0)
    c95, c90 = clusters(Xc, CLUSTER_AGREE), clusters(Xc, LINEAGE_AGREE)
    mod = pd.DataFrame({"model": names, "ability": ability, "cluster": c95, "lineage": c90})
    rep = mod.sort_values("model").drop_duplicates("cluster")["model"]
    mod["representative"] = mod["model"].isin(set(rep))
    params = RAW / "params.json"
    pj = json.loads(params.read_text()) if params.exists() else {}
    mod["params"] = mod["model"].map(lambda m: pj.get(m))

    # GSM8K: every leaderboard row, plus the platinum subset as labels
    g = pd.read_parquet(RAW / "details" / detail["harness_gsm8k_5"], columns=["example"])
    gid = [f"gsm8k:{i:04d}" for i in range(len(g))]
    gq = [gsm_question(e) for e in g["example"]]
    gsm_items = pd.DataFrame({"item_id": gid, "question": [str(e).split("Question:")[-1].split("Answer:")[0].strip() for e in g["example"]]})
    gsm_corr = pd.concat([pd.DataFrame({"item_id": gid}), pd.DataFrame(lb["data"]["harness_gsm8k_5"]["correctness"].astype(np.int8),
                                                                       columns=names)], axis=1)
    plat = pd.read_parquet(next((RAW / "platinum").rglob("*.parquet")))
    pos_of = {q: i for i, q in reversed(list(enumerate(gq)))}
    pi = plat["question"].map(lambda q: pos_of.get(normalize(q)))
    if pi.isna().any():
        raise RuntimeError(f"{int(pi.isna().sum())} platinum GSM8K items do not match a leaderboard row")
    gsm_labels = pd.DataFrame({"item_id": [gid[int(i)] for i in pi], "cleaning_status": plat["cleaning_status"].to_numpy(),
                               "flaw": plat["cleaning_status"].isin(PLATINUM_POSITIVE).astype(int).to_numpy()})

    OUT.mkdir(parents=True, exist_ok=True)
    corr.to_parquet(CORRECT, index=False)
    mod.to_parquet(MODELS, index=False)
    gsm_items.to_parquet(GSM_ITEMS, index=False)
    gsm_corr.to_parquet(GSM_CORRECT, index=False)
    gsm_labels.sort_values("item_id").to_parquet(GSM_LABELS, index=False)
    manifest = {"sources": {"tinybenchmarks": {"url": TB_URL, "sha256": TB_SHA256, "licence": "MIT"},
                            "leaderboard_details": {"repo": DETAILS_REPO, "revision": DETAILS_REV,
                                                    "use": "row text as the join key only", "files": detail},
                            "platinum_bench": {"repo": PLATINUM_REPO, "revision": PLATINUM_REV, "licence": "CC-BY-SA-4.0"}},
                "models": len(names), "representatives": int(mod["representative"].sum()),
                "lineages_among_representatives": int(mod.loc[mod["representative"], "lineage"].nunique()),
                "cluster_agreement": CLUSTER_AGREE, "lineage_agreement": LINEAGE_AGREE,
                "ability_subjects": cal_subj, "ability_rows": int(Xc.shape[0]), "mmlu_join": join,
                "gsm8k": {"rows": len(gid), "platinum_items": len(gsm_labels), "platinum_positives": int(gsm_labels["flaw"].sum())},
                "outputs": {p.name: sha256(p) for p in (CORRECT, MODELS, GSM_ITEMS, GSM_CORRECT, GSM_LABELS)}}
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def fetch_params() -> dict:
    """Parameter counts from model cards (safetensors metadata). Network. A labelled proxy for cost, never dollars."""
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi

    api = HfApi()
    names = model_names(safe_load(RAW / "lb.pickle"))

    def one(m):
        try:
            st = api.model_info(m, expand=["safetensors"]).safetensors
            return m, (int(st.total) if st else None)
        except Exception:
            return m, None

    with ThreadPoolExecutor(8) as ex:
        out = dict(ex.map(one, names))
    (RAW / "params.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    return {"known": sum(v is not None for v in out.values()), "models": len(out)}


# ------------------------------------------------------------------ optional: per-choice log-likelihoods (T2.2b)
PKEY = OUT / "p_key.parquet"
PKEY_RAW = RAW / "pkey"
DETAILS_ORGS = ("open-llm-leaderboard-old", "open-llm-leaderboard")
RETRIES = 5
PKEY_MIN_AGREE = 0.95   # keep a model's P(key) only if its detail run matches the matrix on right versus wrong


def _softmax_key(pred, key: int) -> tuple[float, int]:
    """P(key) under a softmax of the 4 option log-likelihoods, and 1 if the key is the argmax."""
    lp = np.asarray(list(pred), float)[:4]
    e = np.exp(lp - lp.max())
    return float(e[key] / e.sum()), int(np.argmax(lp) == key)


def fetch_model_pkey(model: str, need: pd.DataFrame) -> dict:
    """P(key) on the joined MMLU-Redux rows for one model, from its newest complete detail run per subject.
    need: item_id, subject, pos, key_index, rows (subject row count). Writes RAW/pkey/<model>.json. Resumable."""
    import time

    from huggingface_hub import HfApi, hf_hub_download

    out = PKEY_RAW / (model.replace("/", "__") + ".json")
    if out.exists():
        return json.loads(out.read_text())
    slug = "details_" + model.replace("/", "__", 1)
    for attempt in range(RETRIES):
        try:
            files = None
            for org in DETAILS_ORGS:
                try:
                    files = [s.rfilename for s in HfApi().dataset_info(f"{org}/{slug}").siblings]
                    repo = f"{org}/{slug}"
                    break
                except Exception:
                    continue
            if files is None:
                res = {"model": model, "status": "no details repo"}
                break
            p, tmp = {}, RAW / "tmp" / slug
            for subj, g in need.groupby("subject"):
                cand = _detail_file(files, f"hendrycksTest-{subj}")
                x = None
                for f in reversed(cand):
                    path = hf_hub_download(repo, f, repo_type="dataset", local_dir=tmp)
                    df = pd.read_parquet(path, columns=["predictions"])
                    if len(df) == int(g["rows"].iloc[0]):
                        x = df["predictions"].to_numpy()
                        break
                if x is None:
                    raise LookupError(f"no complete run for {subj}")
                for iid, pos, key in zip(g["item_id"], g["pos"], g["key_index"]):
                    p[iid] = _softmax_key(x[pos], int(key))
            res = {"model": model, "status": "ok", "repo": repo, "p_key": p}
            break
        except LookupError as e:
            res = {"model": model, "status": str(e)}
            break
        except Exception as e:
            res = {"model": model, "status": f"error: {type(e).__name__}"}
            time.sleep(30 * (attempt + 1))
    import shutil

    shutil.rmtree(RAW / "tmp" / slug, ignore_errors=True)
    PKEY_RAW.mkdir(parents=True, exist_ok=True)
    if not res["status"].startswith("error"):
        out.write_text(json.dumps(res, sort_keys=True))
    return res


def fetch_pkey(workers: int = 8, limit: int | None = None) -> dict:
    """Fetch P(key) for every representative model. Network, about 7 GB transient, resumable."""
    from concurrent.futures import ThreadPoolExecutor

    lb = safe_load(RAW / "lb.pickle")
    rows = {k.split("harness_hendrycksTest_")[1].rsplit("_", 1)[0]: v["correctness"].shape[0]
            for k, v in lb["data"].items() if k.startswith("harness_hendrycksTest_")}
    items = pd.read_parquet(paths.ITEMS)[["item_id", "subject", "text_hash", "key_index"]]
    detail = json.loads((RAW / "detail_files.json").read_text())
    rr = []
    for subj in sorted(items["subject"].unique()):
        ex = pd.read_parquet(RAW / "details" / detail[f"harness_hendrycksTest_{subj}_5"], columns=["example"])["example"]
        rr += [(subj, pos, text_hash(*q) if (q := parse_mc(e)) else None) for pos, e in enumerate(ex)]
    matched, _ = join_mmlu(items, pd.DataFrame(rr, columns=["subject", "pos", "th"]))
    need = matched.merge(items[["item_id", "key_index"]], on="item_id")
    need["rows"] = need["subject"].map(rows)
    reps = pd.read_parquet(MODELS).query("representative")["model"].tolist()[:limit]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda m: fetch_model_pkey(m, need), reps))
    return {"ok": sum(r["status"] == "ok" for r in res), "models": len(res)}


def build_pkey() -> dict:
    """Assemble committed p_key.parquet (float16) from the per-model raw files. Records coverage in the manifest."""
    corr = pd.read_parquet(CORRECT)
    reps = pd.read_parquet(MODELS).query("representative")["model"].tolist()
    cols, status, agree = {}, {}, {}
    for m in reps:
        f = PKEY_RAW / (m.replace("/", "__") + ".json")
        r = json.loads(f.read_text()) if f.exists() else {"status": "not fetched"}
        status[m] = r["status"]
        if r["status"] != "ok":
            continue
        pk = corr["item_id"].map(lambda i: r["p_key"][i][0]).to_numpy(float)
        right = corr["item_id"].map(lambda i: r["p_key"][i][1]).to_numpy()
        # share of items where the detail run and the tinyBenchmarks matrix agree on right versus wrong
        agree[m] = float(np.mean(right == corr[m].to_numpy()))
        if agree[m] >= PKEY_MIN_AGREE:
            cols[m] = pk.astype(np.float16)
    P = pd.DataFrame(cols)
    P.insert(0, "item_id", corr["item_id"])
    P.to_parquet(PKEY, index=False)
    a = np.array(list(agree.values()))
    man = json.loads(MANIFEST.read_text())
    man["p_key"] = {"models_fetched": len(agree), "models_kept": len(cols), "models": len(reps),
                    "min_agreement_with_matrix": PKEY_MIN_AGREE,
                    "agreement_quantiles": dict(zip(("min", "q10", "median"), np.quantile(a, [0, 0.1, 0.5]).round(4).tolist())) if len(a) else None,
                    "status_counts": pd.Series(status).value_counts().to_dict(), "sha256": sha256(PKEY)}
    MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True))
    return man["p_key"]
