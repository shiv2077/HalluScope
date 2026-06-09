"""Resumable HDF5 cache for observability records.

One HDF5 *group* per generation, keyed ``{model}/{qid}/{kind}`` (e.g.
``Qwen2.5-1.5B-Instruct/tqa_00042/probe3``). Per-example resume = "does the
group already exist?": a long run that dies mid-way is restarted and skips every
key already present.

Storage policy (CLAUDE.md HARD RULES):
  * NEVER store full-vocab logits. We keep top-100 logprobs + their token ids,
    plus three scalar functionals (entropy, energy, margin) computed online.
  * Hidden states are float16 and only at key positions.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

HIDDEN_KINDS = ("last_prompt", "first_answer", "last_answer", "exact_answer")


@dataclasses.dataclass
class GenRecord:
    """One generation's four observation levels.

    Shapes (T = #generated tokens, K = topk (100), A = #answer tokens,
    P = #captured positions, L = #layers incl. embeddings, H = hidden size):
      text                 : str                      (level a: text)
      token_ids            : int64  [T]
      top_logprobs         : float32 [T, K]           (level b: top-k logprobs)
      top_token_ids        : int64  [T, K]
      functionals          : float32 [T, 3]           (level c: entropy,energy,margin)
      hidden_states        : float16 [P, L, H]        (level d: hidden states)
      hidden_pos_index     : int64  [P]   abs token index of each captured pos
      hidden_pos_kind      : list[str] len P          (subset of HIDDEN_KINDS)
      answer_token_indices : int64  [A]   indices (into generated tokens) of the
                                          exact-answer span
    """

    key: str
    prompt: str
    text: str
    token_ids: np.ndarray
    top_logprobs: np.ndarray
    top_token_ids: np.ndarray
    functionals: np.ndarray
    hidden_states: np.ndarray
    hidden_pos_index: np.ndarray
    hidden_pos_kind: list[str]
    answer_token_indices: np.ndarray
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    FUNCTIONAL_NAMES = ("entropy", "energy", "margin")

    def validate(self) -> None:
        T = self.token_ids.shape[0]
        assert self.top_logprobs.shape == (T, self.top_logprobs.shape[1]), "top_logprobs [T,K]"
        assert self.top_token_ids.shape == self.top_logprobs.shape, "top_token_ids must match logprobs"
        assert self.functionals.shape == (T, 3), f"functionals must be [T,3], got {self.functionals.shape}"
        P = self.hidden_states.shape[0]
        assert self.hidden_pos_index.shape == (P,), "hidden_pos_index [P]"
        assert len(self.hidden_pos_kind) == P, "hidden_pos_kind len P"
        assert np.isfinite(self.functionals).all(), "functionals must be finite"


class H5Store:
    """Append/resume-friendly HDF5 store. Open in 'a' (default) or 'r'."""

    def __init__(self, path: str | Path, mode: str = "a"):
        self.path = Path(path)
        if mode != "r":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self._f: h5py.File | None = None

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.path, self.mode)
        return self._f

    def has(self, key: str) -> bool:
        """True if a (fully written) record exists for ``key`` — the resume check."""
        f = self._file()
        return key in f and f[key].attrs.get("_complete", False)

    def keys(self) -> list[str]:
        out: list[str] = []
        f = self._file()

        def _visit(name, obj):
            if isinstance(obj, h5py.Group) and obj.attrs.get("_complete", False):
                out.append(name)

        f.visititems(_visit)
        return out

    def write(self, rec: GenRecord) -> None:
        rec.validate()
        f = self._file()
        if rec.key in f:
            del f[rec.key]  # overwrite a partial/aborted write
        g = f.require_group(rec.key)
        # short text -> attrs; arrays -> compressed datasets
        g.attrs["prompt"] = rec.prompt
        g.attrs["text"] = rec.text
        g.attrs["hidden_pos_kind"] = json.dumps(rec.hidden_pos_kind)
        g.attrs["meta"] = json.dumps(rec.meta)

        def _ds(name, arr, dtype):
            arr = np.asarray(arr, dtype=dtype)
            # gzip helps for the big top-k / hidden arrays; harmless for small ones
            g.create_dataset(name, data=arr, compression="gzip", compression_opts=4)

        _ds("token_ids", rec.token_ids, np.int64)
        _ds("top_logprobs", rec.top_logprobs, np.float32)
        _ds("top_token_ids", rec.top_token_ids, np.int64)
        _ds("functionals", rec.functionals, np.float32)
        _ds("hidden_states", rec.hidden_states, np.float16)
        _ds("hidden_pos_index", rec.hidden_pos_index, np.int64)
        _ds("answer_token_indices", rec.answer_token_indices, np.int64)
        g.attrs["_complete"] = True  # set LAST: a crash mid-write leaves it absent
        f.flush()

    def read(self, key: str) -> GenRecord:
        f = self._file()
        if key not in f:
            raise KeyError(key)
        g = f[key]
        return GenRecord(
            key=key,
            prompt=str(g.attrs["prompt"]),
            text=str(g.attrs["text"]),
            token_ids=g["token_ids"][()],
            top_logprobs=g["top_logprobs"][()],
            top_token_ids=g["top_token_ids"][()],
            functionals=g["functionals"][()],
            hidden_states=g["hidden_states"][()],
            hidden_pos_index=g["hidden_pos_index"][()],
            hidden_pos_kind=json.loads(g.attrs["hidden_pos_kind"]),
            answer_token_indices=g["answer_token_indices"][()],
            meta=json.loads(g.attrs.get("meta", "{}")),
        )

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self) -> "H5Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
