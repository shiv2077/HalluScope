"""Round-trip + resume semantics for the observability cache. No model needed."""

import numpy as np

from src.hv2.hdf5_store import GenRecord, H5Store


def _synthetic(key="Qwen2.5-0.5B/qid_1/probe0", T=5, K=100, L=25, H=896, P=4, A=2):
    rng = np.random.default_rng(0)
    return GenRecord(
        key=key,
        prompt="Q: capital of France? A:",
        text="Paris",
        token_ids=rng.integers(0, 50000, size=T),
        top_logprobs=-rng.random((T, K)).astype(np.float32) * 10,
        top_token_ids=rng.integers(0, 50000, size=(T, K)),
        functionals=rng.random((T, 3)).astype(np.float32),
        hidden_states=rng.standard_normal((P, L, H)).astype(np.float16),
        hidden_pos_index=np.arange(P),
        hidden_pos_kind=["last_prompt", "first_answer", "last_answer", "exact_answer"],
        answer_token_indices=np.arange(A),
        meta={"model": "Qwen2.5-0.5B", "qid": "qid_1", "kind": "probe0"},
    )


def test_roundtrip_preserves_shapes_and_values(tmp_path):
    rec = _synthetic()
    path = tmp_path / "cache.h5"
    with H5Store(path) as store:
        store.write(rec)
    with H5Store(path, mode="r") as store:
        got = store.read(rec.key)

    assert got.text == "Paris"
    assert got.prompt == rec.prompt
    assert got.token_ids.shape == rec.token_ids.shape
    assert got.top_logprobs.shape == (5, 100)
    assert got.functionals.shape == (5, 3)
    assert got.hidden_states.shape == (4, 25, 896)
    assert got.hidden_states.dtype == np.float16  # CLAUDE.md: hidden states fp16
    assert got.hidden_pos_kind == rec.hidden_pos_kind
    np.testing.assert_array_equal(got.token_ids, rec.token_ids)
    np.testing.assert_allclose(got.top_logprobs, rec.top_logprobs, rtol=1e-6)
    assert got.meta["qid"] == "qid_1"


def test_has_and_resume_skips_existing(tmp_path):
    path = tmp_path / "cache.h5"
    rec = _synthetic()
    with H5Store(path) as store:
        assert not store.has(rec.key)  # before write
        store.write(rec)
        assert store.has(rec.key)  # after write -> resume would skip

    # Reopen (simulate a restarted run) — key must still register as complete.
    with H5Store(path) as store2:
        assert store2.has(rec.key)
        assert rec.key in store2.keys()
        # a different key is absent -> would be (re)generated
        assert not store2.has("Qwen2.5-0.5B/qid_999/probe0")


def test_overwrite_partial_then_complete(tmp_path):
    path = tmp_path / "cache.h5"
    rec = _synthetic()
    with H5Store(path) as store:
        store.write(rec)
        store.write(rec)  # idempotent overwrite must not raise
        assert store.has(rec.key)
        assert len(store.keys()) == 1
