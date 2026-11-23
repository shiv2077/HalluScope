import numpy as np
import pandas as pd

from itemtriage import panels, rank, study2


def _data(seed=0, n=60):
    rng = np.random.RandomState(seed)
    s = rng.randint(0, 6, n).astype(float)          # many ties
    y = (rng.rand(n) < 0.2).astype(float)
    W = np.vstack([np.bincount(rng.randint(0, n, n), minlength=n) for _ in range(25)])
    return s, y, W


def test_weighted_precision_matches_rank_on_expanded_data():
    s, y, W = _data()
    for budget in (0.05, 0.1, 0.2):
        got = study2.precision_w(s, y, W, budget)
        for r in range(len(W)):
            ix = np.repeat(np.arange(len(s)), W[r])
            want = rank.precision_at(y[ix], s[ix], np.zeros(len(ix)), budget)
            assert np.isclose(got[r], want)


def test_weighted_ap_matches_rank_on_expanded_data():
    s, y, W = _data(1)
    got = study2.ap_w(s, y, W)
    for r in range(len(W)):
        ix = np.repeat(np.arange(len(s)), W[r])
        want = rank.average_precision(y[ix], s[ix], np.zeros(len(ix)))
        assert (want is None and np.isnan(got[r])) or np.isclose(got[r], want)


def test_nested_single_draw_is_the_outer_bootstrap():
    pt, reps = study2.nested(np.array([0.3]), np.array([[0.1, 0.2, 0.4]]))
    assert pt == 0.3 and np.allclose(reps, [0.1, 0.2, 0.4])


def test_verdict_is_two_sided():
    assert study2.verdict({"lo": 0.01, "hi": 0.2}) == "holds"
    assert study2.verdict({"lo": -0.2, "hi": -0.01}) == "reversed"
    assert study2.verdict({"lo": -0.1, "hi": 0.1}) == "inconclusive"


def test_panels_are_fixed_and_stratified():
    reps = pd.DataFrame({"model": [f"m{i}" for i in range(12)], "ability": np.linspace(0.9, 0.3, 12),
                         "lineage": [i // 2 for i in range(12)]})
    a, b = panels.draw_panels(reps, (1, 3, 5)), panels.draw_panels(reps, (1, 3, 5))
    assert a == b
    by = {p["id"]: p for p in a}
    assert by["top-3-0"]["models"] == ["m0", "m1", "m2"] and by["bottom-3-0"]["models"] == ["m9", "m10", "m11"]
    lin = dict(zip(reps["model"], reps["lineage"]))
    for p in a:
        if p["stratum"] == "lineage":
            assert len({lin[m] for m in p["models"]}) == p["k"]
