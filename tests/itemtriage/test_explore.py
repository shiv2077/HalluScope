import numpy as np
import pandas as pd

from itemtriage import explore, rank


def test_inclusion_sums_to_k_and_matches_expected_flagged():
    rng = np.random.RandomState(0)
    s1 = rng.randint(0, 4, 50).astype(float)
    s2 = rng.randint(0, 2, 50).astype(float)
    y = rng.randint(0, 2, 50).astype(float)
    for k in (0, 1, 7, 13, 50):
        p = rank.inclusion(s1, s2, k)
        assert np.isclose(p.sum(), k)
        assert np.isclose((p * y).sum(), rank.expected_flagged(y, s1, s2, k)[0])


def test_inclusion_is_invariant_to_row_order():
    s1 = np.array([3.0, 1, 3, 2, 3, 1])
    s2 = np.zeros(6)
    perm = np.array([5, 2, 0, 4, 1, 3])
    assert np.allclose(rank.inclusion(s1, s2, 2)[perm], rank.inclusion(s1[perm], s2[perm], 2))
    assert np.allclose(rank.inclusion(s1, s2, 2), [2 / 3, 0, 2 / 3, 0, 2 / 3, 0])


def test_subject_scores_need_no_labels():
    sc = pd.DataFrame({"item_id": list("abcde"), "subject": ["x", "x", "y", "y", "y"], "conf_w": [0.2, 0.4, 0.1, 0.1, 0.7]})
    out = explore.subject_scores(sc)
    assert np.allclose(out["subject_mean"], [0.3, 0.3, 0.3, 0.3, 0.3])
    assert np.allclose(out.groupby("subject")["within_subject"].sum(), 0.0)
