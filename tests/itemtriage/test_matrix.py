import pickle

import numpy as np
import pandas as pd
import pytest

from itemtriage import matrix


def test_safe_load_accepts_arrays_and_rejects_other_globals(tmp_path):
    ok = tmp_path / "ok.pickle"
    ok.write_bytes(pickle.dumps({"data": {"x": {"correctness": np.eye(2)}}, "models": ["a"]}, protocol=5))
    assert matrix.safe_load(ok)["data"]["x"]["correctness"].shape == (2, 2)
    bad = tmp_path / "bad.pickle"
    bad.write_bytes(pickle.dumps(pd.DataFrame({"a": [1]})))
    with pytest.raises(pickle.UnpicklingError):
        matrix.safe_load(bad)


def test_join_uses_first_duplicate_row_and_reports_unmatched():
    items = pd.DataFrame({"item_id": ["i1", "i2", "i3"], "subject": ["s", "s", "s"], "text_hash": ["h1", "h2", "hx"]})
    rows = pd.DataFrame({"subject": ["s"] * 4, "pos": [0, 1, 2, 3], "th": ["h2", "h1", "h2", None]})
    m, info = matrix.join_mmlu(items, rows)
    assert dict(zip(m["item_id"], m["pos"])) == {"i1": 1, "i2": 0}
    assert info["unmatched"] == ["i3"] and info["matched_to_duplicate_text"] == 1


def test_parse_mc_and_clusters():
    q = matrix.parse_mc("What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:")
    assert q == ("What is 2+2?", ["3", "4", "5", "6"])
    X = np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]] * 5)
    c = matrix.clusters(X, 0.95)
    assert c[0] == c[1] and c[0] != c[2]
