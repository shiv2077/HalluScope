"""Alias-EM grading + normalization + audit log. No model needed."""

from src.hv2.grading import GraderAudit, alias_em, grade, normalize


def test_normalize_strips_articles_punct_case():
    assert normalize("The Eiffel Tower!") == "eiffel tower"
    assert normalize("A  quick,  BROWN  fox.") == "quick brown fox"


def test_exact_alias_match():
    ok, matched = alias_em("Paris", ["Paris", "City of Light"])
    assert ok and matched == "Paris"


def test_alias_case_and_punct_insensitive():
    ok, _ = alias_em("the eiffel tower.", ["Eiffel Tower"])
    assert ok


def test_verbose_answer_contains_alias():
    ok, matched = alias_em("The capital of France is Paris, of course.", ["Paris"])
    assert ok and matched == "Paris"


def test_wrong_answer_rejected():
    ok, matched = alias_em("London", ["Paris", "City of Light"])
    assert not ok and matched is None


def test_no_partial_token_false_positive():
    # "France" must not match inside "Francesca"
    ok, _ = alias_em("Francesca was here", ["France"])
    assert not ok


def test_multiword_alias_subsequence():
    ok, matched = alias_em("I think it is la tour eiffel honestly", ["la tour eiffel"])
    assert ok and matched == "la tour eiffel"


def test_audit_dump_samples_50(tmp_path):
    audit = GraderAudit()
    for i in range(200):
        grade(f"answer {i}", [f"answer {i}"] if i % 2 == 0 else ["different"], qid=f"q{i}", audit=audit)
    out = audit.dump_sample(tmp_path / "audit.csv", n=50, seed=42)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 51  # header + 50 rows
    assert lines[0].startswith("qid,prediction,aliases,correct,matched_alias")
