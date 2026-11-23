"""P(True) log-odds scorer (ADR-019): variant pooling, absent-token bound, tie term bound, ordering."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from routing.scorers import PTRUE_TIE_SCALE, p_true, p_true_logodds  # noqa: E402


def top(pairs):
    return [{"token": t, "logprob": lp} for t, lp in pairs]


def test_pools_case_and_space_variants():
    lo, ya, na = p_true_logodds(top([("Yes", math.log(0.5)), (" yes", math.log(0.2)), ("No", math.log(0.3))]))
    assert not ya and not na
    assert lo == round(math.log(0.3) - math.log(0.7), 6)


def test_absent_token_takes_lowest_returned_logprob():
    t = top([("Yes", -0.01), (" Yes", -5.0), ("Maybe", -9.0)])
    lo, ya, na = p_true_logodds(t)
    assert na and not ya
    expected_yes = math.log(math.exp(-0.01) + math.exp(-5.0))
    assert lo == round(-9.0 - expected_yes, 6)


def test_higher_no_is_riskier_and_tie_term_only_breaks_ties():
    a = top([("Yes", math.log(0.9)), ("No", math.log(0.1))])
    b = top([("Yes", math.log(0.6)), ("No", math.log(0.4))])
    assert p_true(b, 0.0) > p_true(a, 2.9)
    assert p_true(a, 0.5) > p_true(a, 0.1)
    assert p_true(a, 2.9) - p_true(a, 0.0) == pytest.approx(PTRUE_TIE_SCALE * 2.9)


def test_tie_term_bound_is_asserted():
    with pytest.raises(AssertionError):
        p_true(top([("Yes", -0.1), ("No", -2.5)]), 3.5)
