import numpy as np

from routing.calibrate import calibrate_risk


def test_default_start_fails_on_its_first_threshold_and_coverage_start_qualifies():
    rng = np.random.RandomState(0)
    n = 2000
    s = rng.rand(n)
    y = (s > 0.9).astype(int)            # errors only among the riskiest 10%
    assert calibrate_risk(s, y, 0.2)["threshold"] is None        # one accepted row: bound above 1
    r = calibrate_risk(s, y, 0.2, start=0.10)
    assert r["threshold"] is not None and r["coverage_cal"] >= 0.5 and r["ucb"] <= 0.2


def test_coverage_start_still_refuses_when_risk_is_high():
    rng = np.random.RandomState(1)
    s = rng.rand(500)
    y = (rng.rand(500) < 0.5).astype(int)
    assert calibrate_risk(s, y, 0.2, start=0.10)["threshold"] is None
