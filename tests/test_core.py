"""Tests: if these pass, the maths in the audit does what the docs say.

Run with:  python -m pytest -q
"""

import numpy as np
import pytest

from honesty_audit import metrics as M
from honesty_audit.conformal import AdaptiveCQR, SplitCQR, WeightedCQR, cqr_scores
from honesty_audit.data import make_synthetic_load, make_windows


def test_coverage_and_interval_score_by_hand():
    lo, hi = np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]])
    y = np.array([[0.5, 2.0]])  # first inside, second 1.0 above
    assert M.covered(lo, hi, y).tolist() == [[1.0, 0.0]]
    # width 1 each; second also pays 2/alpha * 1 = 10 for alpha 0.2
    assert M.interval_score(lo, hi, y, 0.2).tolist() == [[1.0, 11.0]]


def test_split_cqr_reaches_target_on_exchangeable_data():
    rng = np.random.default_rng(1)
    n, H = 4000, 3
    y = rng.normal(size=(n, H))
    lo, hi = np.full((n, H), -0.5), np.full((n, H), 0.5)  # far too narrow (~38%)
    cal, test = slice(0, 2000), slice(2000, None)
    fix = SplitCQR(alpha=0.2).fit(lo[cal], hi[cal], y[cal])
    l2, h2 = fix.apply(lo[test], hi[test])
    cov = M.covered(l2, h2, y[test]).mean()
    assert 0.78 <= cov <= 0.83  # split conformal: at least 80%, barely above


def test_split_cqr_can_also_shrink_too_wide_ranges():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(3000, 1))
    lo, hi = np.full_like(y, -5.0), np.full_like(y, 5.0)
    fix = SplitCQR(0.2).fit(lo[:1500], hi[:1500], y[:1500])
    assert fix.q[0] < 0  # negative correction = narrower range


def test_weighted_cqr_equals_split_with_equal_weights():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(500, 2))
    lo, hi = np.zeros_like(y) - 0.3, np.zeros_like(y) + 0.3
    a = SplitCQR(0.2).fit(lo, hi, y).q
    b = WeightedCQR(0.2, rho=1.0).fit(lo, hi, y, ages=np.zeros(500)).q
    assert np.allclose(a, b)


def test_adaptive_recovers_after_a_level_shift():
    rng = np.random.default_rng(4)
    T, H = 400, 1
    y = rng.normal(size=(T, H))
    y[200:] += 3.0  # sudden drift the model ignores
    lo, hi = np.full((T, H), -1.28), np.full((T, H), 1.28)
    cal = cqr_scores(lo[:50], hi[:50], y[:50])
    l2, h2 = AdaptiveCQR(0.2, gamma=0.05).run(lo[50:], hi[50:], y[50:], cal)
    cov_aci = M.covered(l2, h2, y[50:])[150:].mean()  # after the shift
    cov_raw = M.covered(lo[50:], hi[50:], y[50:])[150:].mean()
    assert cov_raw < 0.3 and cov_aci > 0.6


def test_cluster_bootstrap_ci_contains_mean():
    vals = np.r_[np.ones(50), np.zeros(50)]
    groups = np.repeat(np.arange(10), 10)
    out = M.cluster_bootstrap(vals, groups, n_boot=500)
    assert out["lo"] <= out["mean"] <= out["hi"] and out["n_series"] == 10


def test_windows_do_not_overlap_and_calibration_comes_first():
    series = make_synthetic_load(n_series=2, n_days=40)
    w = make_windows(series, context_len=96, horizon=24, n_calib=5, n_test=5)
    for sid in {x.series_id for x in w}:
        mine = sorted([x for x in w if x.series_id == sid], key=lambda x: x.origin)
        assert [x.split for x in mine] == ["calib"] * 5 + ["test"] * 5
        assert all(b.origin - a.origin == 24 for a, b in zip(mine, mine[1:]))
        # the context never includes the window's own target
        s = series[sid].to_numpy()
        assert np.allclose(mine[0].context, s[mine[0].origin - 96:mine[0].origin])


def test_crossing_rate():
    q = np.array([[[1, 2, 3], [3, 2, 1]]], dtype=float)
    assert M.crossing_rate(q) == 0.5


def test_verdict_usefulness_check():
    """A level that covers only by being much wider must not get a plain 'pass'."""
    from honesty_audit.audit import TOO_WIDE, _verdict

    covers = {"lo": 0.76, "hi": 0.82}
    assert _verdict(covers, 0.8, 0.05, score_ratio=1.05, max_worse=0.10) == "pass"
    assert _verdict(covers, 0.8, 0.05, score_ratio=1.39, max_worse=0.10) == TOO_WIDE
    # a clear coverage failure stays a FAIL, whatever the score
    assert _verdict({"lo": 0.60, "hi": 0.70}, 0.8, 0.05, score_ratio=1.39, max_worse=0.10) == "FAIL"


def test_verdict_three_way_and_vs_full():
    from honesty_audit.audit import _verdict, _verdict_vs_full

    assert _verdict({"lo": 0.70, "hi": 0.74}, 0.8, 0.05) == "FAIL"  # confident it is under 75%
    assert _verdict({"lo": 0.76, "hi": 0.82}, 0.8, 0.05) == "pass"  # confident it is not
    assert _verdict({"lo": 0.70, "hi": 0.78}, 0.8, 0.05) == "inconclusive"  # cannot tell
    assert _verdict_vs_full({"lo": -0.12, "hi": -0.07}, 0.05) == "worse than full"
    assert _verdict_vs_full({"lo": -0.03, "hi": 0.01}, 0.05) == "as good as full"
    assert _verdict_vs_full({"lo": -0.08, "hi": -0.02}, 0.05) == "inconclusive"
