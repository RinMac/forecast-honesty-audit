"""The fix: conformal recalibration of a model's own quantile intervals.

All methods use the CQR score (Romano et al., 2019):
    score = max(lo - y, y - hi)
Positive means the truth fell outside the interval by that much; negative means
it was inside with room to spare. We then widen (or shrink) every interval by a
quantile of the calibration scores.

Everything is done SEPARATELY for each step ahead (horizon), because errors 5
hours ahead are bigger than errors 1 hour ahead.

IMPORTANT (statistical honesty)
-------------------------------
* Split and weighted CQR only *guarantee* coverage if the data are exchangeable.
  Time series are not, so we report MEASURED coverage, never "guaranteed" coverage.
* Adaptive conformal inference (Gibbs & Candes, 2021) only guarantees that coverage
  is right ON AVERAGE OVER THE LONG RUN, not at every moment.
"""

from __future__ import annotations

import math

import numpy as np


def cqr_scores(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(lo - y, y - hi)


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """The k-th smallest score, k = ceil((n+1)(1-alpha)), as split conformal requires."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha) - 1e-9)
    if k > n:  # too few calibration points for this alpha: widen to infinity
        return float("inf")
    return float(np.sort(scores)[max(k, 1) - 1])


def _weighted_quantile(scores: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """Weighted quantile with an extra point of mass at +infinity (Barber et al., 2023)."""
    order = np.argsort(scores)
    s, w = scores[order], weights[order]
    w = np.append(w, 1.0)  # weight 1 for the unseen test point, placed at +inf
    s = np.append(s, np.inf)
    cdf = np.cumsum(w) / w.sum()
    return float(s[np.searchsorted(cdf, 1 - alpha - 1e-9)])


class SplitCQR:
    """Horizon-wise split conformal: one correction per step ahead."""

    name = "split_cqr"

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.q: np.ndarray | None = None

    def fit(self, lo, hi, y):
        scores = cqr_scores(lo, hi, y)  # [n_cal, H]
        self.q = np.array([_finite_sample_quantile(scores[:, h], self.alpha)
                           for h in range(scores.shape[1])])
        return self

    def apply(self, lo, hi):
        return lo - self.q[None, :], hi + self.q[None, :]


class WeightedCQR(SplitCQR):
    """Like SplitCQR, but recent calibration windows count more (weight rho**age).

    Barber et al. (2023): if the data drift, this loses less coverage than plain
    split conformal. `ages` = how many windows ago each calibration window was.
    """

    name = "weighted_cqr"

    def __init__(self, alpha: float, rho: float = 0.9):
        super().__init__(alpha)
        self.rho = rho

    def fit(self, lo, hi, y, ages=None):
        scores = cqr_scores(lo, hi, y)
        ages = np.zeros(len(scores)) if ages is None else np.asarray(ages, dtype=float)
        w = self.rho ** ages
        self.q = np.array([_weighted_quantile(scores[:, h], w, self.alpha)
                           for h in range(scores.shape[1])])
        return self


class AdaptiveCQR:
    """Adaptive conformal inference (Gibbs & Candes, 2021), one controller per step ahead.

    Walks through test windows of ONE series in time order. After each window it
    sees the truth and nudges its working alpha:
        alpha_next = alpha_now + gamma * (target_alpha - missed)
    Missing makes the next interval wider; covering makes it a bit narrower.
    Windows do not overlap, so each window's truth is known before the next
    forecast is made (no delayed-feedback problem).
    """

    name = "adaptive_cqr"

    def __init__(self, alpha: float, gamma: float = 0.05):
        self.alpha = alpha
        self.gamma = gamma

    def run(self, lo, hi, y, cal_scores: np.ndarray):
        """lo, hi, y: [T, H] for one series in time order. cal_scores: [n_cal, H] to start from."""
        T, H = lo.shape
        alpha_t = np.full(H, self.alpha)
        pool = [list(cal_scores[:, h]) for h in range(H)]
        new_lo, new_hi = np.empty_like(lo), np.empty_like(hi)
        for t in range(T):
            for h in range(H):
                a = alpha_t[h]
                scores_h = np.asarray(pool[h])
                if a <= 0:
                    # theory says "infinite interval"; we use the largest score seen
                    # so widths stay finite. Reported as a practical choice.
                    q = float(scores_h.max())
                elif a >= 1:
                    q = float(scores_h.min())
                else:
                    q = min(_finite_sample_quantile(scores_h, a), float(scores_h.max()))
                new_lo[t, h], new_hi[t, h] = lo[t, h] - q, hi[t, h] + q
                missed = float(not (new_lo[t, h] <= y[t, h] <= new_hi[t, h]))
                alpha_t[h] = alpha_t[h] + self.gamma * (self.alpha - missed)
                pool[h].append(max(lo[t, h] - y[t, h], y[t, h] - hi[t, h]))
        return new_lo, new_hi
