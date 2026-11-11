"""Measuring how honest a set of prediction intervals is.

Arrays use the shape [n_windows, horizon]: one row per forecast window, one
column per step ahead.

Coverage   share of true values that landed inside the interval (want ~ nominal)
Width      how wide the interval is (narrower is better, IF coverage holds)
Interval score (Gneiting & Raftery, 2007): width + penalty for every miss.
           It cannot be gamed by making intervals huge or tiny. Lower is better.

Confidence intervals come from a *series-level* (cluster) bootstrap: we resample
whole series, not single points, because points inside one series are correlated.
"""

from __future__ import annotations

import numpy as np


def covered(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1.0 where the truth fell inside [lo, hi], else 0.0."""
    return ((y >= lo) & (y <= hi)).astype(float)


def interval_score(lo, hi, y, alpha: float) -> np.ndarray:
    """Gneiting & Raftery (2007) interval score for a (1 - alpha) interval."""
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    return (hi - lo) + (2.0 / alpha) * below + (2.0 / alpha) * above


def series_scale(context: np.ndarray, season: int = 1) -> float:
    """Typical size of a change in this series; used to make scores comparable across series."""
    if len(context) <= season:
        return float(np.mean(np.abs(context))) or 1.0
    diffs = np.abs(context[season:] - context[:-season])
    s = float(np.mean(diffs))
    return s if s > 1e-12 else 1.0


def weighted_quantile_loss(q_pred: np.ndarray, y: np.ndarray, levels: np.ndarray) -> float:
    """Mean weighted quantile loss (approximates CRPS). q_pred: [n, H, Q], y: [n, H]."""
    diff = y[..., None] - q_pred
    pinball = np.maximum(levels * diff, (levels - 1.0) * diff)
    return float(2.0 * pinball.sum(axis=(0, 1)).mean() / max(np.abs(y).sum(), 1e-12))


def crossing_rate(q_pred: np.ndarray) -> float:
    """Share of (window, step) where some lower quantile is ABOVE a higher one."""
    bad = (np.diff(q_pred, axis=-1) < 0).any(axis=-1)
    return float(bad.mean())


# --------------------------------------------------------------------------- bootstrap


def cluster_bootstrap(values: np.ndarray, groups: np.ndarray, n_boot: int = 2000,
                      seed: int = 0, ci: float = 0.95) -> dict:
    """Mean of `values` with a percentile CI, resampling whole groups (series).

    values: one number per window (e.g. that window's coverage across the horizon)
    groups: the series each window belongs to
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    sums = np.array([values[groups == g].sum() for g in uniq])
    counts = np.array([(groups == g).sum() for g in uniq])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    a = (1 - ci) / 2
    return {
        "mean": float(values.mean()),
        "lo": float(np.quantile(boot, a)),
        "hi": float(np.quantile(boot, 1 - a)),
        "n_windows": int(len(values)),
        "n_series": int(len(uniq)),
    }


def summarize_interval(lo, hi, y, alpha: float, scales: np.ndarray, groups: np.ndarray,
                       n_boot: int = 2000, seed: int = 0) -> dict:
    """All headline numbers for one set of intervals, with bootstrap CIs."""
    cov = covered(lo, hi, y)  # [n, H]
    width = (hi - lo) / scales[:, None]
    score = interval_score(lo, hi, y, alpha) / scales[:, None]
    return {
        "nominal": 1 - alpha,
        "coverage": cluster_bootstrap(cov.mean(axis=1), groups, n_boot, seed),
        "coverage_by_horizon": cov.mean(axis=0).round(4).tolist(),
        "worst_horizon_coverage": float(cov.mean(axis=0).min()),
        "miss_below": float((y < lo).mean()),  # truth under the range; a fair 80% range has 10% here
        "miss_above": float((y > hi).mean()),  # truth over the range; and 10% here
        "scaled_width": float(width.mean()),
        "scaled_interval_score": cluster_bootstrap(score.mean(axis=1), groups, n_boot, seed),
    }


def paired_difference(cov_a: np.ndarray, cov_b: np.ndarray, groups: np.ndarray,
                      n_boot: int = 2000, seed: int = 0) -> dict:
    """Coverage of B minus coverage of A on the SAME windows, with a cluster-bootstrap CI.

    Pairing removes the noise of 'easy vs hard windows', isolating the effect of compression.
    """
    diff = cov_b.mean(axis=1) - cov_a.mean(axis=1)
    return cluster_bootstrap(diff, groups, n_boot, seed)
