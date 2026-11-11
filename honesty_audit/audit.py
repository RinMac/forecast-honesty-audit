"""The audit: compress step by step, measure honesty, apply the fix, find breaking points.

PRE-REGISTERED RULE (fixed in AuditConfig BEFORE any model is run, and saved with
the results so nobody can move the goalposts afterwards):

  A model FAILS a coverage level when the upper end of the 95% bootstrap CI of its
  measured coverage is below (nominal - tolerance).
  e.g. 80% range, tolerance 0.05: fail if we are confident coverage is under 75%.

  It PASSES when the lower end of the CI is at or above (nominal - tolerance): we
  are confident coverage is NOT under 75%. Anything in between is INCONCLUSIVE:
  the data cannot tell, and we say so instead of calling it a pass.

  It is flagged OVER-CAUTIOUS when the lower end of the CI is above
  (nominal + tolerance): honest, but wasting width.

  SECOND VIEW, against the full model: coverage of the compressed model minus
  coverage of the uncompressed model on the same windows. WORSE THAN FULL if the
  upper end of the CI of that difference is below -tolerance; AS GOOD AS FULL if
  the lower end is at or above -tolerance; otherwise inconclusive.

  USEFULNESS CHECK: a range can reach its coverage just by becoming huge. So a level
  that does not fail is still labelled "covers, but too wide" when its interval score
  is more than `usefulness_max_worse` (10%) worse than the score of the uncompressed
  model with no fix applied.

The BREAKING POINT of a method is the first precision, going down from 16 bits,
that fails or is too wide. "Safe down to" is the lowest precision for which it
and every higher precision passes. Inconclusive levels are never called safe.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from . import metrics as M
from .compress import compress_
from .conformal import AdaptiveCQR, SplitCQR, WeightedCQR, cqr_scores
from .data import Window
from .models import Forecaster

# split_cqr and weighted_cqr are ROLLING: refitted for each test day on the n_calib days before it.
# static_cqr is fitted once on the calibration windows and never updated.
METHODS = ("raw", "split_cqr", "static_cqr", "weighted_cqr", "adaptive_cqr")


@dataclass
class AuditConfig:
    context_len: int = 512
    horizon: int = 24
    n_calib: int = 20  # windows per series used to fit the fix (earlier in time)
    n_test: int = 20  # windows per series used to judge (later in time)
    nominal_levels: list[float] = field(default_factory=lambda: [0.8, 0.6])
    quant_bits: list[int] = field(default_factory=lambda: [16, 8, 6, 4, 3, 2])
    prune_sparsity: list[float] = field(default_factory=list)
    io_keep_patterns: list[str] = field(default_factory=lambda: [
        r"input_patch_embedding", r"output_patch_embedding", r"input_layer", r"output_layer"])
    compare_keep_io: bool = True  # also run each bit-width with input/output layers kept
    tolerance: float = 0.05
    usefulness_max_worse: float = 0.10  # interval score may be at most 10% worse than the full model's
    n_boot: int = 2000
    seed: int = 0
    aci_gamma: float = 0.05
    weighted_rho: float = 0.9
    season: int = 24  # for scaling scores; 24 for hourly data with a daily cycle
    real_int8_check: bool = False  # also time a real int8 model (CPU, dynamic quantization)
    demo_windows: int = 7  # consecutive test windows of one series saved for the demo


def _variants(cfg: AuditConfig):
    vs = [("fp32 (full)", "none", 32, [])]
    for b in cfg.quant_bits:
        vs.append((f"{b}-bit" if b < 16 else "fp16", "quant", b, []))
        if cfg.compare_keep_io and b < 16:
            vs.append((f"{b}-bit + I/O kept", "quant", b, cfg.io_keep_patterns))
    for s in cfg.prune_sparsity:
        vs.append((f"prune {int(s * 100)}%", "prune", s, []))
    return vs


def _quantile_index(levels: np.ndarray, q: float) -> int:
    hit = np.where(np.isclose(levels, q))[0]
    if len(hit) == 0:
        raise ValueError(f"model has no {q} quantile; it offers {levels.tolist()}")
    return int(hit[0])


TOO_WIDE = "covers, but too wide"


def _verdict(cov: dict, nominal: float, tol: float,
             score_ratio: float = 1.0, max_worse: float = float("inf")) -> str:
    """score_ratio = this level's interval score / the uncompressed raw model's score."""
    if cov["hi"] < nominal - tol:
        return "FAIL"
    if score_ratio > 1 + max_worse:
        return TOO_WIDE
    if cov["lo"] > nominal + tol:
        return "over-cautious"
    if cov["lo"] >= nominal - tol:
        return "pass"
    return "inconclusive"


def _verdict_vs_full(diff: dict, tol: float) -> str:
    """diff = bootstrap summary of (coverage of this model - coverage of the full model)."""
    if diff["hi"] < -tol:
        return "worse than full"
    if diff["lo"] >= -tol:
        return "as good as full"
    return "inconclusive"


def run_audit(base: Forecaster, windows: list[Window], cfg: AuditConfig,
              log=print) -> dict:
    t_start = time.time()
    H = cfg.horizon
    contexts = [w.context for w in windows]
    Y = np.array([w.target for w in windows])  # [n, H]
    split = np.array([w.split for w in windows])
    sid = np.array([w.series_id for w in windows])
    origin = np.array([w.origin for w in windows])
    scales = np.array([M.series_scale(w.context, cfg.season) for w in windows])
    cal, test = split == "calib", split == "test"
    levels = np.asarray(base.quantile_levels)
    med_idx = _quantile_index(levels, 0.5)
    # position of each window inside its own series, in time order: 0 = the earliest.
    # make_windows gives every series the same number of windows, so "step k" is the
    # same day in every series; steps 0 .. n_calib-1 are the calibration windows.
    step = np.empty(len(windows), dtype=int)
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        step[idx[np.argsort(origin[idx])]] = np.arange(len(idx))
    sc = scales[:, None]

    variants = _variants(cfg)
    results = {"variants": [], "per_window": {}}
    full_cov = {}  # reference coverage of the uncompressed model, for paired differences
    full_score = {}  # interval score of the uncompressed model with no fix, per nominal level

    demo_sid = sorted(set(sid[test]))[0]
    demo_mask = test & (sid == demo_sid)
    demo_idx = np.where(demo_mask)[0][np.argsort(origin[demo_mask])][-cfg.demo_windows:]
    demo = {"series_id": demo_sid, "variants": {}}

    for name, method, level, keep in variants:
        log(f"[audit] {name}: compressing and forecasting {len(windows)} windows ...")
        f = base.clone()
        rep = compress_(f.module, method, level, keep)
        t0 = time.time()
        Q = f.predict(contexts, H)  # [n, H, nQ]
        latency_ms = 1000 * (time.time() - t0) / len(windows)
        cross = M.crossing_rate(Q)
        Q = np.sort(Q, axis=-1)  # repair crossings before building intervals
        v = {
            "name": name, "method": method, "level": level, "keep_io": bool(keep),
            "compression": rep.to_dict(), "latency_ms_per_window": round(latency_ms, 2),
            "crossing_rate": cross,
            "wql": M.weighted_quantile_loss(Q[test], Y[test], levels),
            "median_scaled_mae": float((np.abs(Q[test, :, med_idx] - Y[test]) / scales[test, None]).mean()),
            "levels": {},
        }
        demo_v = {"median": Q[demo_idx, :, med_idx].ravel().tolist()}

        for nominal in cfg.nominal_levels:
            alpha = 1 - nominal
            lo_i = _quantile_index(levels, round(alpha / 2, 4))
            hi_i = _quantile_index(levels, round(1 - alpha / 2, 4))
            lo, hi = Q[..., lo_i], Q[..., hi_i]

            intervals = {"raw": (lo[test], hi[test])}
            # All fixes work on values divided by each window's series scale, so series with
            # large and small swings can share one calibration set. Results are converted
            # back to the original units (e.g. degrees) at the end.
            ls, hs, Ys = lo / sc, hi / sc, Y / sc

            # rolling split and weighted CQR: for each test day, fit on the n_calib days
            # just before it (all series pooled), then correct that day only
            r_lo, r_hi = ls.copy(), hs.copy()
            w_lo, w_hi = ls.copy(), hs.copy()
            for k in np.unique(step[test]):
                past = (step >= k - cfg.n_calib) & (step < k)
                now = step == k
                r_lo[now], r_hi[now] = SplitCQR(alpha).fit(ls[past], hs[past], Ys[past]).apply(ls[now], hs[now])
                w_lo[now], w_hi[now] = WeightedCQR(alpha, cfg.weighted_rho).fit(
                    ls[past], hs[past], Ys[past], ages=k - 1 - step[past]).apply(ls[now], hs[now])
            intervals["split_cqr"] = (r_lo[test] * sc[test], r_hi[test] * sc[test])
            # static: fitted once on the calibration windows and never updated
            s_lo, s_hi = SplitCQR(alpha).fit(ls[cal], hs[cal], Ys[cal]).apply(ls[test], hs[test])
            intervals["static_cqr"] = (s_lo * sc[test], s_hi * sc[test])
            intervals["weighted_cqr"] = (w_lo[test] * sc[test], w_hi[test] * sc[test])

            # adaptive: run per series, in time order, starting from that series' calibration scores
            a_lo, a_hi = np.empty_like(lo), np.empty_like(hi)
            aci = AdaptiveCQR(alpha, cfg.aci_gamma)
            for s in np.unique(sid):
                c_idx = np.where(cal & (sid == s))[0]
                t_idx = np.where(test & (sid == s))[0]
                t_idx = t_idx[np.argsort(origin[t_idx])]
                if len(t_idx) == 0:
                    continue
                cs = cqr_scores(ls[c_idx], hs[c_idx], Ys[c_idx])
                a_lo[t_idx], a_hi[t_idx] = aci.run(ls[t_idx], hs[t_idx], Ys[t_idx], cs)
            intervals["adaptive_cqr"] = (a_lo[test] * sc[test], a_hi[test] * sc[test])

            lv = {}
            for m_name, (l, h) in intervals.items():
                summ = M.summarize_interval(l, h, Y[test], alpha, scales[test], sid[test],
                                            cfg.n_boot, cfg.seed)
                score = summ["scaled_interval_score"]["mean"]
                if method == "none" and m_name == "raw":  # always the first one computed
                    full_score[nominal] = score
                summ["score_vs_full_raw"] = score / full_score[nominal]
                summ["verdict"] = _verdict(summ["coverage"], nominal, cfg.tolerance,
                                           summ["score_vs_full_raw"], cfg.usefulness_max_worse)
                cov_ind = M.covered(l, h, Y[test])
                key = (nominal, m_name)
                if method == "none":
                    full_cov[key] = cov_ind
                summ["vs_full_model"] = M.paired_difference(full_cov[key], cov_ind, sid[test],
                                                            cfg.n_boot, cfg.seed)
                summ["vs_full_model"]["verdict"] = ("reference" if method == "none" else _verdict_vs_full(
                    summ["vs_full_model"], cfg.tolerance))
                lv[m_name] = summ
                results["per_window"].setdefault(name, {})[f"{nominal}_{m_name}"] = cov_ind.mean(axis=1).round(4).tolist()
            v["levels"][str(nominal)] = lv

            if np.isclose(nominal, 0.8):
                pos = {i: k for k, i in enumerate(np.where(test)[0])}
                rows = [pos[i] for i in demo_idx]
                demo_v["raw"] = [lo[demo_idx].ravel().tolist(), hi[demo_idx].ravel().tolist()]
                for m_name in ("split_cqr", "adaptive_cqr"):
                    l, h = intervals[m_name]
                    demo_v[m_name] = [l[rows].ravel().tolist(), h[rows].ravel().tolist()]
        results["variants"].append(v)
        demo["variants"][name] = demo_v
        r80 = v["levels"].get("0.8", {})
        if r80:
            log(f"         80% coverage raw {r80['raw']['coverage']['mean']:.3f} "
                f"[{r80['raw']['coverage']['lo']:.3f}, {r80['raw']['coverage']['hi']:.3f}] "
                f"-> {r80['raw']['verdict']}; after split CQR {r80['split_cqr']['coverage']['mean']:.3f}")

    # demo context: truth and recent history
    first = demo_idx[0]
    demo["truth"] = Y[demo_idx].ravel().tolist()
    demo["history"] = windows[first].context[-3 * H:].tolist()
    demo["timestamps"] = sum((windows[i].timestamps for i in demo_idx), [])
    results["demo"] = demo
    # which test window is which (same order as the per_window lists), so that
    # coverage per month or per city can be worked out later from results.json
    t_ids = np.where(test)[0]
    results["test_windows"] = {
        "series_id": [str(sid[i]) for i in t_ids],
        "start": [windows[i].timestamps[0] if windows[i].timestamps else int(origin[i]) for i in t_ids],
    }

    if cfg.real_int8_check:
        results["real_int8"] = real_int8_speed(base, contexts[:64], H, log)

    results["breaking_points"] = breaking_points(results, cfg)
    results["meta"] = {
        "model": base.name,
        "n_windows": int(len(windows)), "n_test_windows": int(test.sum()),
        "n_series": int(len(np.unique(sid))),
        "config": asdict(cfg),
        "model_revision": getattr(base, "revision", None),
        "pre_registered_rule": (f"FAIL if the 95% CI upper bound of coverage < nominal - {cfg.tolerance}; "
                                f"pass if the CI lower bound >= nominal - {cfg.tolerance}; inconclusive in between; "
                                f"over-cautious if the CI lower bound > nominal + {cfg.tolerance}; "
                                f"'{TOO_WIDE}' if it does not fail but its interval score is more than "
                                f"{cfg.usefulness_max_worse:.0%} worse than the uncompressed model's"),
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_s": round(time.time() - t_start, 1),
        "notes": ["Quantization is simulated: accuracy and calibration are faithful, speed is not.",
                  "Coverage is MEASURED on held-out windows; time series are not exchangeable, "
                  "so no finite-sample guarantee is claimed."],
    }
    return results


def breaking_points(results: dict, cfg: AuditConfig) -> dict:
    """For each nominal level, method and I/O setting: the first precision that fails or is too wide."""
    out = {}
    quant = [v for v in results["variants"] if v["method"] in ("none", "quant")]
    for nominal in cfg.nominal_levels:
        for keep in (False, True):
            chain = [v for v in quant if v["method"] == "none" or v["keep_io"] == keep]
            chain.sort(key=lambda v: -v["level"])
            for m in METHODS:
                safe, broke, worse, all_pass = None, None, None, True
                for v in chain:
                    s = v["levels"][str(nominal)][m]
                    if worse is None and s["vs_full_model"]["verdict"] == "worse than full":
                        worse = v["name"]
                    if broke is None and s["verdict"] in ("FAIL", TOO_WIDE):
                        broke = v["name"]
                    # "safe" stops moving down at the first level that is not a clear pass
                    all_pass = all_pass and broke is None and s["verdict"] in ("pass", "over-cautious")
                    if all_pass:
                        safe = v["name"]
                if keep and len(chain) == 1:
                    continue
                out[f"{int(nominal * 100)}% | {m} | {'I/O kept' if keep else 'all layers'}"] = {
                    "safe_down_to": safe, "breaks_at": broke, "first_worse_than_full": worse}
    return out


def real_int8_coverage(base: Forecaster, windows: list[Window], cfg: AuditConfig, log=print) -> dict:
    """Does the simulation tell the truth? Compare simulated 8-bit with a REAL int8 model.

    Both models forecast the same test windows (no fix applied). For each range we report
    the coverage of each, and the paired difference (real minus simulated) with its CI.
    The two are not expected to match exactly: the simulation uses one scale per output
    row of each weight matrix, while PyTorch's dynamic int8 uses one scale per whole matrix.
    """
    import torch
    from torch.ao.quantization import quantize_dynamic

    test = [w for w in windows if w.split == "test"]
    contexts = [w.context for w in test]
    Y = np.array([w.target for w in test])
    sid = np.array([w.series_id for w in test])
    scales = np.array([M.series_scale(w.context, cfg.season) for w in test])
    levels = np.asarray(base.quantile_levels)
    med_idx = _quantile_index(levels, 0.5)

    sim = base.clone()
    compress_(sim.module, "quant", 8, [])
    real = base.clone()
    real.set_module(quantize_dynamic(real.module, {torch.nn.Linear}, dtype=torch.qint8))

    out = {"n_test_windows": len(test), "models": {}, "levels": {}}
    preds = {}
    for name, f in (("simulated 8-bit", sim), ("real int8", real)):
        log(f"[int8 check] {name}: forecasting {len(test)} test windows ...")
        Q = np.sort(f.predict(contexts, cfg.horizon), axis=-1)
        preds[name] = Q
        out["models"][name] = {
            "median_scaled_mae": float((np.abs(Q[:, :, med_idx] - Y) / scales[:, None]).mean())}

    for nominal in cfg.nominal_levels:
        alpha = 1 - nominal
        lo_i = _quantile_index(levels, round(alpha / 2, 4))
        hi_i = _quantile_index(levels, round(1 - alpha / 2, 4))
        lv, cov = {}, {}
        for name, Q in preds.items():
            lo, hi = Q[..., lo_i], Q[..., hi_i]
            lv[name] = M.summarize_interval(lo, hi, Y, alpha, scales, sid, cfg.n_boot, cfg.seed)
            lv[name]["verdict"] = _verdict(lv[name]["coverage"], nominal, cfg.tolerance)
            cov[name] = M.covered(lo, hi, Y)
        diff = M.paired_difference(cov["simulated 8-bit"], cov["real int8"], sid, cfg.n_boot, cfg.seed)
        # the simulation is "confirmed" when real int8 is within the tolerance of it, both ways
        diff["simulation_confirmed"] = bool(diff["lo"] >= -cfg.tolerance and diff["hi"] <= cfg.tolerance)
        lv["real_minus_simulated"] = diff
        out["levels"][str(nominal)] = lv
        log(f"         {int(nominal * 100)}% range: simulated {lv['simulated 8-bit']['coverage']['mean']:.3f}, "
            f"real {lv['real int8']['coverage']['mean']:.3f}, "
            f"difference {diff['mean']:+.3f} [{diff['lo']:+.3f}, {diff['hi']:+.3f}]")
    return out


def real_int8_speed(base: Forecaster, contexts, horizon: int, log=print) -> dict:
    """Time a REAL int8 model (PyTorch dynamic quantization of Linear layers, CPU only).

    Uses torch.ao.quantization.quantize_dynamic, which PyTorch is phasing out in favour
    of torchao; if it is unavailable we report that instead of guessing.
    """
    try:
        import torch
        from torch.ao.quantization import quantize_dynamic
        if not isinstance(base.module, torch.nn.Module):
            return {"available": False, "reason": "model is not a PyTorch model"}
    except Exception as e:  # pragma: no cover
        return {"available": False, "reason": repr(e)}
    try:
        t0 = time.time()
        base.predict(contexts, horizon)
        fp32_ms = 1000 * (time.time() - t0) / len(contexts)
        f = base.clone()
        qmod = quantize_dynamic(f.module, {torch.nn.Linear}, dtype=torch.qint8)
        f.set_module(qmod)
        t0 = time.time()
        f.predict(contexts, horizon)
        int8_ms = 1000 * (time.time() - t0) / len(contexts)
        log(f"[audit] real int8 speed: fp32 {fp32_ms:.1f} ms/window, int8 {int8_ms:.1f} ms/window")
        return {"available": True, "fp32_ms_per_window": round(fp32_ms, 2),
                "int8_ms_per_window": round(int8_ms, 2), "speedup": round(fp32_ms / int8_ms, 2)}
    except Exception as e:
        return {"available": False, "reason": repr(e)}
