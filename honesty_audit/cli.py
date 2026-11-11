"""Command line.

Examples
--------
# real audit (Colab / a laptop with internet), hourly temperature in 10 Indian cities:
python -m honesty_audit run --model chronos-bolt-small \
    --data open-meteo-cities:2025-06-01,2026-10-31 --out results/cities_temp

# your own CSV (e.g. Delhi demand), one value column, optional time and series columns:
python -m honesty_audit run --model chronos-bolt-small \
    --data csv:delhi_load.csv:load:timestamp --out results/delhi

# offline test with synthetic data and the stand-in model (no downloads):
python -m honesty_audit run --model stand-in --data synthetic --out results/selftest

# rebuild the honesty report page from saved results:
python -m honesty_audit report results/selftest/results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from .audit import AuditConfig, run_audit
from .data import (fetch_open_meteo, fetch_open_meteo_many, load_csv, make_synthetic_load,
                   make_windows)
from .models import TinyQuantileNet, load_model
from .report import write_report


def _load_data(spec: str, seed: int):
    if spec == "synthetic":
        return make_synthetic_load(seed=seed + 1), "synthetic electricity-like load (NOT real data)"
    kind, _, rest = spec.partition(":")
    if kind == "open-meteo":
        lat, lon, start, end = rest.split(",")
        return (fetch_open_meteo(float(lat), float(lon), start, end, name="navi_mumbai_temp"),
                f"Open-Meteo hourly temperature at {lat},{lon}, {start} to {end} (reanalysis)")
    if kind == "open-meteo-cities":
        start, end = rest.split(",")
        return (fetch_open_meteo_many(start, end),
                f"Open-Meteo hourly temperature, 10 Indian cities, {start} to {end} (reanalysis)")
    if kind == "csv":
        parts = rest.split(":")
        path, value_col = parts[0], parts[1]
        time_col = parts[2] if len(parts) > 2 and parts[2] else None
        series_col = parts[3] if len(parts) > 3 and parts[3] else None
        with open(path, "rb") as fh:  # the checksum proves which exact file was used
            sha = hashlib.sha256(fh.read()).hexdigest()
        return (load_csv(path, value_col, time_col, series_col),
                f"CSV file {os.path.basename(path)} (SHA-256 {sha})")
    raise SystemExit(f"unknown --data {spec!r}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="honesty_audit")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a full audit")
    r.add_argument("--model", default="chronos-bolt-small")
    r.add_argument("--data", default="synthetic")
    r.add_argument("--out", default="results/run")
    r.add_argument("--context", type=int, default=512)
    r.add_argument("--horizon", type=int, default=24)
    r.add_argument("--n-calib", type=int, default=20)
    r.add_argument("--n-test", type=int, default=20)
    r.add_argument("--bits", default="16,8,6,4,3,2")
    r.add_argument("--prune", default="", help="e.g. 0.3,0.5,0.7")
    r.add_argument("--tolerance", type=float, default=0.05)
    r.add_argument("--max-score-worse", type=float, default=0.10,
                   help="usefulness check: allowed worsening of the interval score, e.g. 0.10 = 10%%")
    r.add_argument("--no-keep-io", action="store_true")
    r.add_argument("--real-int8", action="store_true")
    r.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("report", help="rebuild the HTML report from results.json")
    g.add_argument("results")
    args = p.parse_args(argv)

    if args.cmd == "report":
        with open(args.results) as fh:
            res = json.load(fh)
        path = write_report(res, os.path.dirname(args.results) or ".")
        print(f"report written to {path}")
        return

    np.random.seed(args.seed)
    cfg = AuditConfig(
        context_len=args.context, horizon=args.horizon, n_calib=args.n_calib,
        n_test=args.n_test, quant_bits=[int(b) for b in args.bits.split(",") if b],
        prune_sparsity=[float(s) for s in args.prune.split(",") if s],
        tolerance=args.tolerance, usefulness_max_worse=args.max_score_worse, compare_keep_io=not args.no_keep_io,
        real_int8_check=args.real_int8, seed=args.seed)
    series, data_desc = _load_data(args.data, args.seed)
    windows = make_windows(series, cfg.context_len, cfg.horizon, cfg.n_calib, cfg.n_test)

    if args.model == "stand-in":
        model = TinyQuantileNet(cfg.context_len, cfg.horizon, seed=args.seed)
        # train ONLY on data before the first calibration window, so nothing leaks
        first = min(w.origin for w in windows)
        train = [np.asarray(s.values[:first]) for s in series.values()]
        print("[run] training the stand-in network on pre-calibration data ...")
        model.fit(train, seed=args.seed)
    else:
        model = load_model(args.model)

    res = run_audit(model, windows, cfg)
    res["meta"]["data"] = data_desc
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(res, fh)
    path = write_report(res, args.out)
    print(f"[run] done in {res['meta']['runtime_s']} s -> {args.out}/results.json and {path}")
    for k, v in res["breaking_points"].items():
        if k.startswith("80%") and "all layers" in k:
            print(f"   {k}: safe down to {v['safe_down_to']}, breaks at {v['breaks_at']}")


if __name__ == "__main__":
    main()
