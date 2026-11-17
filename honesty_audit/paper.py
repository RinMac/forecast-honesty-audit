"""Tables and figures for the paper, made straight from the results.json files.

Nothing is typed in by hand: every number in tables.md and every point in the figures is
read from results/full_chronos_bolt_<size>/results.json (and int8_check.json).

    python -m honesty_audit paper --results results --out paper

writes paper/tables.md, paper/figure1_coverage_vs_bits.png and paper/figure2_coverage_by_hour.png.
"""

from __future__ import annotations

import json
import os

from .audit import accuracy_breaking_points

SIZES = ["tiny", "mini", "small", "base"]
BITS_ORDER = ["fp32 (full)", "fp16", "8-bit", "6-bit", "4-bit", "3-bit", "2-bit"]
BITS_LABEL = ["32", "16", "8", "6", "4", "3", "2"]
METHOD_NAMES = {"raw": "Own intervals", "split_cqr": "Rolling split CQR", "static_cqr": "Static split CQR",
                "weighted_cqr": "Weighted CQR", "adaptive_cqr": "Adaptive (ACI)"}
# categorical palette (fixed order), checked with the dataviz validator: passes colour-blind
# separation; two colours are below 3:1 contrast, so every line also has its own marker and a label
COLOURS = {"tiny": "#2a78d6", "mini": "#eb6834", "small": "#1baf7a", "base": "#eda100"}
MARKERS = {"tiny": "o", "mini": "s", "small": "^", "base": "D"}
INK, MUTED, FAIL_FILL, SURFACE = "#0b0b0b", "#52514e", "#f6dede", "#fcfcfb"


def load(results_dir: str) -> dict:
    out = {}
    for size in SIZES:
        path = os.path.join(results_dir, f"full_chronos_bolt_{size}", "results.json")
        if os.path.exists(path):
            with open(path) as fh:
                r = json.load(fh)
            int8 = os.path.join(results_dir, f"full_chronos_bolt_{size}", "int8_check.json")
            if os.path.exists(int8):
                with open(int8) as fh:
                    r["int8_check"] = json.load(fh)
            out[size] = r
    return out


def _v(r: dict) -> dict:
    return {v["name"]: v for v in r["variants"]}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _short(name: str | None) -> str:
    if name is None:
        return "none"
    return name.replace(" + I/O kept", "").replace("fp32 (full)", "full model")


# --------------------------------------------------------------------------- tables


def table1(r: dict, level: str = "0.8") -> str:
    V, full = _v(r), _v(r)["fp32 (full)"]
    rows = ["| Precision | Coverage (95% CI) | Verdict | Change vs full model (pp, 95% CI) | Scaled interval score | Scaled median error |",
            "| --- | --- | --- | --- | --- | --- |"]
    for n in BITS_ORDER:
        s = V[n]["levels"][level]["raw"]
        c, d = s["coverage"], s["vs_full_model"]
        diff = "reference" if n == "fp32 (full)" else f"{100 * d['mean']:+.1f} ({100 * d['lo']:+.1f} to {100 * d['hi']:+.1f})"
        rows.append(f"| {n} | {_pct(c['mean'])} ({_pct(c['lo'])} to {_pct(c['hi'])}) | {s['verdict']} | {diff} | "
                    f"{s['scaled_interval_score']['mean']:.2f} | {V[n]['median_scaled_mae']:.3f} "
                    f"({V[n]['median_scaled_mae'] / full['median_scaled_mae']:.2f}x) |")
    return "\n".join(rows)


def table2(R: dict, level: str = "80") -> str:
    methods = ["raw", "split_cqr", "static_cqr", "weighted_cqr", "adaptive_cqr"]
    rows = ["| Model | " + " | ".join(METHOD_NAMES[m] for m in methods) + " |", "| --- |" + " --- |" * len(methods)]
    for size, r in R.items():
        bp = r["breaking_points"]
        cells = []
        for m in methods:
            b = bp[f"{level}% | {m} | all layers"]
            cells.append(f"{_short(b['safe_down_to'])} / {_short(b['breaks_at'])}")
        rows.append(f"| Chronos-Bolt-{size} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def table_accuracy(R: dict) -> str:
    rows = ["| Model | Coverage breaks at (80%, own intervals) | Accuracy breaks at (median error > 10% worse) | Median error at 4 / 3 / 2 bits (x full) |",
            "| --- | --- | --- | --- |"]
    for size, r in R.items():
        a = accuracy_breaking_points(r)["all layers"]
        e = a["error_ratio"]
        rows.append(f"| Chronos-Bolt-{size} | {_short(r['breaking_points']['80% | raw | all layers']['breaks_at'])} | "
                    f"{_short(a['breaks_at'])} | {e['4-bit']:.2f} / {e['3-bit']:.2f} / {e['2-bit']:.2f} |")
    return "\n".join(rows)


def table_coverage_by_size(R: dict, level: str, method: str) -> str:
    rows = ["| Precision | " + " | ".join(f"{s}" for s in R) + " |", "| --- |" + " --- |" * len(R)]
    for n in BITS_ORDER:
        cells = []
        for r in R.values():
            s = _v(r)[n]["levels"][level][method]
            cells.append(f"{_pct(s['coverage']['mean'])} {s['verdict']}")
        rows.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def extra_findings(R: dict) -> str:
    lines = []
    for size, r in R.items():
        V = _v(r)
        full80 = V["fp32 (full)"]["levels"]["0.8"]["raw"]
        h = full80["coverage_by_horizon"]
        io = r["breaking_points"]["80% | raw | I/O kept"]
        prune = ", ".join(f"{n.split()[1]} {V[n]['levels']['0.8']['raw']['verdict']}" for n in ("prune 30%", "prune 50%", "prune 70%") if n in V)
        sp = r.get("real_int8", {})
        ic = r.get("int8_check", {}).get("levels", {}).get("0.8", {})
        diff = ic.get("real_minus_simulated")
        lines.append(
            f"- **{size}** ({V['fp32 (full)']['compression']['params_total']:,} weights, "
            f"{V['fp32 (full)']['compression']['est_weight_megabytes']:.1f} MB at fp32, "
            f"{V['4-bit']['compression']['est_weight_megabytes']:.1f} MB at 4-bit): "
            f"full-model 80% coverage {_pct(full80['coverage']['mean'])}, 60% coverage "
            f"{_pct(V['fp32 (full)']['levels']['0.6']['raw']['coverage']['mean'])}; "
            f"by hour ahead {_pct(h[0])} (1 h), {_pct(h[11])} (12 h), {_pct(h[23])} (24 h); "
            f"misses below / above the 80% range {_pct(full80['miss_below'])} / {_pct(full80['miss_above'])}. "
            f"I/O kept: safe down to {_short(io['safe_down_to'])}, breaks at {_short(io['breaks_at'])}. "
            f"Crossed quantiles {_pct(V['4-bit']['crossing_rate'])} / {_pct(V['3-bit']['crossing_rate'])} / "
            f"{_pct(V['2-bit']['crossing_rate'])} at 4 / 3 / 2 bits (0% at full precision). "
            f"Pruning: {prune}. "
            + (f"Real int8 on CPU: {sp['fp32_ms_per_window']} ms vs {sp['int8_ms_per_window']} ms per window "
               f"({sp['speedup']}x faster). " if sp.get("available") else "")
            + (f"Real int8 minus simulated 8-bit coverage: {100 * diff['mean']:+.1f} pp "
               f"({100 * diff['lo']:+.1f} to {100 * diff['hi']:+.1f})." if diff else ""))
    return "\n".join(lines)


def write_tables(R: dict, path: str) -> None:
    any_r = next(iter(R.values()))
    m = any_r["meta"]
    parts = [
        "# Tables for the paper (generated from results.json, do not edit by hand)",
        f"Model run settings: {m['n_test_windows']} test windows per model, {m['n_series']} cities, "
        f"{m['config']['n_calib']} calibration days, tolerance {m['config']['tolerance']}, "
        f"usefulness limit {m['config']['usefulness_max_worse']:.0%}. Data: {m.get('data', '')}.",
        "## Table 1. Chronos-Bolt-small, 80% interval, all linear layers quantized, own intervals",
        table1(R["small"]) if "small" in R else "(small not run)",
        "## Table 2. Lowest precision that passes, 80% interval, all layers quantized",
        "Each cell: safe down to / breaks at.",
        table2(R),
        "## Table 3. Coverage breaking point against accuracy breaking point (hypothesis H2)",
        table_accuracy(R),
        "## Coverage of the 80% interval by model size, own intervals",
        table_coverage_by_size(R, "0.8", "raw"),
        "## Coverage of the 80% interval by model size, after rolling split CQR",
        table_coverage_by_size(R, "0.8", "split_cqr"),
        "## Coverage of the 60% interval by model size, own intervals",
        table_coverage_by_size(R, "0.6", "raw"),
        "## Additional findings, per model",
        extra_findings(R),
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(parts) + "\n")


# --------------------------------------------------------------------------- figures


def _style(ax, ylabel: str | None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9.5)


def figure1(R: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=SURFACE)
    x = list(range(len(BITS_ORDER)))
    for ax, method, title in ((axes[0], "raw", "Model's own 80% intervals"),
                              (axes[1], "split_cqr", "After rolling split conformal recalibration")):
        _style(ax, "Measured coverage" if method == "raw" else None)
        ax.axhspan(0, 75, color=FAIL_FILL, zorder=0)
        ax.axhline(80, color=INK, linewidth=1, linestyle=(0, (4, 3)))
        # labels sit in the right margin and at the bottom, clear of the lines
        ax.annotate("promised\n80%", xy=(1.01, 80), xycoords=("axes fraction", "data"), va="center",
                    color=INK, fontsize=8, annotation_clip=False)
        ax.text(0.05, 27, "fail zone (below 75%)", color="#a13a32", fontsize=8.5)
        for size, r in R.items():
            V = _v(r)
            y = [100 * V[n]["levels"]["0.8"][method]["coverage"]["mean"] for n in BITS_ORDER]
            ax.plot(x, y, color=COLOURS[size], linewidth=2, marker=MARKERS[size], markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"Chronos-Bolt-{size}")
        ax.set_xticks(x, BITS_LABEL)
        ax.set_xlabel("bits per weight (smaller model to the right)", color=MUTED, fontsize=9.5)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
        ax.set_ylim(25, 95)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, "At 3 bits coverage stays near 80% but intervals are too wide (interval score more than 10% "
             "worse than the full model), so 3 bits counts as broken under the pre-registered rule.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure2(R: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=SURFACE)
    low = min(100 * min(_v(r)[n]["levels"]["0.8"]["raw"]["coverage_by_horizon"])
              for r in R.values() for n in ("fp32 (full)", "3-bit"))
    for ax, name, title in ((axes[0], "fp32 (full)", "Full model"), (axes[1], "3-bit", "3 bits (the breaking point)")):
        _style(ax, "Measured coverage of the 80% interval" if name == "fp32 (full)" else None)
        ax.axhspan(0, 75, color=FAIL_FILL, zorder=0)
        ax.axhline(80, color=INK, linewidth=1, linestyle=(0, (4, 3)))
        for size, r in R.items():
            h = _v(r)[name]["levels"]["0.8"]["raw"]["coverage_by_horizon"]
            ax.plot(range(1, len(h) + 1), [100 * c for c in h], color=COLOURS[size], linewidth=2,
                    marker=MARKERS[size], markersize=4.5, markevery=4, markeredgecolor=SURFACE,
                    label=f"Chronos-Bolt-{size}")
        ax.set_xlabel("hours ahead", color=MUTED, fontsize=9.5)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
        ax.set_ylim(max(0, 10 * int(low // 10) - 5), 100)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, "Own 80% intervals, no recalibration. Shaded: below 75%. At 3 bits the overall coverage hides "
             "large swings from one hour ahead to the next.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def make_all(results_dir: str, out_dir: str) -> list[str]:
    R = load(results_dir)
    if not R:
        raise SystemExit(f"no results.json found under {results_dir}")
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, p) for p in ("tables.md", "figure1_coverage_vs_bits.png", "figure2_coverage_by_hour.png")]
    write_tables(R, paths[0])
    figure1(R, paths[1])
    figure2(R, paths[2])
    return paths
