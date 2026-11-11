"""Live demo: a judge picks a city, a date and a bit-width; we forecast on the spot.

Nothing here is pre-computed. The model sees only the hours BEFORE the chosen date,
forecasts the 24 hours of that date, and we then compare its 80% range with what
really happened. The full model and the compressed model are shown side by side.

One day is only 24 values, so a single demo cannot prove or disprove the "80%"
promise. It shows how the audit works; the audit itself uses 3,650 such days.
That is why the page also shows, next to the live day, what the full audit found
for the same model and bit-width (read from the saved results, if they exist).
"""

from __future__ import annotations

import html
import json
import os

import numpy as np
import pandas as pd

from .compress import compress_
from .conformal import SplitCQR
from .metrics import series_scale

_AUDIT_CACHE: dict[str, dict | None] = {}


def _origin(s: pd.Series, date: str, need_before: int, horizon: int) -> int:
    """Index of 00:00 on `date`, checking there is enough data before and after it."""
    pos = s.index.get_indexer([pd.Timestamp(date)])[0]
    # first midnight with enough history before it, and the last full day in the data
    first_ok = pd.Timestamp(s.index[need_before]).ceil("D").date() if need_before < len(s) else None
    last_ok = pd.Timestamp(s.index[len(s) - horizon]).date()
    if pos < need_before or pos + horizon > len(s):
        raise ValueError(f"pick a date from {first_ok} to {last_ok} (the model needs history before it)")
    return int(pos)


def _range_80(model, contexts, horizon):
    """Forecast and return (lower, median, upper) of the 80% range, each [n, horizon]."""
    Q = np.sort(model.predict(contexts, horizon), axis=-1)
    lv = np.asarray(model.quantile_levels)
    pick = lambda q: Q[..., int(np.where(np.isclose(lv, q))[0][0])]
    return pick(0.1), pick(0.5), pick(0.9)


def audit_summary(model_name: str, bits: int, results_dir: str = "results") -> dict | None:
    """What the full pre-registered audit found for this model and bit-width (80% range).

    Reads results/full_<model>/results.json written by `python -m honesty_audit run`.
    Returns None when that file does not exist, so the demo still works without it.
    """
    path = os.path.join(results_dir, "full_" + model_name.replace("-", "_"), "results.json")
    if path not in _AUDIT_CACHE:
        try:
            with open(path) as fh:
                _AUDIT_CACHE[path] = json.load(fh)
        except OSError:
            _AUDIT_CACHE[path] = None
    res = _AUDIT_CACHE[path]
    if res is None:
        return None
    full = res["variants"][0]
    hit = [v for v in res["variants"] if not v["keep_io"] and (
        (bits >= 32 and v["method"] == "none") or (v["method"] == "quant" and v["level"] == bits))]
    if not hit:
        return None
    v = hit[0]
    out = {"n_days": res["meta"]["n_test_windows"], "name": v["name"],
           "error_vs_full": v["median_scaled_mae"] / full["median_scaled_mae"], "methods": {}}
    for m in ("raw", "split_cqr"):
        s = v["levels"]["0.8"][m]
        out["methods"][m] = {"mean": s["coverage"]["mean"], "lo": s["coverage"]["lo"], "hi": s["coverage"]["hi"],
                             "verdict": s["verdict"], "vs_full": s["vs_full_model"]["verdict"],
                             "score_ratio": s["score_vs_full_raw"]}
    return out


def run_demo(model, series: dict[str, pd.Series], city: str, date: str, bits: int,
             fix: bool = False, context_len: int = 512, horizon: int = 24,
             n_calib: int = 60, season: int = 24) -> dict:
    if city not in series:
        raise ValueError(f"unknown city {city!r}; choose from {', '.join(series)}")
    s = series[city]
    need = context_len + (n_calib * horizon if fix else 0)
    o = _origin(s, date, need, horizon)
    v = s.to_numpy(dtype=float)
    context, truth = v[o - context_len:o], v[o:o + horizon]

    small = model.clone()  # the original model is never changed
    rep = compress_(small.module, "quant" if bits < 32 else "none", bits, [])
    f_lo, f_med, f_hi = (a[0] for a in _range_80(model, [context], horizon))
    c_lo, c_med, c_hi = (a[0] for a in _range_80(small, [context], horizon))

    out = {
        "city": city, "date": str(pd.Timestamp(date).date()), "bits": bits, "model": model.name,
        "hours": [str(t) for t in s.index[o:o + horizon]],
        "history": context[-3 * horizon:].tolist(), "truth": truth.tolist(),
        "full": {"lo": f_lo.tolist(), "med": f_med.tolist(), "hi": f_hi.tolist()},
        "small": {"lo": c_lo.tolist(), "med": c_med.tolist(), "hi": c_hi.tolist()},
        # memory the weights need: 32 bits each for the full model, `bits` each when shrunk
        "size": {"params": rep.params_total, "full_mb": rep.params_total * 4 / 1e6,
                 "small_mb": rep.est_weight_megabytes},
        "context_days": context_len / 24,
    }
    if fix:
        # rolling split CQR, as pre-registered: fit on the n_calib days just before the
        # chosen date, all cities pooled, on values divided by each window's scale
        ctxs, ys = [], []
        for other in series.values():
            w = other.to_numpy(dtype=float)
            for k in range(1, n_calib + 1):
                p = o - k * horizon
                ctxs.append(w[p - context_len:p])
                ys.append(w[p:p + horizon])
        sc = np.array([series_scale(c, season) for c in ctxs])[:, None]
        lo, _, hi = _range_80(small, ctxs, horizon)
        cqr = SplitCQR(0.2).fit(lo / sc, hi / sc, np.array(ys) / sc)
        my = series_scale(context, season)
        x_lo, x_hi = cqr.apply(c_lo[None, :] / my, c_hi[None, :] / my)
        out["fixed"] = {"lo": (x_lo[0] * my).tolist(), "hi": (x_hi[0] * my).tolist(),
                        "med": c_med.tolist(), "calib_forecasts": len(ctxs),
                        "shift": float(np.mean(cqr.q) * my)}  # average widening per side, in degrees
    for key in ("full", "small", "fixed"):
        if key in out:
            lo, hi, med = (np.array(out[key][k]) for k in ("lo", "hi", "med"))
            out[key]["hits"] = [bool(a) for a in (truth >= lo) & (truth <= hi)]
            out[key]["inside"] = int(sum(out[key]["hits"]))
            out[key]["below"] = int((truth < lo).sum())  # truth colder than the whole range
            out[key]["above"] = int((truth > hi).sum())  # truth hotter than the whole range
            out[key]["width"] = float((hi - lo).mean())
            out[key]["error"] = float(np.abs(med - truth).mean())  # average miss of the middle forecast
    out["audit"] = audit_summary(model.name, bits)
    return out


# --------------------------------------------------------------------------- the page

VERDICT_TEXT = {
    "pass": "We are 95% confident its coverage is not below 75%.",
    "FAIL": "We are 95% confident its coverage is below 75%.",
    "covers, but too wide": "It reaches the coverage only with ranges clearly worse than "
                            "the full model's (interval score more than 10% worse), so it counts as broken.",
    "over-cautious": "Coverage is confidently above 85%: honest, but the ranges are wider than needed.",
    "inconclusive": "The data cannot tell whether coverage is above or below 75%.",
}


def _day_meaning(b: dict, full_width: float, H: int) -> str:
    """One plain sentence about how this model did on the chosen day."""
    need = int(np.ceil(0.8 * H))
    if b["width"] > 3 * full_width:
        return (f"Its range is {b['width'] / full_width:.0f} times wider than the full model's. A range that wide "
                f"catches almost anything, so a high count here does not mean it is honest.")
    if b["inside"] >= need:
        return f"On this day it kept its promise: an 80% range should catch about {need} of {H} values."
    side = "colder" if b["below"] > b["above"] else "hotter"
    return (f"On this day it fell short: an 80% range should catch about {need} of {H}. Most misses were on the "
            f"{side} side. One day is too few to judge; see the full audit below.")


def _coverage_bar(m: dict, col: str) -> str:
    """Small picture: measured coverage with its confidence interval, against the 80% promise."""
    W, L, R = 520, 6, 6
    X = lambda p: L + (W - L - R) * (p - 0.2) / 0.8  # the bar shows 20% .. 100%
    c = lambda p: min(max(p, 0.2), 1.0)
    return (f'<svg viewBox="0 0 {W} 56" class="bar" role="img" aria-label="Coverage {m["mean"]:.0%}">'
            f'<rect x="{L}" y="18" width="{W - L - R}" height="14" rx="3" class="track"/>'
            f'<rect x="{L}" y="18" width="{X(0.75) - L:.1f}" height="14" rx="3" class="failzone"/>'
            f'<rect x="{L}" y="18" width="{X(c(m["mean"])) - L:.1f}" height="14" rx="3" fill="{col}" fill-opacity=".75"/>'
            f'<line x1="{X(c(m["lo"])):.1f}" x2="{X(c(m["hi"])):.1f}" y1="25" y2="25" class="ci"/>'
            f'<line x1="{X(0.8):.1f}" x2="{X(0.8):.1f}" y1="10" y2="40" class="promise"/>'
            f'<text x="{X(0.8):.1f}" y="8" class="ax" text-anchor="middle">promise 80%</text>'
            f'<text x="{X(0.75) - 4:.1f}" y="52" class="ax" text-anchor="end">below 75%: fail zone</text>'
            f'<text x="{X(c(m["mean"])) + 6:.1f}" y="52" class="ax">{m["mean"]:.1%}</text></svg>')


def render_html(d: dict) -> str:
    """One self-contained page (no internet needed) with the chart, the numbers and what they mean."""
    H = len(d["truth"])
    hist = d["history"]
    e = html.escape
    name = "full model (no compression)" if d["bits"] >= 32 else f"{d['bits']}-bit model"
    bands = [("full", "#2f6fb0", "Full model"), ("small", "#d9822b", name.capitalize())]
    if "fixed" in d:
        bands.append(("fixed", "#2e8b57", f"{name.capitalize()} + conformal fix"))
    # the y-axis follows the history, the truth and the FULL model's range. A badly
    # compressed model can give ranges hundreds of degrees wide; those are cut off at the
    # chart edge (and we say so) instead of squashing everything else flat.
    vals = hist + d["truth"] + d["full"]["lo"] + d["full"]["hi"]
    pad = 0.6 * (max(vals) - min(vals))
    y0, y1 = min(vals) - pad, max(vals) + pad
    off_chart = [lab for k, _, lab in bands if min(d[k]["lo"]) < y0 or max(d[k]["hi"]) > y1]
    W, Ht, L, R, T, B = 920, 400, 52, 16, 16, 34
    n = len(hist) + H
    X = lambda i: L + (W - L - R) * i / (n - 1)
    Y = lambda v: T + (Ht - T - B) * (y1 - min(max(v, y0 - 50), y1 + 50)) / (y1 - y0)
    path = lambda xs, ys: " ".join(f"{'M' if k == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for k, (x, y) in enumerate(zip(xs, ys)))
    fx = list(range(len(hist), n))

    # ---- the chart
    svg = [f'<svg viewBox="0 0 {W} {Ht}" role="img" aria-label="Forecast ranges and true values">',
           f'<clipPath id="plot"><rect x="{L}" y="{T}" width="{W - L - R}" height="{Ht - T - B}"/></clipPath>']
    for t in np.linspace(y0, y1, 6):
        svg.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'
                   f'<text x="{L - 8}" y="{Y(t) + 4:.1f}" class="ax" text-anchor="end">{t:.0f}°</text>')
    svg.append(f'<line x1="{X(len(hist) - 0.5):.1f}" x2="{X(len(hist) - 0.5):.1f}" y1="{T}" y2="{Ht - B}" class="now"/>')
    svg.append(f'<text x="{X(len(hist) / 2):.1f}" y="{Ht - 10}" class="ax" text-anchor="middle">last 3 days (the model sees this)</text>')
    svg.append(f'<text x="{X(len(hist) + H / 2):.1f}" y="{Ht - 10}" class="ax" text-anchor="middle">{e(d["date"])} (hidden from the model)</text>')
    svg.append(f'<path d="{path(range(len(hist)), hist)}" class="hist"/>')
    for key, col, _ in bands:
        b = d[key]
        area = path(fx, b["hi"]) + " " + path(fx[::-1], b["lo"][::-1]).replace("M", "L", 1) + " Z"
        svg.append(f'<path d="{area}" fill="{col}" fill-opacity=".22" stroke="{col}" stroke-width="1.5" clip-path="url(#plot)"/>')
    last = bands[-1][0]  # dots are coloured against the last band shown (the fix, if it is on)
    for i, y in enumerate(d["truth"]):
        svg.append(f'<circle cx="{X(fx[i]):.1f}" cy="{Y(y):.1f}" r="4.5" class="truth {"in" if d[last]["hits"][i] else "out"}"/>')
    svg.append("</svg>")
    legend = '<span><i class="sw line"></i>temperature the model was shown</span>'
    legend += "".join(f'<span><i class="sw" style="background:{col}55;border:1.5px solid {col}"></i>{e(lab)}: its 80% range</span>'
                      for _, col, lab in bands)
    legend += ('<span class="truthonly"><i class="sw dot" style="background:#1b2430"></i>true value, inside</span>'
               '<span class="truthonly"><i class="sw dot" style="background:#c0392b"></i>true value, outside</span>')

    # ---- one card per model
    cards = ""
    max_w = max(d[k]["width"] for k, _, _ in bands)
    for key, col, label in bands:
        b = d[key]
        strip = "".join(f'<i class="{"hit" if h else "miss"}" title="{e(d["hours"][i][11:16])}"></i>' for i, h in enumerate(b["hits"]))
        cards += f"""<div class="card" style="border-top:4px solid {col}"><div class="lab">{e(label)}</div>
<div class="big truthonly">{b["inside"]} / {H} <small>= {b["inside"] / H:.0%}</small></div><div class="big hiddenonly">? / {H}</div>
<div class="sub">true values inside its 80% range</div>
<div class="strip truthonly" aria-hidden="true">{strip}</div><div class="strip hiddenonly" aria-hidden="true">{'<i></i>' * H}</div>
<div class="sub">one square per hour, 00:00 to 23:00: green = inside, red = outside</div>
<table class="mini">
<tr><td>Average width of the range</td><td>{b["width"]:.1f}°</td></tr>
<tr><td colspan="2"><div class="wbar"><div style="width:{max(100 * b["width"] / max_w, 1.5):.1f}%;background:{col}"></div></div></td></tr>
<tr class="truthonly"><td>True value was colder than the range</td><td>{b["below"]} h</td></tr>
<tr class="truthonly"><td>True value was hotter than the range</td><td>{b["above"]} h</td></tr>
<tr class="truthonly"><td>Average error of the middle forecast</td><td>{b["error"]:.2f}°</td></tr>
</table>
<p class="means truthonly"><b>What this means:</b> {e(_day_meaning(b, d["full"]["width"], H))}</p></div>"""

    # ---- model size
    sz = d["size"]
    ratio = sz["full_mb"] / max(sz["small_mb"], 1e-9)
    size_html = f"""<div class="box"><h2>How much smaller is the {e(name)}?</h2>
<table class="mini wide">
<tr><td>Full model (32 bits per weight)</td><td>{sz["full_mb"]:.1f} MB</td></tr>
<tr><td colspan="2"><div class="wbar"><div style="width:100%;background:#2f6fb0"></div></div></td></tr>
<tr><td>{e(name.capitalize())}</td><td>{sz["small_mb"]:.1f} MB</td></tr>
<tr><td colspan="2"><div class="wbar"><div style="width:{max(100 / ratio, 1):.1f}%;background:#d9822b"></div></div></td></tr>
</table>
<p class="means">The model has {sz["params"]:,} weights (numbers it learned). Storing each one with {d["bits"]} bits instead of 32
would make it <b>{ratio:.1f} times smaller</b>. Here the shrinking is simulated: the rounding errors are real, so accuracy and
honesty are measured faithfully, but the model in this demo does not actually use less memory.</p></div>"""

    # ---- what the full audit found
    audit_html = ""
    a = d.get("audit")
    if a:
        rows = ""
        for m, col, lab in (("raw", "#d9822b", "No fix"), ("split_cqr", "#2e8b57", "With the rolling conformal fix")):
            s = a["methods"][m]
            rows += f"""<div class="arow"><div class="lab">{lab}: {s["mean"]:.1%} <small>(95% confidence interval {s["lo"]:.1%} to {s["hi"]:.1%})</small></div>
{_coverage_bar(s, col)}<p class="means"><b>Verdict: {e(s["verdict"])}.</b> {e(VERDICT_TEXT.get(s["verdict"], ""))}
{"" if d["bits"] >= 32 else f"Interval score is {s['score_ratio']:.2f} times the full model's (above 1.10 is too much)."}</p></div>"""
        audit_html = f"""<div class="box"><h2>What the full audit found for the {e(name)}</h2>
<p>The same check as above, repeated on <b>{a["n_days"]:,} forecast days</b> (365 days in 10 cities), with rules fixed in
advance. This is the result that counts; the single day above only shows how the check works.</p>{rows}
<p class="means">Across those days the middle forecast's error was {a["error_vs_full"]:.2f} times the full model's
({"no change" if abs(a["error_vs_full"] - 1) < 0.005 else "more than 1 means less accurate"}).</p></div>"""

    # ---- hour by hour
    head = "".join(f"<th>{e(lab)}</th>" for _, _, lab in bands)
    trs = ""
    for i in range(H):
        cells = "".join(f'<td class="{"ok" if d[k]["hits"][i] else "no"}">{d[k]["lo"][i]:.1f}° to {d[k]["hi"][i]:.1f}° '
                        f'{"✓" if d[k]["hits"][i] else "✗"}</td>' for k, _, _ in bands)
        trs += f'<tr><td>{e(d["hours"][i][11:16])}</td><td><b>{d["truth"][i]:.1f}°</b></td>{cells}</tr>'
    fix_note = ""
    if "fixed" in d:
        f = d["fixed"]
        verb = "widened" if f["shift"] >= 0 else "narrowed"
        fix_note = (f'<p class="means"><b>How the fix worked here:</b> it looked at the {e(name)}\'s last '
                    f'{f["calib_forecasts"]} forecasts (60 days in 10 cities, all before this date), saw how far the true '
                    f'values fell outside its ranges, and {verb} today\'s range by about {abs(f["shift"]):.2f}° on each side.</p>')

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Live honesty check</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f6f7f9;color:#1b2430;line-height:1.45}}
main{{max-width:980px;margin:0 auto;padding:20px 16px}}
h1{{font-size:1.5rem;margin:0 0 4px}} h2{{font-size:1.1rem;margin:0 0 6px}} p{{margin:6px 0;color:#4a5665}}
.box{{background:#fff;border:1px solid #dde2e8;border-radius:10px;padding:14px;margin:14px 0}}
svg{{width:100%;height:auto;display:block}}
.grid{{stroke:#e6e9ee}} .ax{{font-size:12px;fill:#6a7686}} .now{{stroke:#9aa5b3;stroke-dasharray:4 4}}
.hist{{fill:none;stroke:#6a7686;stroke-width:1.6}}
.truth{{stroke:#fff;stroke-width:1.2}} .truth.in{{fill:#1b2430}} .truth.out{{fill:#c0392b}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.85rem;color:#4a5665;margin-top:8px}}
.sw{{display:inline-block;width:18px;height:11px;margin-right:6px;vertical-align:-1px;border-radius:2px}}
.sw.line{{height:0;border-top:2px solid #6a7686;vertical-align:4px}} .sw.dot{{width:10px;height:10px;border-radius:50%}}
.cards{{display:flex;gap:12px;flex-wrap:wrap}} .card{{flex:1 1 260px;background:#fff;border:1px solid #dde2e8;border-radius:10px;padding:14px}}
.lab{{font-weight:600}} .big{{font-size:2rem;font-weight:700;margin:4px 0 0}} .big small{{font-size:1rem;font-weight:600;color:#6a7686}}
.sub{{font-size:.8rem;color:#6a7686}}
.strip{{display:flex;gap:2px;margin:8px 0 2px}} .strip i{{flex:1;height:14px;border-radius:2px;background:#d5dae1}}
.strip i.hit{{background:#2e8b57}} .strip i.miss{{background:#c0392b}}
table.mini{{width:100%;border-collapse:collapse;font-size:.88rem;margin-top:10px}} table.mini td{{padding:3px 0}}
table.mini td:last-child{{text-align:right;font-weight:600;white-space:nowrap}} table.wide{{max-width:560px}}
.wbar{{background:#eef0f3;border-radius:3px;height:8px}} .wbar div{{height:8px;border-radius:3px}}
.means{{font-size:.88rem;background:#f6f7f9;border-radius:8px;padding:8px 10px;margin:10px 0 0;color:#1b2430}}
.arow{{margin:12px 0}} .bar{{max-width:560px}} .track{{fill:#eef0f3}} .failzone{{fill:#c0392b;fill-opacity:.13}}
.ci{{stroke:#1b2430;stroke-width:3}} .promise{{stroke:#1b2430;stroke-width:1.5;stroke-dasharray:3 3}}
table.hours{{width:100%;border-collapse:collapse;font-size:.85rem}} table.hours th,table.hours td{{padding:4px 8px;border-bottom:1px solid #e6e9ee;text-align:left}}
td.ok{{color:#1f6b42}} td.no{{color:#c0392b;font-weight:600}}
dl{{margin:0;font-size:.9rem}} dt{{font-weight:600;margin-top:8px}} dd{{margin:0;color:#4a5665}}
button{{font:inherit;padding:10px 16px;border-radius:8px;border:1px solid #1b2430;background:#1b2430;color:#fff;cursor:pointer}}
body.hide .truth, body.hide .truthonly{{display:none}} body:not(.hide) .hiddenonly{{display:none}}
.note{{font-size:.85rem}} .tablewrap{{overflow-x:auto}}
</style></head><body class="hide"><main>
<h1>Live honesty check: {e(d["city"].replace("_", " ").title())}, {e(d["date"])}</h1>
<p><b>What is happening:</b> the forecasting AI ({e(d["model"])}) was shown the hourly temperature of the
{d["context_days"]:.0f} days before this date and nothing after. It forecast the 24 hours of this date. It does not give one
number per hour but a range, and promises: "I'm 80% sure the true temperature will be inside". We then shrank the model and
asked it the same question. Each coloured band below is one model's 80% range.</p>
<p><button id="reveal" type="button">Reveal the true values</button>
<span class="sub hiddenonly"> Guess first: how many of the 24 true values will land inside each band?</span></p>
<div class="box">{"".join(svg)}<div class="legend">{legend}</div></div>
<div class="cards">{cards}</div>
<p class="note">One day is only {H} values, so a single day cannot prove or disprove the 80% promise: even a
perfectly honest model can land well above or below 19 of 24 on any one day.{(" <b>Off the chart:</b> the range of the " + e(" and the ".join(off_chart).lower()) + " is so wide that it runs past the top or bottom edge.") if off_chart else ""}</p>
{fix_note}
{audit_html}
{size_html}
<div class="box truthonly"><h2>Hour by hour</h2>
<div class="tablewrap"><table class="hours"><thead><tr><th>Hour</th><th>True value</th>{head}</tr></thead><tbody>{trs}</tbody></table></div></div>
<div class="box hiddenonly"><h2>Hour by hour</h2><p>Reveal the true values to see every hour's range and whether it held.</p></div>
<div class="box"><h2>What the words mean</h2><dl>
<dt>80% range (prediction interval)</dt><dd>A lower and an upper value. The model claims the truth will fall between them 80% of the time.</dd>
<dt>Coverage</dt><dd>How often the truth really fell inside. An honest 80% range has coverage close to 80%.</dd>
<dt>Bits (quantization)</dt><dd>How precisely each learned number in the model is stored. 32 bits is full precision; fewer bits make the model smaller and cheaper to run, but add rounding errors.</dd>
<dt>Width</dt><dd>Upper value minus lower value. Narrow is useful, but only if the coverage holds. A very wide range is "right" without telling you anything.</dd>
<dt>Conformal fix</dt><dd>A correction that widens or narrows the range using the model's own recent misses. It needs no retraining.</dd>
<dt>Interval score</dt><dd>One number that rewards narrow ranges and punishes every miss. Lower is better. We use it to catch ranges that cover only by being huge.</dd>
<dt>95% confidence interval</dt><dd>How sure we are of a measured coverage. Narrow interval = lots of data = a firm number.</dd>
</dl></div>
</main><script>
document.getElementById("reveal").addEventListener("click", function () {{
  var hidden = document.body.classList.toggle("hide");
  this.textContent = hidden ? "Reveal the true values" : "Hide the true values";
}});
</script></body></html>"""


def write_demo(d: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"demo_{d['city']}_{d['date']}_{d['bits']}bit.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(d))
    return path
