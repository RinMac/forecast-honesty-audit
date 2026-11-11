"""Build the interactive "honesty report" page from audit results.

The page is a single self-contained HTML file (no internet needed once written),
so it can be opened on a laptop at the venue even without Wi-Fi.
It is also the live demo: a slider walks down the bit-widths, the band shows the
model's 80% range, and a counter checks the promise against the true values.
"""

from __future__ import annotations

import json
import math
import os

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "report_template.html")


def _clean(obj):
    """JSON cannot hold inf/nan; replace them with None."""
    if isinstance(obj, float):
        return None if (math.isinf(obj) or math.isnan(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def page_payload(results: dict) -> dict:
    """The subset of results the page needs (per-window arrays are left out)."""
    return _clean({k: v for k, v in results.items() if k != "per_window"})


def write_report(results: dict, out_dir: str, filename: str = "honesty_report.html") -> str:
    with open(TEMPLATE_PATH, encoding="utf-8") as fh:
        html = fh.read()
    data = json.dumps(page_payload(results), separators=(",", ":"))
    data = data.replace("</", "<\\/")  # never let the data close the <script> tag
    html = html.replace("/*__AUDIT_DATA__*/null", data)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path
