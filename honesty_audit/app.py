"""A small local page for the live demo: dropdowns instead of typing in a terminal.

It runs ONLY on this laptop (address 127.0.0.1) and needs no internet and no extra
library: the web server is part of Python itself. The page has a form (city, date,
bits, fix). Pressing "Forecast" asks this program for /demo?city=...&date=..., which
runs the same `run_demo` as the terminal command and sends back the same page.
"""

from __future__ import annotations

import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .demo import render_html, run_demo

BITS = [(32, "32-bit (full model)"), (16, "16-bit"), (8, "8-bit"), (6, "6-bit"),
        (4, "4-bit"), (3, "3-bit"), (2, "2-bit")]


def _form_page(model_name: str, cities: list[str], first: str, last: str) -> str:
    city_opts = "".join(f'<option value="{html.escape(c)}">{html.escape(c.replace("_", " ").title())}</option>'
                        for c in cities)
    bit_opts = "".join(f'<option value="{b}"{" selected" if b == 4 else ""}>{label}</option>' for b, label in BITS)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Auditing Uncertainty in Time-Series AI Models: live demo</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f6f7f9;color:#1b2430}}
header{{background:#1b2430;color:#fff;padding:14px 20px}} h1{{font-size:1.25rem;margin:0}}
header p{{margin:4px 0 0;color:#c5ccd6;font-size:.9rem}}
form{{display:flex;flex-wrap:wrap;gap:14px;align-items:end;padding:14px 20px;background:#fff;border-bottom:1px solid #dde2e8}}
label{{display:flex;flex-direction:column;font-size:.85rem;font-weight:600;gap:4px}}
select,input[type=date]{{font:inherit;padding:8px 10px;border:1px solid #b8c0cb;border-radius:8px;background:#fff}}
label.check{{flex-direction:row;align-items:center;gap:8px;padding-bottom:8px}}
button{{font:inherit;font-weight:600;padding:10px 20px;border-radius:8px;border:0;background:#2f6fb0;color:#fff;cursor:pointer}}
iframe{{width:100%;height:calc(100vh - 150px);min-height:620px;border:0;display:block}}
</style></head><body>
<header><h1>Auditing Uncertainty in Time-Series AI Models</h1>
<p>When does a smaller forecasting AI stop being honest? Live demo with {html.escape(model_name)}. You choose; the laptop forecasts now, offline. Nothing is pre-computed.</p></header>
<form action="/demo" target="out" method="get">
<label>City<select name="city">{city_opts}</select></label>
<label>Day to forecast<input type="date" name="date" min="{first}" max="{last}" value="{last}" required></label>
<label>Shrink the model to<select name="bits">{bit_opts}</select></label>
<label class="check"><input type="checkbox" name="fix" value="1" checked> also show conformal recalibration (widens or narrows the range)</label>
<button type="submit">Forecast</button>
</form>
<iframe name="out" title="Forecast result" srcdoc="&lt;p style='font-family:system-ui;padding:24px;color:#4a5665'&gt;Choose a city, a day and a bit-width, then press Forecast.&lt;/p&gt;"></iframe>
</body></html>"""


def _message_page(text: str) -> str:
    return (f'<!doctype html><meta charset="utf-8"><p style="font-family:system-ui;padding:24px;color:#c0392b">'
            f"{html.escape(text)}</p>")


def serve(model, series: dict, port: int = 8765, open_browser: bool = True) -> None:
    some = next(iter(series.values()))
    first = str(some.index[512].ceil("D").date())  # first day with enough history before it
    last = str(some.index[-1].date())
    form = _form_page(model.name, list(series), first, last)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                body = form
            elif url.path == "/demo":
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                try:
                    res = run_demo(model, series, q.get("city", ""), q.get("date", ""),
                                   int(q.get("bits", 4)), fix="fix" in q)
                    body = render_html(res)
                except ValueError as e:  # wrong city or a date without enough history
                    body = _message_page(str(e))
            else:
                self.send_error(404)
                return
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):  # keep the terminal quiet during the demo
            pass

    address = f"http://127.0.0.1:{port}"
    print(f"[app] ready at {address}  (press Ctrl+C here to stop)")
    if open_browser:
        import webbrowser
        webbrowser.open(address)
    # "Threading" = it can answer a second request while the first is still open, so a
    # browser that keeps a spare connection waiting cannot freeze the page
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
