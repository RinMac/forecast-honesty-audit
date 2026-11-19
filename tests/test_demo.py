"""Tests for the live demo page and the local web server.

They use the small NumPy stand-in model and synthetic data, so they run offline in seconds.
The stand-in's numbers mean nothing; these tests only check that the demo is put together right.
"""

import re
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from honesty_audit.data import make_synthetic_load
from honesty_audit.demo import render_html, run_demo
from honesty_audit.models import TinyQuantileNet

DATE = "2026-06-01"  # day 92 of the synthetic data: enough history for the fix (512 h + 60 days)


@pytest.fixture(scope="module")
def setup():
    series = make_synthetic_load(n_series=3, n_days=120, seed=0)
    model = TinyQuantileNet(512, 24, hidden=32)
    model.fit([s.to_numpy()[:1500] for s in series.values()], epochs=60, log=None)
    return model, series


@pytest.mark.parametrize("bits", [32, 8, 4, 2])
def test_counts_add_up_for_every_model(setup, bits):
    model, series = setup
    d = run_demo(model, series, "feeder_1", DATE, bits, fix=True)
    assert len(d["truth"]) == len(d["hours"]) == 24
    for key in ("full", "small", "fixed"):
        b = d[key]
        assert len(b["hits"]) == 24
        assert b["inside"] == sum(b["hits"])
        assert b["inside"] + b["below"] + b["above"] == 24  # every hour is inside, colder or hotter
        assert all(np.isfinite(b["lo"])) and all(np.isfinite(b["hi"]))
    assert d["audit"] is None  # the stand-in has no saved full audit


def test_full_precision_choice_leaves_the_model_unchanged(setup):
    model, series = setup
    d = run_demo(model, series, "feeder_2", DATE, 32)
    assert d["small"]["lo"] == d["full"]["lo"] and d["small"]["hi"] == d["full"]["hi"]
    assert "fixed" not in d


def test_date_with_a_time_is_moved_to_midnight(setup):
    model, series = setup
    d = run_demo(model, series, "feeder_1", DATE + " 15:00", 4)
    assert d["date"] == DATE and d["hours"][0].endswith("00:00:00")


@pytest.mark.parametrize("city, date, bits, fix", [
    ("nowhere", DATE, 4, False),          # unknown city
    ("feeder_1", "2026-03-05", 4, False),  # not enough history before it
    ("feeder_1", "2026-04-01", 4, True),   # enough for the forecast, not for the fix
    ("feeder_1", "2027-01-01", 4, False),  # after the end of the data
    ("feeder_1", "not a date", 4, False),
    ("feeder_1", "", 4, False),
    ("feeder_1", DATE, 1, False),          # 1 bit is not allowed
])
def test_bad_choices_give_a_clear_error(setup, city, date, bits, fix):
    model, series = setup
    with pytest.raises(ValueError):
        run_demo(model, series, city, date, bits, fix=fix)


def test_page_hides_truth_but_shows_the_hour_table(setup):
    model, series = setup
    page = render_html(run_demo(model, series, "feeder_3", DATE, 4, fix=True))
    assert '<body class="hide">' in page  # truth hidden until "Reveal"
    assert page.count('class="reveal"') == 2  # one button at the top, one at the hour table
    body = page.split("<tbody>")[1].split("</tbody>")[0]
    assert body.count("<tr>") == 24  # the hour table is always there
    assert body.count('class="rng ') == 24 * 3  # three ranges per hour: full, compressed, recalibrated
    assert "<tfoot>" in page
    for bad in ("nan", "None", "inf°"):
        assert not re.search(rf"\b{bad}\b", page), bad


def test_page_escapes_text(setup):
    model, series = setup
    d = run_demo(model, series, "feeder_1", DATE, 4)
    d["city"] = "<script>x</script>"
    assert "<script>x</script>" not in render_html(d)


def test_local_web_server_answers(setup):
    from honesty_audit.app import make_server

    model, series = setup
    server = make_server(model, series, port=0)  # 0 = any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        get = lambda path: urllib.request.urlopen(base + path, timeout=60).read().decode("utf-8")
        assert "Forecast" in get("/")
        page = get(f"/demo?city=feeder_1&date={DATE}&bits=4&fix=1")
        assert "Live honesty check" in page and "Hour by hour" in page
        assert "unknown city" in get(f"/demo?city=nowhere&date={DATE}&bits=4")
        assert "bits must be" in get(f"/demo?city=feeder_1&date={DATE}&bits=abc")
        assert "bits must be" in get(f"/demo?city=feeder_1&date={DATE}&bits=1")
        with pytest.raises(urllib.error.HTTPError):
            get("/nothing-here")
    finally:
        server.shutdown()
        server.server_close()
