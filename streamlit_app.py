"""The live demo as a Streamlit page, so it can be hosted online and opened from a link.

Run it on your own laptop:   streamlit run streamlit_app.py
Hosted version:              Streamlit Community Cloud runs this same file from the GitHub repo.

It does no research work of its own. It only shows the choices (city, day, bits, fix),
calls `run_demo` from honesty_audit/demo.py, and displays the page that `render_html` builds.
"""

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from honesty_audit.cli import _load_data
from honesty_audit.demo import render_html, run_demo
from honesty_audit.models import load_model

DATA = "csv:data/open_meteo_10cities_2025-06-01_2026-10-31.csv:temperature_2m:timestamp:city"
MODEL = "chronos-bolt-tiny"  # the smallest size: fast, and it fits in a free server's memory
BITS = {"32-bit (full model)": 32, "16-bit": 16, "8-bit": 8, "6-bit": 6, "4-bit": 4, "3-bit": 3, "2-bit": 2}

st.set_page_config(page_title="Forecast Honesty Audit: live demo", layout="wide")


@st.cache_resource(show_spinner="Loading the forecasting model (first visit only) ...")
def load():
    """Load the data and the model ONCE; every visitor after that reuses them."""
    series, _ = _load_data(DATA, 0)
    return load_model(MODEL), series


model, series = load()
some = next(iter(series.values()))
first = pd.Timestamp(some.index[512]).ceil("D").date()  # first day with enough history before it
last = pd.Timestamp(some.index[-1]).date()

st.title("When does a smaller forecasting AI stop being honest?")
st.caption(f"Live demo with {model.name}. You choose; the forecast is made now. Nothing is pre-computed. "
           "Data: hourly temperature in 10 Indian cities (Open-Meteo).")

with st.form("choices"):
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    city = c1.selectbox("City", list(series), format_func=lambda c: c.replace("_", " ").title())
    day = c2.date_input("Day to forecast", value=last, min_value=first, max_value=last)
    bits = BITS[c3.selectbox("Shrink the model to", list(BITS), index=4)]
    fix = c4.checkbox("Also show the conformal fix", value=True)
    go = st.form_submit_button("Forecast", type="primary")

if go:
    try:
        with st.spinner("Forecasting ..."):
            result = run_demo(model, series, city, str(day), bits, fix=fix)
        components.html(render_html(result), height=3600, scrolling=True)
    except ValueError as e:  # for example a date without enough history for the fix
        st.error(str(e))
else:
    st.info("Choose a city, a day and a bit-width, then press Forecast.")
