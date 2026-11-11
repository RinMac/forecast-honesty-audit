"""Loading time series and cutting them into forecast windows.

Key idea: we never let the future leak into the past.
  * Forecast windows do not overlap (each origin is `horizon` steps after the last),
    so neighbouring test points are not near-copies of each other.
  * For every series, the EARLIER windows are used for calibration (fitting the
    conformal fix) and the LATER windows for testing.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Window:
    """One forecasting problem: show the model `context`, ask it to predict `target`."""

    series_id: str
    origin: int  # index in the series where the forecast starts
    context: np.ndarray  # past values the model sees
    target: np.ndarray  # true future values (hidden from the model)
    split: str  # "calib" or "test"
    timestamps: list[str] = field(default_factory=list)  # timestamps of the target, for plots


# --------------------------------------------------------------------------- loading


def load_csv(path: str, value_col: str, time_col: str | None = None,
             series_col: str | None = None) -> dict[str, pd.Series]:
    """Read a CSV. With `series_col`, one file can hold many series (long format)."""
    df = pd.read_csv(path)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
    out: dict[str, pd.Series] = {}
    groups = df.groupby(series_col) if series_col else [("series_0", df)]
    for sid, g in groups:
        g = g.sort_values(time_col) if time_col else g
        s = pd.Series(g[value_col].astype(float).to_numpy(),
                      index=pd.DatetimeIndex(g[time_col]) if time_col else None)
        out[str(sid)] = s
    return out


def fetch_open_meteo(latitude: float, longitude: float, start: str, end: str,
                     variable: str = "temperature_2m", name: str = "open_meteo") -> dict[str, pd.Series]:
    """Hourly historical weather from Open-Meteo's free archive API (no key needed).

    Values are model reanalysis, not direct station readings. Free for non-commercial use.
    Example (Nerul, Navi Mumbai): fetch_open_meteo(19.03, 73.02, "2025-01-01", "2026-10-31")
    """
    query = urllib.parse.urlencode({
        "latitude": latitude, "longitude": longitude, "start_date": start,
        "end_date": end, "hourly": variable, "timezone": "Asia/Kolkata",
    })
    url = f"https://archive-api.open-meteo.com/v1/archive?{query}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.load(resp)
    s = pd.Series(payload["hourly"][variable], index=pd.to_datetime(payload["hourly"]["time"]),
                  dtype=float)
    return {name: s.interpolate(limit=6).dropna()}


INDIAN_CITIES = {
    "navi_mumbai": (19.03, 73.02), "pune": (18.52, 73.86), "delhi": (28.61, 77.21),
    "kolkata": (22.57, 88.36), "chennai": (13.08, 80.27), "bengaluru": (12.97, 77.59),
    "ahmedabad": (23.02, 72.57), "jaipur": (26.91, 75.79), "lucknow": (26.85, 80.95),
    "guwahati": (26.14, 91.74),
}


def fetch_open_meteo_many(start: str, end: str, cities: dict | None = None,
                          variable: str = "temperature_2m") -> dict[str, pd.Series]:
    """One series per city. The bootstrap resamples whole series, so we need several:
    with a single series the confidence interval would be meaningless."""
    out = {}
    for name, (lat, lon) in (cities or INDIAN_CITIES).items():
        out.update(fetch_open_meteo(lat, lon, start, end, variable, name=name))
    return out


def make_synthetic_load(n_series: int = 6, n_days: int = 120, seed: int = 0,
                        drift_after: float = 0.88) -> dict[str, pd.Series]:
    """Hourly electricity-demand-like series, for testing and offline demos ONLY.

    Each series has a daily cycle, a weekly cycle, noise, and a level shift partway
    through (after `drift_after` of its length) so that the data are not
    exchangeable - just like real demand during a heatwave.
    These are synthetic numbers and must never be reported as real results.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days * 24)
    idx = pd.date_range("2026-03-01", periods=len(t), freq="h")
    out = {}
    for i in range(n_series):
        base = rng.uniform(800, 3000)
        daily = 0.18 * base * np.sin(2 * np.pi * (t - 6 - rng.uniform(0, 3)) / 24)
        weekly = 0.06 * base * np.sin(2 * np.pi * t / (24 * 7) + rng.uniform(0, 6))
        noise = rng.normal(0, 0.04 * base, len(t))
        # AR(1)-style smoothing so the noise is autocorrelated, like real load
        for k in range(1, len(t)):
            noise[k] = 0.6 * noise[k - 1] + noise[k]
        shift = np.where(t > drift_after * len(t), rng.uniform(0.08, 0.2) * base, 0.0)
        out[f"feeder_{i + 1}"] = pd.Series(base + daily + weekly + noise + shift, index=idx)
    return out


# --------------------------------------------------------------------------- windows


def make_windows(series: dict[str, pd.Series], context_len: int, horizon: int,
                 n_calib: int, n_test: int) -> list[Window]:
    """Cut each series into non-overlapping windows; earliest `n_calib` go to calibration.

    Windows are taken from the END of each series so the most recent data are tested.
    Series too short for n_calib + n_test windows are skipped (and reported).
    """
    windows: list[Window] = []
    skipped = []
    n_total = n_calib + n_test
    for sid, s in series.items():
        values = np.asarray(s.values, dtype=float)
        if np.isnan(values).any():
            values = pd.Series(values).interpolate(limit_direction="both").to_numpy()
        needed = context_len + n_total * horizon
        if len(values) < needed:
            skipped.append(sid)
            continue
        first_origin = len(values) - n_total * horizon
        has_time = isinstance(s.index, pd.DatetimeIndex)
        for k in range(n_total):
            origin = first_origin + k * horizon
            ts = ([str(x) for x in s.index[origin:origin + horizon]] if has_time else [])
            windows.append(Window(
                series_id=sid,
                origin=origin,
                context=values[origin - context_len:origin],
                target=values[origin:origin + horizon],
                split="calib" if k < n_calib else "test",
                timestamps=ts,
            ))
    if skipped:
        print(f"[data] skipped {len(skipped)} series that were too short: {skipped[:5]}")
    if not windows:
        raise ValueError("No windows could be made: series are shorter than context + windows.")
    return windows
