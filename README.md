# Forecast honesty audit

**Research question:** how far can a time-series foundation model be compressed before its
prediction intervals stop being reliable, and how much further does conformal recalibration let it go?

Aavishkar 2026-27, Engineering & Technology (UG). Rin-gt and Soham.

When a forecasting model gives an 80% range, about 8 in 10 true values should land inside it.
This tool compresses a model step by step (16, 8, 6, 4, 3 and 2 bits, and optionally pruning),
measures whether that promise still holds, applies a conformal fix, and writes an interactive
**honesty report** that doubles as the live demo.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                                   # check the maths

# offline self-test: synthetic data + tiny stand-in network (NOT research results)
python -m honesty_audit run --model stand-in --data synthetic --context 168 --out results/selftest

# real audit: Chronos-Bolt on hourly temperature for 10 Indian cities
python -m honesty_audit run --model chronos-bolt-small \
    --data open-meteo-cities:2025-06-01,2026-10-31 --prune 0.3,0.5,0.7 --real-int8 \
    --out results/chronos_bolt_small_cities
```

Each run writes `results.json` (every number) and `honesty_report.html` (open in any browser,
works offline apart from fonts). In Colab, use `notebooks/run_audit_colab.ipynb`.

## What happens in a run

1. **Windows.** Each series is cut into non-overlapping 24-step windows. The earliest
   `n_calib` windows fit the fix; the later `n_test` windows judge it. Nothing from the future
   is used to calibrate the past.
2. **Compression.** For each level, an independent copy of the model is compressed.
   Quantization is *simulated*: weights are rounded to n bits and converted back, so accuracy
   and calibration are measured faithfully but there is no speed-up. `--real-int8` also times
   a genuinely int8 model on CPU. Each bit-width is run twice: all layers, and with input and
   output layers kept at full precision.
3. **Forecasts.** The model gives quantiles 0.1 to 0.9. The 80% range is 0.1 to 0.9 and the
   60% range is 0.2 to 0.8. Crossed quantiles are counted, then sorted back into order.
4. **The fix**, refitted for every compression level from that model's own errors:
   split conformal (CQR), weighted conformal (recent windows count more) and adaptive
   conformal inference. All are applied per step ahead.
5. **Scoring.** Coverage with a 95% confidence interval from a series-level bootstrap,
   coverage per step ahead, width, interval score, quantile loss, and the paired change versus
   the full model on the same windows.
6. **Verdict.** The pre-registered rule (saved in `results.json`): a level FAILS if the upper
   end of its coverage CI is below nominal minus the tolerance (default 0.05). The breaking
   A level that does not fail is labelled "covers, but too wide" if its interval score is more
   than 10% worse than the uncompressed model's. The breaking point is the first precision,
   going down from 16 bits, that fails or is too wide.

## What you may and may not claim

- May: "On these datasets, the 4-bit model's 80% ranges covered X% of true values (95% CI ...)."
- May: "The conformal fix restored empirical coverage on held-out windows."
- May not: "The fix guarantees 80% coverage." Time series are not exchangeable, so only
  measured coverage is reported. Adaptive conformal only guarantees long-run average coverage.
- The stand-in model and synthetic data exist to test the code. Never quote their numbers.

## Files

| Path | What it does |
| --- | --- |
| `honesty_audit/data.py` | loads CSV / Open-Meteo / synthetic data; cuts windows |
| `honesty_audit/models.py` | Chronos-Bolt, TimesFM-2.5 and stand-in wrappers |
| `honesty_audit/compress.py` | simulated quantization and magnitude pruning |
| `honesty_audit/conformal.py` | split, weighted and adaptive conformal fixes |
| `honesty_audit/metrics.py` | coverage, interval score, bootstrap CIs |
| `honesty_audit/audit.py` | the experiment and the pass/fail rule |
| `honesty_audit/report.py`, `report_template.html` | the honesty report / demo page |
| `tests/` | checks that the maths behaves as documented |
| `docs/WALKTHROUGH.md` | plain-language explanation of every part, for judges' questions |

## Known limits

- Simulated quantization does not measure real speed or memory on a phone.
- Open-Meteo values are reanalysis, not station readings, and are free for non-commercial use.
- `TimesFM25Forecaster` follows the TimesFM README's API; check it against the version you install.
- Coverage is pooled and per step ahead; conditional coverage (Decker & Tresp, 2026) is future work.

## Key references

Ansari et al. (2024) Chronos; Das et al. (2024) TimesFM; Romano et al. (2019) CQR;
Gibbs & Candès (2021) adaptive conformal inference; Barber et al. (2023) conformal prediction
beyond exchangeability; Gneiting & Raftery (2007) interval score; Gholami et al. (2021)
quantization survey; Decker & Tresp (2026) calibration of time-series foundation models.
