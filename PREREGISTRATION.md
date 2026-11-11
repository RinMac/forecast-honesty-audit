# Pre-registration: Forecast Honesty Audit

**Project:** Auditing the uncertainty of compressed time-series foundation models: when does a
smaller forecasting AI stop being honest?

**Authors:** Rishi and Soham, S.Y. B.Sc. Computer Science (Honours), SIES (Nerul) College of
Arts, Science and Commerce (Autonomous), Navi Mumbai.

**Mentor:** Dr. Fatema Kothari.

**Event:** University of Mumbai, 21st Aavishkar Research Convention 2026-27, Category 5, UG level.

**Date written:** 11 November 2026. The date that counts is the timestamp of the public Git
commit that contains this file.

**Status: FINAL. Approved for publication on 11 November 2026.**

This document fixes our models, data, tests and pass/fail rules **before** we run the real
audit. We wrote it so that nobody, including us, can adjust the rules after seeing the results.

---

## 1. Research question (unchanged)

How far can a time-series foundation model be compressed before its prediction intervals stop
being reliable, and how much further does conformal recalibration let it go?

## 2. What we expect (hypotheses)

- **H1.** Weight quantization lowers the measured coverage of the model's 80% range, and there is
  a bit-width at which the range fails our rule (section 7).
- **H2.** Coverage is damaged at a higher bit-width than point accuracy. "Accuracy is damaged"
  means the median forecast's scaled error is more than 10% worse than the full model's.
- **H3.** Conformal recalibration moves the breaking point to a lower bit-width.
- **H4.** Keeping the input and output layers at full precision moves the breaking point to a
  lower bit-width.

We will report the result of each hypothesis whether it is supported or not.

## 3. Models

Chronos-Bolt from Amazon, loaded with `chronos-forecasting` 2.3.2, in float32, on CPU:
`amazon/chronos-bolt-tiny`, `-mini`, `-small`, `-base`. We will record the exact Hugging Face
revision of each model in the results.

Chronos-Bolt gives quantiles 0.1 to 0.9 only. We therefore audit the **80% range** (0.1 to 0.9)
and the **60% range** (0.2 to 0.8).

Chronos-Bolt was released in November 2024. All our data starts in June 2025, so the model cannot
have seen it during training.

## 4. Data

- **Main data:** hourly air temperature at 2 m for 10 Indian cities (Navi Mumbai, Pune, Delhi,
  Kolkata, Chennai, Bengaluru, Ahmedabad, Jaipur, Lucknow, Guwahati), 1 June 2025 to
  31 October 2026, from the Open-Meteo historical archive. These are reanalysis values, not
  station readings.
- **Frozen copy:** `data/open_meteo_10cities_2025-06-01_2026-10-31.csv`, downloaded 11 Nov 2026.
  12,432 hours per city, no gaps.
- **SHA-256 checksum of that file:**
  `0685492f047c9e9e888fcaef83fe5c86579ff4be74b6da8b1811b19bd90ec108`

## 5. Windows

- Each forecast sees 512 hours of history and predicts the next 24 hours.
- Windows do not overlap: one forecast per city per day, starting at 00:00.
- **Test windows:** 365 per city, 1 November 2025 to 31 October 2026 (3,650 windows,
  87,600 hourly values).
- **Calibration windows:** the 60 per city just before that, 2 September to 31 October 2025.
- No value from a test window, or later, is ever used to fit anything applied to that window.

## 6. Compression

- **Simulated weight quantization** at 16, 8, 6, 4, 3 and 2 bits. Symmetric, one scale per output
  channel, round to nearest, linear layers only. Activations are not quantized.
- Each bit-width below 16 is run twice: all linear layers, and with `input_patch_embedding` and
  `output_patch_embedding` kept at full precision.
- **Magnitude pruning** at 30%, 50% and 70% (secondary).
- **Real int8** dynamic quantization on CPU, to check the simulated 8-bit result and to time it.

Our findings apply to this simple quantization method. Stronger methods may survive lower
bit-widths; we do not test them.

## 7. Rules (the part that must not change)

All coverage numbers are measured on the test windows. Every confidence interval (CI) is a 95%
interval from a bootstrap that resamples whole cities (2,000 resamples, seed 0).

**Tolerance:** 5 percentage points.

### 7.1 Verdict against the promise

For a range that promises coverage `p` (0.80 or 0.60):

| Verdict | Condition |
| --- | --- |
| **FAIL** | upper end of the coverage CI is below `p - 0.05` |
| **PASS** | lower end of the coverage CI is at or above `p - 0.05` |
| **INCONCLUSIVE** | anything in between: the data cannot tell |
| also **over-cautious** | lower end of the coverage CI is above `p + 0.05` |

### 7.2 Verdict against the full model

For each compressed model we take its coverage minus the uncompressed model's coverage, on the
same windows and with the same method (raw with raw, fixed with fixed).

| Verdict | Condition |
| --- | --- |
| **WORSE THAN FULL** | upper end of the CI of the difference is below `-0.05` |
| **AS GOOD AS FULL** | lower end of the CI of the difference is at or above `-0.05` |
| **INCONCLUSIVE** | anything in between |

### 7.3 "Covers, but too wide"

A range can reach its coverage just by becoming very wide, which is useless. So we also use the
interval score (Gneiting & Raftery, 2007), scaled per series. A level that does not FAIL under
7.1 is labelled **covers, but too wide** if its mean interval score is more than **10%** worse
than that of the uncompressed model with no fix applied.

### 7.4 Breaking point

Going down from 16 bits, for each model, range and method:

- **Breaks at:** the first bit-width that is FAIL (7.1) or "covers, but too wide" (7.3).
- **Safe down to:** the lowest bit-width for which it and every higher bit-width is PASS and not
  too wide.
- Bit-widths in between are reported as inconclusive. We will not call them safe.

We report the breaking point under 7.1 and, separately, the first bit-width that is WORSE THAN
FULL under 7.2. If the uncompressed model itself is FAIL under 7.1, we will say so plainly; the
comparison in 7.2 then carries the answer to how much compression adds.

## 8. The fixes (conformal recalibration)

All fixes work separately for each of the 24 steps ahead, are refitted for every compressed
model, and use the CQR score (Romano et al., 2019) divided by each window's series scale, so
that cities with large and small swings can share one calibration set.

- **Rolling split CQR (main fix):** for each test day, fitted on the 60 days just before it,
  all cities pooled.
- **Static split CQR:** fitted once on the 60 calibration days and never updated.
- **Weighted CQR** (Barber et al., 2023), weight 0.9 per day of age, on the same rolling 60 days.
- **Adaptive conformal inference** (Gibbs & Candès, 2021), step size 0.05, run per city in time
  order.

Time series are not exchangeable, so none of these comes with a coverage guarantee here. We
report measured coverage only.

## 9. Main and secondary results

**Main result:** the 80% range, quantization of all layers, with no fix and with rolling split
CQR, for each of the four model sizes.

**Secondary (reported, but not used to choose the headline):** the 60% range; input/output layers
kept; pruning; the other three fixes; coverage per month; coverage per step ahead; misses below
and above the range counted separately; quantile crossing rate; weighted quantile loss; width;
simulated 8-bit against real int8; a seasonal-naive baseline.

## 10. Check on days that do not exist yet

After this file is committed, and shortly before the zonal round (27 November 2026), we will
download the days after the commit date for the same 10 cities and run the same audit on them
with the same rules. Calibration for those days uses the rolling 60 days before each one.

This sample will be small (about two weeks). It is a supporting check, not the main result. Its
purpose is to show the rules were tested on data that was in the future when they were written.

## 11. What we had already seen when we wrote this

- A self-test on **synthetic** data with a small stand-in network. Not real results.
- One quick check that the code runs (11 Nov 2026): Chronos-Bolt-tiny, 16-bit and 4-bit only,
  5 calibration days and 5 test days per city (22 to 31 October 2026). Those 10 days are inside
  the test year above. We saw that the full model covered about 73% on them, that 4-bit covered
  less, and that static CQR did not help. These observations influenced sections 7.2, 7.3 and 8.
- The same quick check was run once more on the same 10 days after the rules in sections 7 and 8
  were coded, to confirm the code works. On those 5 test days rolling split CQR gave slightly
  higher coverage than no fix. Five days is far too few to conclude anything.
- We had seen no result for any other model size, bit-width or period.

## 12. Known limits

- Ten cities is a small number of independent series, and their weather is related. The
  confidence intervals may be somewhat too narrow.
- One kind of data (temperature) and one model family. We do not claim the results hold for
  other data or other models.
- Quantization is simulated, which measures accuracy and calibration but not speed.

## 13. If we deviate

Any change from this plan will be listed in a dated "Deviations" section at the end of this
file, with the reason, in a new commit. The original text stays in the Git history.

## 14. Use of AI

The audit code was written with the help of Claude (Anthropic), which also acted as a reviewer
and helped draft this document. The decisions in sections 5, 7 and 8 were made by the authors.
