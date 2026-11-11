# Code walkthrough: what each part does and why

Read this before the zonal round. For every file there is a plain explanation, the one idea a
judge is most likely to probe, and a question to practise answering out loud.

---

## 1. `data.py`: where the numbers come from and how we cut them

**What it does.** Loads series (a CSV, Open-Meteo weather, or synthetic test data) and cuts each
into *windows*: the model sees `context_len` past points and must forecast the next `horizon`
points.

**Why windows don't overlap.** If window 2 started one hour after window 1, their targets would
share 23 of 24 hours. They would be near-copies, and we would think we had far more evidence than
we really do.

**Why calibration comes first in time.** The fix is fitted on the *earlier* windows and judged on
the *later* ones. Fitting on later data would let the future leak into the past, which would make
the fix look better than it is.

**Why several cities.** Our confidence intervals resample whole series. With one series there is
nothing to resample and the interval collapses to a single number.

> Practise: "Why not just use one long series from Navi Mumbai?"

---

## 2. `models.py`: the forecasting models

**What it does.** Wraps each model so the rest of the code can ask the same question:
"here are past values, give me quantiles 0.1, 0.2 ... 0.9 for the next 24 steps."

- **Chronos-Bolt** (Amazon) predicts those nine quantiles directly. It cannot give a 95% range,
  because its widest quantiles are 0.1 and 0.9. So we audit 80% (0.1 to 0.9) and 60% (0.2 to 0.8).
- **TimesFM-2.5** (Google) returns a mean plus the same nine quantiles.
- **TinyQuantileNet** is a small network we train in seconds, only to test the code offline.

**Key idea: a quantile.** The 0.9 quantile is the value the model thinks has a 90% chance of not
being exceeded. The range between the 0.1 and 0.9 quantiles should contain the truth 80% of the time.

> Practise: "Why 80% and not 95%?"

---

## 3. `compress.py`: shrinking the model

**Quantization.** A weight like 0.4372 is normally stored with 32 bits. At 4 bits a weight can only
take 15 possible values, so 0.4372 might become 0.4286. We round every weight in the model's
linear layers this way. Each row of a weight matrix gets its own scale (largest |weight| / 7 at
4 bits), so big and small rows are both represented well.

**Why "simulated".** We round the weights and convert them back to ordinary numbers. The model
then makes exactly the same mistakes a 4-bit model would, so accuracy and honesty are measured
correctly, but it does not run faster. We say so in every report. `--real-int8` times a genuinely
int8 model to show the real speed-up exists.

**Why keep input and output layers.** Earlier work on TimesFM (Pavlova et al., 2026) found these
layers are the most sensitive. The output layer is the one that *produces the quantiles*, so it
matters most for honesty. We test both settings.

**Pruning.** Sets the smallest weights to zero. Without special hardware this saves no time or
memory, so we report pruning only for its effect on honesty.

> Practise: "Your 4-bit model isn't actually faster. So what's the point?"
> (Answer: the honesty damage happens in the rounding, which we reproduce exactly; real 4-bit
> runtimes do the same rounding. We measure speed separately for int8.)

---

## 4. `conformal.py`: the fix

**The score.** For each past window: `score = max(low - truth, truth - high)`. Positive means the
truth fell outside the range by that much; negative means it was inside with room to spare.

**Split conformal (CQR).** Take the scores from the calibration windows, pick the one that is just
above 80% of them, and widen every future range by that amount. If the model was over-cautious,
the amount is negative and the range gets narrower. We do this separately for each hour ahead,
because errors 20 hours ahead are bigger than errors 1 hour ahead.

**Weighted conformal.** Same, but recent windows count more. If the data drift, old windows are
less relevant (Barber et al., 2023).

**Adaptive conformal (ACI).** Walks forward in time. Every time the range misses, the next range
gets a little wider; every time it covers, a little narrower (Gibbs & Candès, 2021).

**The honesty rule for the fix itself.** Split conformal *guarantees* 80% only if the data are
exchangeable (order doesn't matter). Time series are not. So we only ever say "measured coverage".
ACI's guarantee is about the long-run average, not every moment.

> Practise: "Conformal prediction already exists. What's new?"
> (Answer: the fix is known; how much extra compression it buys for these models is what we measure.)

---

## 5. `metrics.py`: how honesty is scored

- **Coverage**: share of true values inside the range. Target: the promised level.
- **Width**: how wide the range is. A range from minus to plus infinity always covers and is useless.
- **Interval score** (Gneiting & Raftery, 2007): width plus a penalty for every miss. It cannot be
  gamed by making ranges huge or tiny. Lower is better.
- **Crossing rate**: compression can push the 0.1 quantile above the 0.2 one. We count how often.
- **Cluster bootstrap**: to get a 95% confidence interval we resample whole series many times,
  because hours inside one series are correlated and are not independent evidence.
- **Paired difference**: compressed minus full model *on the same windows*, which removes the noise
  of easy versus hard days.

> Practise: "How do you know 76% isn't just bad luck?"

---

## 6. `audit.py`: the experiment and the verdict

For each compression level: copy the model, compress the copy, forecast every window, build the
80% and 60% ranges, fit each fix on the calibration windows, score the test windows.

**The pre-registered rule.** Decided before running and saved in `results.json`:
an 80% range **fails** if we are 95% confident its coverage is below 75%
(the upper end of the confidence interval is under 0.75).
**Breaking point** = the first failing precision as we go down from 16 bits.

**Usefulness check:** a level that does not fail is still labelled "covers, but too wide" when its
interval score is more than 10% worse than the uncompressed model's (with no fix). This stops a
model from "passing" by giving huge ranges. Such a level also counts as a breaking point.

> Practise: "Why 5 percentage points? Isn't that arbitrary?"
> (Answer: any threshold is a choice; what matters is that we fixed it before seeing results and
> report every number, so a reader can apply their own.)

---

## 7. `report.py` and the honesty report page

Turns `results.json` into one HTML file. The slider walks down the bit-widths, the band is the
range, red dots are misses. "Hide the true values" lets you ask the audience to guess before
revealing. The tables show the breaking points and every level with confidence intervals.

The page labels itself "Pipeline test, not a research result" whenever it was made with the
stand-in model or synthetic data.

---

## Things to be honest about if asked

- The audit was built with help from an AI assistant; you designed the question, the rule, and
  ran and interpreted the experiments. Be ready to explain any line.
- Open-Meteo data are reanalysis, not thermometer readings.
- Quantization is simulated except the int8 timing.
- We test two models and a limited set of datasets; results may differ elsewhere.
