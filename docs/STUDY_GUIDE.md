# Study guide: the whole project, explained

For Rishi and Soham. Read this until you can explain every part without looking. It is written
in plain language, and it answers the questions you are likely to be asked, including the small
ones. All numbers come from the full run of 11 November 2026, saved in `results/`.

**Contents**

1. The project in one minute
2. The ideas you need, in plain words
3. What we actually did, step by step
4. The rules: how a model passes or fails
5. What we found
6. What we can and cannot claim
7. How to read the demo page
8. Confusions you may have (with answers)
9. Questions a judge may ask (with answers)
10. Where everything is
11. What is not finished

---

## 1. The project in one minute

Forecasting AIs do not give one number. They give a range and a promise: "I'm 80% sure the true
value will be inside." People shrink these AIs so they run on cheap devices. Studies of
shrinking check whether the main forecast stays accurate. They do not check whether the "80%
sure" stays true.

We checked. We took Amazon's Chronos-Bolt forecasting AI in four sizes, shrank each one step by
step, and tested the promise on a full year of hourly temperature in 10 Indian cities.

**What we found:**

- Down to **4 bits**, every size kept its promise.
- At **3 bits**, every size "broke": it still covered about 80%, but only by giving ranges
  clearly worse than the original model's.
- At **2 bits**, every size plainly failed (coverage 30% to 60% instead of 80%).
- A statistical correction (conformal recalibration) brought the coverage back to about 80% at
  every level, but it could not make the ranges useful again. So it did **not** move the
  breaking point.

**One sentence for a judge:** "A compressed forecasting model can keep its coverage number while
its ranges become useless, so coverage alone is not enough to call it honest."

---

## 2. The ideas you need, in plain words

### Forecast range (prediction interval)
A lower value and an upper value. "Tomorrow at 3 pm it will be between 31° and 34°."

### The 80% promise
The model claims the true value will land inside its range 80% of the time. That also means it
is **supposed to miss 20% of the time**: about 5 hours out of every 24. A miss is not a bug.

### Coverage
How often the truth really landed inside, measured over many forecasts. An honest 80% range has
coverage close to 80%. Much lower means overconfident. Much higher means the ranges are wider
than they need to be.

### Honest (calibrated)
The promise matches reality. This is all "honest" means in our title.

### Quantiles
Chronos-Bolt outputs nine values for each hour, called the 0.1, 0.2, ... 0.9 quantiles. The 0.1
quantile is a value the model thinks the truth will be below only 10% of the time. The 0.9
quantile is a value the truth should be below 90% of the time. Between the 0.1 and 0.9 quantiles
lies 80%: that is the **80% range**. Between 0.2 and 0.8 lies the **60% range**. The 0.5
quantile is the middle forecast (the median). Chronos-Bolt has no quantiles outside 0.1 to 0.9,
which is why we cannot test a 90% or 95% range.

### Weights
A forecasting AI is a long list of numbers it learned in training, called weights. Think of
millions of tiny dials. Data goes in, gets multiplied and added through the dials, and a
forecast comes out.

| Model | Weights | Size at full precision |
| --- | --- | --- |
| Chronos-Bolt tiny | 8.6 million | 34.6 MB |
| mini | 21.2 million | 84.9 MB |
| small | 47.7 million | 190.8 MB |
| base | 205.2 million | 820.9 MB |

### Bits
A bit is one switch, off or on. With n switches you can make 2ⁿ different patterns: 2 bits give
4, 8 bits give 256, 32 bits give about 4.3 billion. A 32-bit weight can take billions of
different values, so it is stored very precisely.

### Quantization (our compression)
Storing each weight with fewer bits, so it can take fewer different values. Every weight is
rounded to the nearest allowed value. Our method, for one row of weights:

1. Find the largest weight in the row (ignoring minus signs), say 0.83.
2. Spread the allowed values evenly from −0.83 to +0.83. At 8 bits there are 255 of them, at
   4 bits 15, at 3 bits 7, at 2 bits only 3 (−0.83, 0, +0.83).
3. Round each weight to the nearest allowed value.

Example: the weight 0.40 becomes 0.40 at 8 bits, 0.36 at 4 bits, and 0 at 2 bits.

The proper name is **post-training, uniform, symmetric, per-channel, round-to-nearest weight
quantization, simulated**. It is lossy: the rounding cannot be undone. It is not Huffman coding.
Huffman is lossless, so it would not change the model's behaviour at all.

16-bit in our tables means the weights were converted to half precision (float16) and back.

### Simulated
We round the weights and then store them as ordinary numbers. The rounding errors are real, so
accuracy and honesty are measured faithfully. But the file is not actually smaller and the model
is not faster. To check the simulation, we also ran a real 8-bit model (section 5).

### Pruning (our second compression)
Setting the smallest weights to exactly zero: the smallest 30%, 50% or 70% in each layer.

### Conformal recalibration (we also call it "the fix")
A correction applied to the model's **answer**, not to the model. It looks at how far the truth
fell outside the model's recent ranges and moves both edges of today's range by that amount.

For each hour ahead separately:

1. Take the model's 600 most recent forecasts (60 days × 10 cities), all before the day being
   forecast.
2. For each, compute a score: the larger of (lower edge − truth) and (truth − upper edge).
   Positive = truth was outside by that much. Negative = truth was inside with room to spare.
3. Sort the 600 scores and take the 481st smallest. Call it q. (481 = 80% of 601, rounded up.)
4. New range = (lower edge − q) to (upper edge + q).

If q is positive the range gets wider. If q is negative it gets narrower.

**It does not make the forecast more accurate.** It cannot move the middle forecast. It can
restore the "80% inside" promise, but only by changing the width.

We tested four versions: **rolling** (refitted for every test day on the 60 days before it; our
main one), **static** (fitted once and never updated), **weighted** (recent days count more) and
**adaptive** (adjusts itself after every day depending on whether it just missed).

The scores are divided by each city's typical daily swing before pooling, so that a city with
big temperature swings and a city with small ones can share one set of scores.

### Width
Upper edge minus lower edge. Narrow is useful, but only if coverage holds.

### Interval score
One number that combines width and misses: it adds the width of the range to a penalty for
every miss (10 times the distance by which the truth fell outside, for an 80% range). Lower is
better. A model cannot get a good score by making ranges huge, nor by making them tiny. From
Gneiting & Raftery (2007).

### Confidence interval (CI) and bootstrap
A measured coverage like 82.3% is an estimate. The 95% CI (for example 81.7% to 83.0%) says how
sure we are. We get it with a **bootstrap**: we re-pick 10 cities at random, with repeats
allowed, from our 10 cities, recompute the coverage, and do this 2,000 times. The middle 95% of
those results is the CI. We resample whole cities because hours inside one city are related to
each other, so treating them as independent would make us look more certain than we are.

### Window
One forecasting task: show the model 512 hours of history (about 21 days), ask it to forecast
the next 24 hours. Our windows do not overlap: one per city per day, starting at midnight.

### Calibration windows and test windows
Calibration windows are used to fit the recalibration. Test windows are used to judge. A test
window's truth is never used to fit anything applied to that same window. That is what "no
leakage" means.

### Pre-registration
Writing down the plan and the pass/fail rules, in public and with a date, **before** running the
real experiment. It stops anyone, including us, from adjusting the rules after seeing the
results. Ours is `PREREGISTRATION.md`.

---

## 3. What we actually did, step by step

1. **Got real data.** Hourly temperature for 10 Indian cities, 1 June 2025 to 31 October 2026,
   from Open-Meteo (reanalysis: weather-model reconstructions, not direct station readings).
   12,432 hours per city, no gaps. Frozen in one CSV file with a published checksum.
2. **Cut it into windows.** The last 365 days per city are test windows (1 Nov 2025 to
   31 Oct 2026): 3,650 forecasts, 87,600 hourly values. The 60 days before that are the first
   calibration windows.
3. **Loaded the model.** Chronos-Bolt in four sizes. We did not train it. Amazon trained it. It
   was released in November 2024 and our data starts in June 2025, so it cannot have seen our
   data during training.
4. **Shrank it.** 16, 8, 6, 4, 3 and 2 bits. Each bit-width below 16 twice: all layers, and with
   the input and output layers kept at full precision. Plus pruning at 30%, 50%, 70%.
5. **Forecast every test window** with every version of the model.
6. **Measured.** Coverage with its CI, width, interval score, error of the middle forecast,
   how often quantiles came out in the wrong order, and the difference from the full model on
   the same windows.
7. **Applied the four recalibrations** to every version and measured again.
8. **Applied the rules** (next section) to get verdicts and breaking points.
9. **Checked the simulation** against a real 8-bit model.

Everything for one model size is one command. Run times on the laptop: tiny 80 seconds, mini
2.5 minutes, small 5 minutes, base 19 minutes.

---

## 4. The rules: how a model passes or fails

Tolerance: 5 percentage points. For the 80% range the line is therefore 75%.

**Verdict against the promise**

| Verdict | Meaning |
| --- | --- |
| FAIL | The whole CI is below 75%. We are confident coverage is too low. |
| pass | The whole CI is at or above 75%. We are confident it is not too low. |
| inconclusive | The CI straddles 75%. The data cannot tell. |
| over-cautious | The whole CI is above 85%. Honest, but wider than needed. |
| covers, but too wide | It does not fail, but its interval score is more than 10% worse than the uncompressed model's. It reaches coverage only with clearly worse ranges. |

**Verdict against the full model.** Coverage of the compressed model minus coverage of the full
model, on the same windows: "worse than full" if we are confident the loss is more than
5 points, "as good as full" if we are confident it is not, otherwise inconclusive.

**Breaking point.** Going down from 16 bits: the first bit-width that is FAIL or "covers, but
too wide". "Safe down to" is the lowest bit-width for which it and everything above it passes.

**Worked example (tiny, 3-bit, no fix).** Coverage 83.2%, CI 82.2% to 84.3%. The whole CI is
above 75%, so it does not fail. But its interval score is 1.31 times the full model's, which is
more than 1.10. Verdict: covers, but too wide. So 3-bit is the breaking point.

---

## 5. What we found

### Main result: 80% range, all layers quantized, no recalibration

| Precision | Tiny | Mini | Small | Base |
| --- | --- | --- | --- | --- |
| Full model | 82.3% pass | 81.0% pass | 81.3% pass | 81.2% pass |
| 8-bit | 82.2% pass | 81.5% pass | 81.5% pass | 81.5% pass |
| 6-bit | 82.9% pass | 81.0% pass | 81.0% pass | 82.0% pass |
| 4-bit | 77.5% pass | 81.1% pass | 85.5% pass | 88.0% over-cautious |
| 3-bit | 83.2% too wide | 76.9% too wide | 88.6% too wide | 90.1% too wide |
| 2-bit | 59.8% FAIL | 54.1% FAIL | 56.0% FAIL | 30.5% FAIL |

**Every size: safe down to 4-bit, breaks at 3-bit.**

### With rolling conformal recalibration

Coverage comes back to between 78.5% and 80.1% at every bit-width, for every size. But at 3-bit
and 2-bit the interval score is still far worse than the full model's (1.18 to 2.38 times at
3-bit, 3.35 to 35 times at 2-bit), so the verdict stays "covers, but too wide". The breaking
point stays at 3-bit.

### Accuracy of the middle forecast, compared with the full model

| Precision | Tiny | Mini | Small | Base |
| --- | --- | --- | --- | --- |
| 8-bit, 6-bit | 1.00× | 1.00× | 1.00× | 1.00× |
| 4-bit | 1.05× | 1.06× | 1.04× | 1.07× |
| 3-bit | 1.25× | 1.40× | 1.30× | 1.49× |
| 2-bit | 4.15× | 4.15× | 3.80× | 4.69× |

(1.25× means the error is 25% larger.)

### Our four hypotheses

| | Hypothesis | Result |
| --- | --- | --- |
| H1 | Compression breaks the range at some bit-width | **Supported**, all four sizes (3-bit) |
| H2 | Coverage is damaged before accuracy | **Not supported.** Accuracy passes our 10% limit at 3-bit, the same step where the range breaks |
| H3 | Recalibration moves the breaking point lower | **Not supported.** It restores coverage but not usefulness |
| H4 | Keeping input/output layers at full precision moves the breaking point lower | **Mixed.** True for tiny only |

Two hypotheses were not supported. That is a normal and respectable outcome. We wrote them down
in advance and we report them as they came out.

### Other findings

- **The full model is honest on a full year:** 81% to 82% coverage for a promised 80%.
- **It is not equally honest at every hour ahead.** For tiny, coverage is 92% for the first
  hour, 83% at hour 12 and 79% at hour 24. Too cautious early, slightly overconfident late.
- **At 4-bit, coverage drifts in different directions:** down 5 points for tiny, up 7 for base.
  Compression does not simply make a model overconfident.
- **Crossed quantiles** (a lower quantile coming out above a higher one) go from 0% for the full
  model to 12%–61% at 4-bit and 94%–100% at 2-bit. We count them, then sort them before building
  ranges.
- **Pruning:** 30% passes on every size. 50% and 70% break.
- **Real 8-bit check:** a real int8 model's coverage was within 1 percentage point of our
  simulated 8-bit on the same 3,650 windows, for all four sizes. Real int8 was also 1.6 to
  4 times faster than full precision on the laptop's CPU.
- **Sizes:** at 4 bits the weights would need one eighth of the memory (tiny: 34.6 MB to 4.3 MB;
  base: 821 MB to 103 MB).

---

## 6. What we can and cannot claim

**We can say**

- "Measured coverage on 3,650 held-out forecast days."
- "For Chronos-Bolt on hourly temperature, with simple round-to-nearest quantization, the 80%
  range held down to 4 bits and broke at 3 bits under our pre-registered rule."
- "We found no published study of how quantization or pruning changes the calibration of
  time-series foundation models" (searched to 9 Nov 2026).

**We must not say**

- "Guaranteed coverage." Conformal guarantees need data that can be shuffled without changing
  anything (exchangeable). Time series are not like that.
- "Compressed forecasting models are safe at 4 bits" in general. We tested one model family,
  one kind of data, one compression method.
- "No one has studied this." We can only say we found no study.
- "The simulation is confirmed at 4, 3 or 2 bits." We could only check 8 bits against a real
  model.
- Anything from the synthetic self-test or the 5-day quick check as a result.

---

## 7. How to read the demo page

Local: `python -m honesty_audit app`. Online: https://forecast-honesty-audit.streamlit.app/

You pick a city, a day and a bit-width. The model is shown the 21 days before that day and
forecasts its 24 hours, right then. Nothing is pre-computed.

- **Grey line:** the last 3 days of temperature the model was shown.
- **Blue band:** the full model's 80% range. **Orange:** the compressed model's.
  **Green:** the compressed model's range after conformal recalibration.
- **Dots:** the true values, hidden until you press Reveal. Red = outside the last band shown.
- **Cards:** how many of the 24 true values fell inside each range, a strip of 24 squares (one
  per hour), the average width, misses on the cold and hot side, and the middle forecast's error.
- **"What the full audit found":** the result over all 3,650 days for that bit-width. This is
  the result that counts.
- **Hour-by-hour table:** every hour's true value and each range.

**The most important reading rule:** a tick (✓) only means "the true value fell inside this
range". It does not mean the range is accurate or useful. A range of 27° to 312° gets a tick
for almost any temperature. Always look at the width too.

**The second rule:** one day is 24 values. Even a perfectly honest model can score 14 or 24 out
of 24 on a single day. The demo shows how the check works. The verdict comes from 3,650 days.

The online demo uses the tiny model so that it fits a free server. Free Streamlit apps sleep
when unused, so open the link a few minutes early.

---

## 8. Confusions you may have (with answers)

**Is our data being compressed?**
No. The temperature data is never compressed. Every model sees the same temperatures. Only the
model's weights are compressed.

**The full model missed at 16:00, but the 2-bit model got a tick. Did 2 bits fix it?**
No. The full model's range was about 3.5° wide and the truth was 0.2° outside it. The 2-bit
model's range at that hour was 285° wide. It "caught" the truth by saying almost nothing.

**Why is the full model allowed to miss at all?**
Because it promises 80%, not 100%. About 5 misses in 24 hours is what honesty looks like.

**The recalibrated range is even wider than the 2-bit one, and the page says it is inside. How
is that a fix?**
It repairs the promise, not the precision. It moved both edges outward by q until 80% of recent
forecasts would have been inside. So the tick is real, but the range is wider. That is exactly
our finding: recalibration restores coverage, not usefulness.

**Why does recalibration sometimes make a range narrower?**
For that hour ahead, the model's recent ranges were wider than needed, so q was negative.

**Why are some 2-bit ranges 300° wide?**
At 2 bits every weight is one of three values. That wrecks the part of the model that produces
the upper edge of the range, so it outputs nonsense.

**Why did 3-bit get 83% coverage and still "break"?**
Because its ranges became wider and its misses bigger: its interval score was 1.31 times the
full model's. Our rule says more than 1.10 is too much. Coverage alone would have called it
fine. This is the reason the usefulness rule exists.

**Why did coverage go *up* at 4 bits for the bigger models?**
Their rounding errors happened to push the range edges outward. Rounding does not push in one
fixed direction. That is why you have to measure instead of assuming.

**The quick check showed 73% for the full model. The full run shows 82%. Which is right?**
The full run. The quick check used 5 days in late October; the full run used 365. Five days was
a fluke, and the pre-registration said in advance that it was too small to conclude anything.

**The table shows a range starting at 26.7° and a true value of 26.7° marked ✗. Why?**
Rounding on the screen. The true value was something like 26.69° and the edge 26.74°.

**What does "scaled" mean in the tables?**
Divided by each city's typical change from one day to the next at the same hour. It lets us
average cities with big and small temperature swings. On that scale, 1.0 is roughly the error
of guessing "same as yesterday at this hour". Our full models score about 0.85.

**Is the compressed model really smaller on the laptop?**
No, it is simulated (section 2). Only the real int8 check used a truly smaller model.

**Did we train an AI?**
No. We audited one that Amazon trained. Our contribution is the audit: the method, the rules,
the measurements and the tool.

**Why temperature? Why these cities?**
Free, public, real, hourly, recent enough that the model cannot have seen it, and relevant to
India. Ten cities because the bootstrap needs several separate series.

**Why 24 hours ahead and 512 hours of history?**
24 hours is one full daily cycle and a common forecasting task. 512 hours is about three weeks,
enough for the model to see the daily pattern many times.

**Why 80% and 60%, not 95%?**
Chronos-Bolt only outputs quantiles from 0.1 to 0.9, so 80% is the widest range it offers.

**What is the difference between rolling and static recalibration?**
Static is fitted once on September–October 2025 and used unchanged for a year. Rolling is
refitted every day on the 60 days before. Both passed for the full model; rolling stays closer
to 80%.

**Why does keeping the input/output layers help for tiny but not the others?**
We do not know yet. For small and base, those rows at 3-bit and 2-bit look odd (a much worse
score but a better middle forecast), which suggests a few extreme forecasts. This is unexplained
and we say so.

**The rules were published only 39 minutes before the results. Is that a problem?**
The whole run takes about half an hour, so the gap is naturally short. GitHub's own record shows
the order. Anyone can rerun everything from the public repo and get the same numbers. We will
also test on days that did not exist when the rules were written (section 11).

**Did we look at any real results before fixing the rules?**
Yes, one small quick check (Chronos-Bolt tiny, 16-bit and 4-bit, 5 test days). It is disclosed
in section 11 of the pre-registration, including that it influenced the rules.

---

## 9. Questions a judge may ask (with answers)

**What is new here?**
Compression studies measure accuracy. Calibration studies look at full-size models. We measured
what compression does to calibration, with rules fixed in advance. We found no published study
doing this for time-series foundation models.

**What is your main finding?**
The range holds to 4 bits and breaks at 3, for all four model sizes. And at 3 bits, three of the
four sizes still show 80% or more coverage, so someone checking coverage alone would miss it.

**Does recalibration solve the problem?**
It restores the coverage number at every level, even 2 bits. It does not restore usefulness.
So it does not let you compress further under our rule.

**Why should anyone care?**
Decisions use the edges of the range: how much backup power, how much stock. A range that is
wrong, or too wide to act on, costs money even if the middle forecast looks fine.

**How do you know the result is not chance?**
3,650 forecast days per model, 95% confidence intervals, and the same pattern in four separate
model sizes.

**How do you know your simulated compression is realistic?**
At 8 bits we compared it with a real int8 model: coverage within 1 percentage point on all four
sizes. Lower bit-widths remain simulated only; PyTorch has no real runtime for them.

**Chronos-tiny already runs in a millisecond. Why compress it?**
The base model is 821 MB; at 4 bits it would need about 103 MB. Small devices and
microcontrollers have very little memory. And the question applies to any forecasting model.

**Would a better compression method survive lower?**
Probably. We used the simplest standard method. Our claim is limited to that method.

**What are the weaknesses?**
One model family, one kind of data, ten related cities, simulated compression below 8 bits, and
one unexplained result (the input/output layer rows).

**Is the demo pre-recorded?**
No. You choose the city, day and bit-width and it forecasts live. The link works on your own
device.

**What did you do, and what did the AI do?**
Be exact and honest here. The audit code was written with the help of Claude (an AI assistant),
which also reviewed the method and helped draft documents. We chose the topic and the question,
made the decisions on the rules (tolerance, usefulness threshold, verdicts), ran the
experiments, and are responsible for the claims. **Confirm with your coordinator how AI
assistance must be declared, and make sure you can explain each file before the event.**

---

## 10. Where everything is

| What | Where |
| --- | --- |
| Public repo | https://github.com/RinMac/forecast-honesty-audit |
| Live demo online | https://forecast-honesty-audit.streamlit.app/ |
| The plan and rules | `PREREGISTRATION.md` |
| Frozen data | `data/open_meteo_10cities_2025-06-01_2026-10-31.csv` |
| Results per model | `results/full_chronos_bolt_<size>/` (`results.json`, `honesty_report.html`, `int8_check.json`) |
| File-by-file code walkthrough | `docs/WALKTHROUGH.md` (written before the demo and int8 files were added) |

| Code file | What it does |
| --- | --- |
| `honesty_audit/data.py` | Loads data, cuts it into windows |
| `honesty_audit/models.py` | Wraps Chronos-Bolt so it returns quantiles |
| `honesty_audit/compress.py` | Quantization and pruning |
| `honesty_audit/conformal.py` | The recalibration methods |
| `honesty_audit/metrics.py` | Coverage, interval score, bootstrap |
| `honesty_audit/audit.py` | Runs the experiment, applies the rules, the real int8 check |
| `honesty_audit/report.py` | Builds the honesty report page |
| `honesty_audit/demo.py` | The live demo for one chosen day |
| `honesty_audit/app.py`, `streamlit_app.py` | The local and the online demo pages |
| `honesty_audit/cli.py` | The commands below |

```
python -m pytest -q                         run the 16 tests
python -m honesty_audit app                 local demo page
python -m honesty_audit demo --fix          demo in the terminal
python -m honesty_audit run --model chronos-bolt-tiny --data csv:data/open_meteo_10cities_2025-06-01_2026-10-31.csv:temperature_2m:timestamp:city --n-calib 60 --n-test 365 --prune 0.3,0.5,0.7 --real-int8 --out results/full_chronos_bolt_tiny
```

---

## 11. What is not finished (as of 12 November 2026)

- **Seasonal-naive baseline** (promised as a secondary result): not written.
- **Coverage per month** (promised as a secondary result): the data is saved, the table is not made.
- **The input/output-layer rows** at 3-bit and 2-bit for small and base: unexplained.
- **The check on new days:** download the days after 11 November shortly before the event and
  run the same audit on them. Small sample, supporting evidence only.
- **Poster:** the "expected result" picture no longer matches the real result (it suggests the
  fix moves the breaking point; it did not).
- **Tests:** there are no automated tests for the demo pages or the int8 check.
- **Housekeeping:** add Soham to the repo, choose a licence, confirm the AI-use rule.
