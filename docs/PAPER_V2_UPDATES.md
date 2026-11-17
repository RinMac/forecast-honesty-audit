# Paper v2: updates, section by section

For Rishi and Soham. Written 17 November 2026 against `Research_Paper_v2_for_discussion.pdf`
(dated 16 Nov) and `Paper v2 To-Do.md`.

**Read this first.** Version 2 of the paper still describes the plan from before
11 November (20 calibration and 20 test windows per city, one static conformal fit, an
undecided threshold X). The experiment you actually pre-registered and ran is different: 60
calibration days, 365 test days, rolling recalibration, X = 10%. So several sections need more
than filling blanks: they need correcting. Every number below comes from `paper/tables.md`,
which is generated from `results/*/results.json` by `python -m honesty_audit paper`. Check each
number against your own report pages before you submit, and rewrite the text in your own words.

Status of the to-do list:

| To-do item | Status |
| --- | --- |
| Run the quick check | Done 11 Nov (Chronos-Bolt-tiny, 5 + 5 days) |
| Decide X | Done 11 Nov: 10%, in the pre-registration |
| Decide the accuracy rule | Done 11 Nov: hypothesis H2 in the pre-registration (median error more than 10% worse). Now computed by the code |
| Confirm the hypothesis and its date | Done: H1 to H4, fixed 11 Nov 2026 |
| Pre-register on GitHub | Done: commit `72f1328`, pushed 11 Nov 2026, 17:21 IST |
| Full audit | Done 11 Nov for tiny, mini, small and base. Re-run on 17 Nov to check it reproduces (see `results/REPRODUCIBILITY.md`) |
| Author block | **Open: you must fill it** (Rishi's surname, both email addresses) |
| Usefulness rule in code | Done 11 Nov |
| Accuracy breaking point in code | Done 17 Nov (`accuracy_breaking_points`) |
| Pin the weather model | Done 17 Nov. The data turned out to be ECMWF IFS, not ERA5 (below) |
| Record versions | Done 17 Nov (`meta.versions` in every new results.json) |
| Figures from results | Done 17 Nov (`python -m honesty_audit paper`) |
| Real int8 check (was "worth adding") | Done 11 Nov |
| Baseline, per-quantile error, second data type, TimesFM | Not done |

---

## Title and author block

- The poster and demo now use "Auditing Uncertainty in Time-Series AI Models". Decide which
  title the paper carries and keep it the same everywhere (check with your coordinator which
  title is registered).
- **Fill yourselves:** Rishi's surname and both email addresses. I have not guessed them.

## Abstract

Replace the `[TO FILL]` sentence with something like:

> Across four model sizes and 3,650 held-out forecast days each, the 80% intervals stayed
> honest down to 4-bit weights and broke at 3 bits, where coverage held only through intervals
> clearly worse than the full model's; at 2 bits coverage fell to 31–60%. Conformal
> recalibration restored coverage to about 80% at every precision but not the quality of the
> intervals, so it did not move the breaking point.

Also change "a pass/fail rule fixed before testing" to "a pass/fail rule published before the
full experiment".

## Section 1, Introduction

**Hypothesis.** Your pre-registration (11 Nov) fixed four hypotheses, not one. The paragraph
in the paper is H2. Either list all four, or keep H2 and add one sentence: "This and three
further hypotheses were published on 11 November 2026, before the full experiment (Section 3.7)."
Add the accuracy rule, which makes H2 testable:

> We say accuracy breaks at the first precision whose median-forecast error is more than 10%
> larger than the full model's.

**Why this matters in practice.** Replace the `[TO FILL]` with real numbers:

> For Chronos-Bolt-base, 4-bit weights would need about 103 MB instead of 821 MB. On a laptop
> CPU, a real int8 version of the same model forecast about 4 times faster than full precision.

**Contribution 1** can stay. Contribution 2 should now say the fixes were *tested* and report
what happened ("none of them moved the breaking point").

## Section 3, Method

Delete the `[TO FILL: after the runs, check every setting…]` note and add instead:

> All settings below were published on 11 November 2026 before the full experiment
> (Section 3.7). The only change since is the project's display title.

### 3.1 Models
> We audit Chronos-Bolt in the tiny, mini, small and base sizes (8.6 M, 21.2 M, 47.7 M and
> 205.2 M weights), loaded through chronos-forecasting 2.3.2 in fp32.

Delete the TimesFM-2.5 note (it was not run).

### 3.2 Data
Replace "These values are reanalysis estimates…" and the `[TO FILL]` with:

> The values come from the ECMWF Integrated Forecasting System (IFS) at 9 km resolution, as
> served by Open-Meteo's historical weather API (Zippenfenig, 2023). They are model-based
> estimates, not thermometer readings. We downloaded them once on 11 November 2026 and froze
> them in a single file (SHA-256 `0685492f…ec108`).

How we know it is IFS: on 17 Nov we re-downloaded two cities with each named model. The frozen
file matched `ecmwf_ifs` exactly (12,432 of 12,432 hours) and did not match ERA5, ERA5-Land or
ERA5-seamless. Open-Meteo's default "best match" gave the same values as IFS for this period.
The code now asks for `ecmwf_ifs` by name. **Also fix the word "reanalysis" on the poster and in
the study guide notes.**

### 3.3 Forecast windows
Replace the 40-window paragraph:

> Each forecast uses the previous 512 hours as context and predicts the next 24 hours, starting
> at midnight. Windows do not overlap. For each city the last 365 days (1 November 2025 to
> 31 October 2026) are test windows and the 60 days before them are the first calibration
> windows. With ten cities this gives 3,650 test windows, or 87,600 hourly test values per
> interval level. No value from a test day, or later, is used to fit anything applied to that day.

### 3.4 Compression
Fill the two blanks:

> Only the weights of linear layers are rounded (70 layers in Chronos-Bolt-tiny, holding 99.9% of
> its weights). Every embedding (including the relative-position bias) and every normalisation
> layer stays in full precision.

> Pruning is applied per layer: within each linear layer, the given share of weights with the
> smallest absolute value is set to zero.

### 3.5 Recalibration
Replace the description of split CQR, and add the scaling, the rolling fit, and the static
version, because that is what was pre-registered and run:

> Before recalibration, every score is divided by its window's scale (Section 3.6), so that
> cities with large and small temperature swings can share one calibration set; intervals are
> converted back to °C afterwards.
>
> **Rolling split CQR (main method).** For each test day, the correction is fitted on the 60 days
> before it, all ten cities pooled, so n = 600 scores per forecast step. We take the k-th
> smallest score, k = ⌈(n + 1)(1 − α)⌉ = 481 for an 80% interval.
>
> **Static split CQR.** Fitted once on the 60 calibration days and never updated, as a comparison.
>
> **Weighted CQR.** As rolling split CQR, with each day weighted by 0.9 raised to its age in days.
>
> **ACI.** Run separately for each city in time order, starting from that city's 60 calibration
> scores, with γ = 0.05. When the working miscoverage level reaches zero or below, theory gives
> an infinite interval; we instead widen by the largest score seen, so intervals stay finite.

### 3.6 Measures
Change "divided by each series' mean absolute 24-hour change" to:

> divided by the mean absolute difference between values 24 hours apart within that window's
> own 512-hour context

Add the accuracy rule here too, and "misses below and above the interval are counted separately".

### 3.7 Rule fixed in advance
Fill the blanks:

> We published the following rules on 11 November 2026, before the full experiment, in a public
> repository (commit 72f1328, https://github.com/RinMac/forecast-honesty-audit).

Set X to **10%**, and correct the last sentence, because the pre-registered breaking point
counts "too wide" too:

> The breaking point is the first precision, going down from 16 bits, that fails or is too wide
> to be useful. A level is called safe only if it and every higher precision pass; a level whose
> confidence interval straddles 75% is reported as inconclusive.

Add that the full run was repeated on 17 Nov and reproduced exactly (`results/REPRODUCIBILITY.md`).
Add honestly what was seen before: "Before publishing the rules we had run one code check on
5 test days per city for one model; it is described in the pre-registration."

### 3.8 Software
> Python 3.14.7, NumPy 2.5.2, pandas 3.0.5, PyTorch 2.13.0, chronos-forecasting 2.3.2 and
> transformers 5.16.1 (see `results/REPRODUCIBILITY.md`). Eighteen automated tests check the
> coverage, interval-score, quantization and rule code against hand-computed cases, and the
> conformal methods against simulated data.

## Section 4, Results

Paste Tables 1, 2 and 3 from `paper/tables.md` and insert `paper/figure1_coverage_vs_bits.png`
and `paper/figure2_coverage_by_hour.png`. Suggested text:

**4.1 Full models.** All four full models were close to honest: 80% intervals covered 81.0% to
82.3% and 60% intervals 60.3% to 62.9%. They were cautious one hour ahead (88–92%) and close to
or slightly below 80% at 24 hours ahead (78–79%).

**4.2 Under compression.** For Chronos-Bolt-small (Table 1), coverage and median error barely
moved down to 6 bits. At 4 bits coverage rose to 85.5% while median error grew 4%. At 3 bits
coverage reached 88.6% only because intervals widened: the interval score was 29% worse than the
full model's, so the level is "too wide". At 2 bits coverage fell to 56.0% and median error was
3.8 times larger.

**4.3 Breaking points (Table 2).** Every size, with every method, was safe down to 4 bits and
broke at 3 bits. Recalibration brought coverage to between 78.5% and 80.1% at every precision,
but at 3 and 2 bits the intervals stayed far worse than the full model's (interval score 1.2 to
35 times worse), so no method moved the breaking point.

**4.4 Additional findings.** One or two sentences each, from the last part of `tables.md`:
- **60% intervals:** every size again breaks at 3 bits (mini fails outright there, 53.1%). For
  tiny, 4 bits was inconclusive (55.9%), so tiny's 60% interval is only confirmed safe to 6 bits.
- **Input/output layers kept:** moved the breaking point to 2 bits for tiny only.
- **By hour ahead (Figure 2):** at 3 bits, coverage swung between 31% and 98% from one hour ahead
  to the next, although the overall figure looked acceptable.
- **Crossed quantiles:** 0% at full precision; 12–61% at 4 bits; 94–100% at 2 bits.
- **Pruning:** 30% passed for every size; 50% and 70% broke.
- **Real int8:** roughly 1.3 to 4 times faster than fp32 on CPU (the timing varies between runs;
  see `results/REPRODUCIBILITY.md`), and its coverage was within 1 percentage point of simulated
  8-bit for all four sizes.

## Section 5, Discussion

**5.1 Interpretation.** Answer the four questions in the template, but replace question (4)
("the shift in temperature over October") because the design changed:
1. *Research question:* safe to 4 bits, broken at 3, for all four sizes; recalibration adds no
   extra compression under the rule.
2. *Hypothesis H2:* **not supported**. Accuracy also broke at 3 bits (median error 25–49% worse),
   the same step as coverage. Say this plainly.
3. *Model size:* larger models did not tolerate compression better; the breaking point was the
   same, and base fell furthest at 2 bits (30.5%).
4. *Which fix:* all four restored coverage to about 80%; none restored interval quality.

**5.2 Guideline.** For example: "For Chronos-Bolt on hourly data, do not quantize below 4 bits
with round-to-nearest rounding. Check interval width or interval score together with coverage,
and coverage per hour ahead, because both an overall coverage number and recalibration can hide
a broken model."

**5.3** Replace the `[TO FILL]` with the base-model numbers given under Section 1.

**5.4 Limitations.** Delete "Limited sample. Ten cities and 20 test windows each…" and write:
"Ten related cities: the confidence intervals resample whole cities, but the cities share
weather patterns, so the intervals may be somewhat too narrow." Change "Reanalysis data" to
"Model-based data (ECMWF IFS)". Add "The simulation was checked against a real model only at
8 bits." Add "An unexplained result: with input/output layers kept, small and base at 3 and
2 bits gave much worse interval scores but better median error."

**5.5 Conclusion.** Use the abstract sentence.

## References

Replace the Open-Meteo entry with the citation Open-Meteo asks for, and add the retrieval date:

> Zippenfenig, P. (2023). *Open-Meteo.com Weather API* [Computer software]. Zenodo.
> https://doi.org/10.5281/ZENODO.7970649 (historical weather API, ECMWF IFS data, retrieved
> 11 November 2026)

Also check Open-Meteo's licence page for any attribution line ECMWF data requires.

## AI-assistance statement

Make it exact. The current text says the authors "ran the experiments". The full runs, the
re-run, the tables and the figures were executed by the AI assistant at your request. A truthful
version:

> The software, the experiment runs, the tables and figures, and drafts of this text were
> produced with the assistance of an AI system (Claude, Anthropic). The authors chose the
> research question, made the pre-registered decisions (tolerance, usefulness limit and
> hypotheses), directed and checked the work, and take full responsibility for the content.

Confirm the wording with your coordinator.

## Before submitting

- Remove the colour key and all highlights.
- Search the PDF for `[TO FILL` and `[author` until nothing is left.
- Check every number against `paper/tables.md` and your report pages.
