# Reproducibility check, 17 November 2026

The full pre-registered audit (published 11 November 2026, files in this folder) was run a
second time on 17 November 2026 with the same command, data file and settings:

```
python -m honesty_audit run --model chronos-bolt-<size> --data csv:data/open_meteo_10cities_2025-06-01_2026-10-31.csv:temperature_2m:timestamp:city --n-calib 60 --n-test 365 --prune 0.3,0.5,0.7 --real-int8 --out <folder>
```

Every number in the second run's `results.json` was compared with the published one, leaving
out only timing (milliseconds per window, the real-int8 speed test, run time) and fields that
did not exist on 11 November.

| Model | Values compared | Values that differ | Largest difference |
| --- | --- | --- | --- |
| Chronos-Bolt-tiny | 580,169 | 0 | 0 |
| Chronos-Bolt-mini | 580,169 | 0 | 0 |
| Chronos-Bolt-small | 580,169 | 0 | 0 |
| Chronos-Bolt-base | 580,169 | 0 | 0 |

**The results reproduce exactly.** The published files are kept unchanged as the record of the
pre-registered run.

## Software used (both runs, same laptop)

| | Version |
| --- | --- |
| Python | 3.14.7 (Windows 11) |
| NumPy | 2.5.2 |
| pandas | 3.0.5 |
| PyTorch | 2.13.0 (CPU used) |
| chronos-forecasting | 2.3.2 |
| transformers | 5.16.1 |

Model weights (Hugging Face revisions): tiny `a0e552de`, mini `25126833`, small `772f3d25`,
base `5d9f166d`. New runs record all of this automatically in `meta.versions` and
`meta.model_revision`.

## Speed varies between runs

The real int8 speed-up (int8 against fp32 on the laptop CPU, 64 windows) is a timing and changes
with whatever else the computer is doing:

| Model | 11 Nov | 17 Nov |
| --- | --- | --- |
| tiny | 1.59x | 1.25x |
| mini | 2.30x | 2.32x |
| small | 2.69x | 2.85x |
| base | 3.97x | 3.97x |

Report speed as approximate ("about 1.3 to 4 times faster"), not to two decimals.

## Accuracy breaking point (hypothesis H2), computed by the new code

Median error more than 10% worse than the full model's, all layers quantized: **3 bits for all
four sizes**, the same step where the 80% interval breaks. With input/output layers kept: 2 bits
for tiny, 3 bits for the others.
