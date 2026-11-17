# Tables for the paper (generated from results.json, do not edit by hand)

Model run settings: 3650 test windows per model, 10 cities, 60 calibration days, tolerance 0.05, usefulness limit 10%. Data: CSV file open_meteo_10cities_2025-06-01_2026-10-31.csv (SHA-256 0685492f047c9e9e888fcaef83fe5c86579ff4be74b6da8b1811b19bd90ec108).

## Table 1. Chronos-Bolt-small, 80% interval, all linear layers quantized, own intervals

| Precision | Coverage (95% CI) | Verdict | Change vs full model (pp, 95% CI) | Scaled interval score | Scaled median error |
| --- | --- | --- | --- | --- | --- |
| fp32 (full) | 81.3% (80.4% to 82.2%) | pass | reference | 4.10 | 0.849 (1.00x) |
| fp16 | 81.3% (80.4% to 82.2%) | pass | -0.0 (-0.0 to +0.0) | 4.10 | 0.849 (1.00x) |
| 8-bit | 81.5% (80.6% to 82.3%) | pass | +0.2 (+0.1 to +0.2) | 4.10 | 0.849 (1.00x) |
| 6-bit | 81.0% (80.1% to 81.9%) | pass | -0.3 (-0.4 to -0.2) | 4.11 | 0.850 (1.00x) |
| 4-bit | 85.5% (84.9% to 86.2%) | pass | +4.2 (+3.4 to +4.9) | 4.41 | 0.884 (1.04x) |
| 3-bit | 88.6% (88.1% to 89.2%) | covers, but too wide | +7.3 (+6.1 to +8.6) | 5.30 | 1.103 (1.30x) |
| 2-bit | 56.0% (54.9% to 57.2%) | FAIL | -25.3 (-26.8 to -23.9) | 19.40 | 3.224 (3.80x) |

## Table 2. Lowest precision that passes, 80% interval, all layers quantized

Each cell: safe down to (breaks at).

| Model | Own intervals | Rolling split CQR | Static split CQR | Weighted CQR | Adaptive (ACI) |
| --- | --- | --- | --- | --- | --- |
| Chronos-Bolt-tiny | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) |
| Chronos-Bolt-mini | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) |
| Chronos-Bolt-small | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) |
| Chronos-Bolt-base | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) | 4-bit (breaks at 3-bit) |

## Table 3. Coverage breaking point against accuracy breaking point (hypothesis H2)

| Model | Coverage breaks at (80%, own intervals) | Accuracy breaks at (median error > 10% worse) | Median error at 4 / 3 / 2 bits (x full) |
| --- | --- | --- | --- |
| Chronos-Bolt-tiny | 3-bit | 3-bit | 1.05 / 1.25 / 4.15 |
| Chronos-Bolt-mini | 3-bit | 3-bit | 1.06 / 1.40 / 4.15 |
| Chronos-Bolt-small | 3-bit | 3-bit | 1.04 / 1.30 / 3.80 |
| Chronos-Bolt-base | 3-bit | 3-bit | 1.07 / 1.49 / 4.69 |

## Coverage of the 80% interval by model size, own intervals

| Precision | tiny | mini | small | base |
| --- | --- | --- | --- | --- |
| fp32 (full) | 82.3% pass | 81.0% pass | 81.3% pass | 81.2% pass |
| fp16 | 82.3% pass | 81.0% pass | 81.3% pass | 81.2% pass |
| 8-bit | 82.2% pass | 81.5% pass | 81.5% pass | 81.5% pass |
| 6-bit | 82.9% pass | 81.0% pass | 81.0% pass | 82.0% pass |
| 4-bit | 77.5% pass | 81.1% pass | 85.5% pass | 88.0% over-cautious |
| 3-bit | 83.2% covers, but too wide | 76.9% covers, but too wide | 88.6% covers, but too wide | 90.1% covers, but too wide |
| 2-bit | 59.8% FAIL | 54.1% FAIL | 56.0% FAIL | 30.5% FAIL |

## Coverage of the 80% interval by model size, after rolling split CQR

| Precision | tiny | mini | small | base |
| --- | --- | --- | --- | --- |
| fp32 (full) | 80.0% pass | 80.0% pass | 80.1% pass | 79.8% pass |
| fp16 | 80.0% pass | 80.0% pass | 80.1% pass | 79.8% pass |
| 8-bit | 80.0% pass | 80.0% pass | 80.0% pass | 79.9% pass |
| 6-bit | 79.9% pass | 80.0% pass | 80.0% pass | 79.8% pass |
| 4-bit | 80.0% pass | 80.0% pass | 79.6% pass | 79.8% pass |
| 3-bit | 79.6% covers, but too wide | 79.6% covers, but too wide | 79.6% covers, but too wide | 79.7% covers, but too wide |
| 2-bit | 78.5% covers, but too wide | 78.6% covers, but too wide | 79.0% covers, but too wide | 78.6% covers, but too wide |

## Coverage of the 60% interval by model size, own intervals

| Precision | tiny | mini | small | base |
| --- | --- | --- | --- | --- |
| fp32 (full) | 62.9% pass | 60.3% pass | 61.1% pass | 61.2% pass |
| fp16 | 62.9% pass | 60.3% pass | 61.1% pass | 61.2% pass |
| 8-bit | 62.8% pass | 61.0% pass | 61.0% pass | 61.3% pass |
| 6-bit | 62.9% pass | 60.3% pass | 61.1% pass | 61.7% pass |
| 4-bit | 55.9% inconclusive | 59.4% pass | 68.5% over-cautious | 66.7% over-cautious |
| 3-bit | 63.4% covers, but too wide | 53.1% FAIL | 64.0% covers, but too wide | 68.3% covers, but too wide |
| 2-bit | 32.4% FAIL | 29.0% FAIL | 30.5% FAIL | 17.2% FAIL |

## Additional findings, per model

- **tiny** (8,642,560 weights, 34.6 MB at fp32, 4.3 MB at 4-bit): full-model 80% coverage 82.3%, 60% coverage 62.9%; by hour ahead 91.8% (1 h), 82.8% (12 h), 78.8% (24 h); misses below / above the 80% range 10.3% / 7.4%. I/O kept: safe down to 3-bit, breaks at 2-bit. Crossed quantiles 59.9% / 76.8% / 100.0% at 4 / 3 / 2 bits (0% at full precision). Pruning: 30% pass, 50% covers, but too wide, 70% covers, but too wide. Real int8 on CPU: 0.86 ms vs 0.54 ms per window (1.59x faster). Real int8 minus simulated 8-bit coverage: -0.2 pp (-0.4 to -0.1).
- **mini** (21,221,376 weights, 84.9 MB at fp32, 10.6 MB at 4-bit): full-model 80% coverage 81.0%, 60% coverage 60.3%; by hour ahead 90.4% (1 h), 81.4% (12 h), 78.3% (24 h); misses below / above the 80% range 10.5% / 8.5%. I/O kept: safe down to 4-bit, breaks at 3-bit. Crossed quantiles 60.9% / 82.7% / 100.0% at 4 / 3 / 2 bits (0% at full precision). Pruning: 30% pass, 50% FAIL, 70% FAIL. Real int8 on CPU: 1.96 ms vs 0.85 ms per window (2.3x faster). Real int8 minus simulated 8-bit coverage: -0.7 pp (-0.8 to -0.5).
- **small** (47,693,824 weights, 190.8 MB at fp32, 23.8 MB at 4-bit): full-model 80% coverage 81.3%, 60% coverage 61.1%; by hour ahead 89.8% (1 h), 82.2% (12 h), 79.1% (24 h); misses below / above the 80% range 9.9% / 8.8%. I/O kept: safe down to 4-bit, breaks at 3-bit. Crossed quantiles 30.2% / 58.8% / 94.5% at 4 / 3 / 2 bits (0% at full precision). Pruning: 30% pass, 50% covers, but too wide, 70% FAIL. Real int8 on CPU: 3.97 ms vs 1.48 ms per window (2.69x faster). Real int8 minus simulated 8-bit coverage: +0.6 pp (+0.4 to +0.7).
- **base** (205,234,176 weights, 820.9 MB at fp32, 102.6 MB at 4-bit): full-model 80% coverage 81.2%, 60% coverage 61.2%; by hour ahead 87.7% (1 h), 82.5% (12 h), 79.1% (24 h); misses below / above the 80% range 9.9% / 8.9%. I/O kept: safe down to 4-bit, breaks at 3-bit. Crossed quantiles 11.8% / 35.4% / 99.8% at 4 / 3 / 2 bits (0% at full precision). Pruning: 30% pass, 50% covers, but too wide, 70% FAIL. Real int8 on CPU: 17.02 ms vs 4.29 ms per window (3.97x faster). Real int8 minus simulated 8-bit coverage: +0.1 pp (-0.1 to +0.3).
