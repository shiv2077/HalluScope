# Routing report

Prices from config/pricing.json (priced_on 2026-09-14). Brackets are 95 percent bootstrap intervals (2000 resamples, seed 42) or Wilson intervals for proportions. Raw counts carry no interval. Review lift, length strata and the cascade's budget points use the budget policy: the top ceil(budget * n) rows of the test batch by batch quantile, recomputed inside every bootstrap resample. Cascade risk points and the cutoff drift subsection use fixed cutoffs from results/calibration.json.

## Summary

On 288 test rows with a consensus label, the cheap model (gpt-4o-mini) has an error rate of 48.3% [42.6, 54.0]. Under the budget policy at 20 percent, the best scorer is `p_true`, catching 36.0% [31.7, 40.8] of errors against 20.1% [15.9, 25.1] for random, a lift of 1.79 [1.58, 2.03]; the best free scorer is `mean_logprob` at lift 1.61 [1.38, 1.84]. Residual error among unreviewed rows at that budget is 38.7% [32.2, 45.2] for `p_true`. The cascade at the `p_true` budget 0.3 point has error 34.7% [29.2, 40.1] at $0.16 [0.13, 0.18] per thousand queries, against all-cheap 47.7% [41.8, 53.5] at $0.02 [0.02, 0.02] and all-strong 26.4% [21.5, 31.8] at $0.39 [0.38, 0.41]. Fixed cutoffs from the 160-row calibration split are reported separately as risk-policy points and in the cutoff drift table.

## Data and labels

- Calibration: 160 rows with a consensus label (65 INCORRECT), 2 AMBIGUOUS excluded. Scores file hash 2071130b013df522, labels hash e76c9fcaed2edddf.
- Test: 300 questions, 288 with a consensus cheap-model label, 12 AMBIGUOUS excluded. Cascade uses the 277 rows where both models have a consensus label (23 excluded).
- Judges: `gpt-4o` and `gpt-4.1`, blind to every score, consensus rule in routing/data.py. Agreement and kappa in results/phase1_notes.md.

## Validity gates

See results/gates_test.md. Template rows: 0. Split disjoint: True. Power at actual n: 0.99.

| scorer | AUROC | vs length p | flag |
|---|---|---|---|
| consistency_k3 | 0.730 [0.668, 0.786] | 0.000 |  |
| length | 0.580 [0.517, 0.645] | n/a | control |
| length_residual | 0.721 [0.664, 0.782] | 0.001 |  |
| mean_logprob | 0.735 [0.679, 0.789] | 0.000 |  |
| min_k_logprob | 0.741 [0.685, 0.796] | 0.000 |  |
| p_true | 0.799 [0.750, 0.847] | 0.000 |  |

## Review lift

Budget policy: the top ceil(budget * n) rows of the test batch. Random-baseline catch rate equals the exact flagged fraction.

![review lift](../figures/review_lift.png)

| scorer | budget | flagged | catch rate | random | lift | precision | residual error |
|---|---:|---|---|---|---|---|---|
| mean_logprob | 0.10 | 10.1% [7.1, 14.1] | 16.5% [12.9, 19.6] | 10.1% [7.1, 14.1] | 1.64 [1.28, 1.94] | 79.3% [61.6, 90.2] | 44.8% [39.0, 51.4] |
| mean_logprob | 0.20 | 20.1% [15.9, 25.1] | 32.4% [27.8, 37.0] | 20.1% [15.9, 25.1] | 1.61 [1.38, 1.84] | 77.6% [65.3, 86.4] | 40.9% [34.8, 47.4] |
| mean_logprob | 0.30 | 30.2% [25.2, 35.7] | 43.2% [38.6, 49.7] | 30.2% [25.2, 35.7] | 1.43 [1.28, 1.64] | 69.0% [58.6, 77.7] | 39.3% [32.3, 45.3] |
| min_k_logprob | 0.10 | 10.1% [7.1, 14.1] | 18.0% [14.5, 21.5] | 10.1% [7.1, 14.1] | 1.79 [1.44, 2.13] | 86.2% [69.4, 94.5] | 44.0% [37.8, 50.2] |
| min_k_logprob | 0.20 | 20.1% [15.9, 25.1] | 31.7% [27.1, 36.3] | 20.1% [15.9, 25.1] | 1.57 [1.34, 1.80] | 75.9% [63.5, 85.0] | 41.3% [35.2, 47.8] |
| min_k_logprob | 0.30 | 30.2% [25.2, 35.7] | 45.3% [39.5, 50.4] | 30.2% [25.2, 35.7] | 1.50 [1.31, 1.67] | 72.4% [62.2, 80.7] | 37.8% [31.3, 45.3] |
| length | 0.10 | 10.1% [7.1, 14.1] | 13.7% [9.4, 17.2] | 10.1% [7.1, 14.1] | 1.36 [0.94, 1.70] | 65.5% [47.3, 80.1] | 46.3% [40.5, 52.5] |
| length | 0.20 | 20.1% [15.9, 25.1] | 23.0% [18.7, 28.1] | 20.1% [15.9, 25.1] | 1.14 [0.93, 1.40] | 55.2% [42.5, 67.3] | 46.5% [40.0, 52.6] |
| length | 0.30 | 30.2% [25.2, 35.7] | 33.8% [28.1, 38.8] | 30.2% [25.2, 35.7] | 1.12 [0.93, 1.28] | 54.0% [43.6, 64.1] | 45.8% [39.3, 52.2] |
| length_residual | 0.10 | 10.1% [7.1, 14.1] | 17.3% [13.7, 20.0] | 10.1% [7.1, 14.1] | 1.71 [1.36, 1.99] | 82.8% [65.5, 92.4] | 44.4% [38.6, 51.0] |
| length_residual | 0.20 | 20.1% [15.9, 25.1] | 31.7% [27.1, 36.4] | 20.1% [15.9, 25.1] | 1.57 [1.35, 1.81] | 75.9% [63.5, 85.0] | 41.3% [35.2, 47.8] |
| length_residual | 0.30 | 30.2% [25.2, 35.7] | 44.6% [39.4, 50.7] | 30.2% [25.2, 35.7] | 1.48 [1.31, 1.68] | 71.3% [61.0, 79.7] | 38.3% [31.3, 45.3] |
| consistency_k3 | 0.10 | 10.1% [7.1, 14.1] | 18.0% [15.3, 21.1] | 10.1% [7.1, 14.1] | 1.79 [1.52, 2.09] | 86.2% [69.4, 94.5] | 44.0% [38.2, 50.2] |
| consistency_k3 | 0.20 | 20.1% [15.9, 25.1] | 34.5% [29.7, 38.9] | 20.1% [15.9, 25.1] | 1.71 [1.47, 1.93] | 82.8% [71.1, 90.4] | 39.6% [33.5, 46.5] |
| consistency_k3 | 0.30 | 30.2% [25.2, 35.7] | 50.4% [43.7, 55.4] | 30.2% [25.2, 35.7] | 1.67 [1.45, 1.83] | 80.5% [70.9, 87.4] | 34.3% [28.4, 42.3] |
| p_true | 0.10 | 10.1% [7.1, 14.1] | 17.3% [14.5, 20.7] | 10.1% [7.1, 14.1] | 1.71 [1.44, 2.05] | 82.8% [65.5, 92.4] | 44.4% [38.2, 50.6] |
| p_true | 0.20 | 20.1% [15.9, 25.1] | 36.0% [31.7, 40.8] | 20.1% [15.9, 25.1] | 1.79 [1.58, 2.03] | 86.2% [75.1, 92.8] | 38.7% [32.2, 45.2] |
| p_true | 0.30 | 30.2% [25.2, 35.7] | 51.1% [46.0, 57.1] | 30.2% [25.2, 35.7] | 1.69 [1.52, 1.89] | 81.6% [72.2, 88.4] | 33.8% [26.9, 40.8] |

`*` marks a scorer not distinguishable from length. `length` is the control. Precision is the fraction of flagged rows that are INCORRECT, which is what a reviewer sees. Residual error is the error rate among unflagged rows, assuming review fixes what it finds.


### Residual error against budget

![residual error](../figures/residual_error.png)

Base error rate with no review: 48.3% [42.6, 54.0]. Budget policy from 0.05 to 0.50.

| scorer | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean_logprob | 46.9% [41.0, 52.7] | 44.8% [39.0, 51.4] | 42.2% [36.5, 49.2] | 40.9% [34.8, 47.4] | 39.4% [33.3, 46.3] | 39.3% [32.3, 45.3] | 36.4% [29.9, 44.4] | 35.5% [27.9, 42.4] | 32.3% [25.3, 41.1] | 29.2% [22.2, 38.9] |
| min_k_logprob | 45.4% [39.6, 51.6] | 44.0% [37.8, 50.2] | 42.2% [36.5, 49.2] | 41.3% [35.2, 47.8] | 39.8% [33.3, 46.8] | 37.8% [31.3, 45.3] | 37.4% [30.5, 44.4] | 35.5% [27.9, 43.0] | 32.9% [25.3, 41.1] | 29.9% [22.9, 38.9] |
| length | 46.9% [41.0, 52.7] | 46.3% [40.5, 52.5] | 46.7% [40.6, 52.9] | 46.5% [40.0, 52.6] | 45.8% [39.4, 52.3] | 45.8% [39.3, 52.2] | 44.9% [39.0, 52.4] | 45.9% [38.4, 53.5] | 44.9% [37.3, 52.5] | 43.8% [35.4, 51.4] |
| length_residual | 46.5% [40.7, 52.7] | 44.4% [38.6, 51.0] | 43.0% [36.9, 49.2] | 41.3% [35.2, 47.8] | 39.4% [33.3, 46.3] | 38.3% [31.3, 45.3] | 36.4% [28.9, 43.3] | 33.7% [27.3, 41.3] | 34.2% [25.9, 41.1] | 31.2% [23.6, 40.3] |
| consistency_k3 | 45.8% [39.9, 52.0] | 44.0% [38.2, 50.2] | 41.8% [35.7, 48.4] | 39.6% [33.5, 46.5] | 37.5% [31.0, 44.5] | 34.3% [28.4, 42.3] | 33.7% [26.7, 40.6] | 31.4% [24.4, 40.1] | 30.4% [23.4, 38.0] | 29.9% [22.9, 38.2] |
| p_true | 46.2% [40.3, 52.4] | 44.4% [38.2, 50.6] | 41.4% [34.8, 48.4] | 38.7% [32.2, 45.2] | 36.1% [30.1, 43.1] | 33.8% [26.9, 40.8] | 31.0% [24.6, 38.5] | 28.5% [21.5, 36.0] | 25.3% [19.0, 32.9] | 22.9% [16.7, 30.6] |

`p_true` reaches half the base error rate (24.1%) at a budget of 0.50, where residual error is 22.9% [16.7, 30.6].

### Cutoff drift

For each scorer and budget: the cutoff a calibration-derived quantile would give, the flag rate that fixed cutoff actually produces on test, and the batch-quantile cutoff the budget policy used on test instead. A cutoff fixed on one draw does not deliver a fixed flag rate on another, which is why budget policies use batch quantiles.

| scorer | budget | calibration cutoff | flag rate on test at that cutoff | batch cutoff on test |
|---|---:|---:|---|---:|
| mean_logprob | 0.10 | 0.1641 | 19.8% [15.6, 24.8] | 0.2087 |
| mean_logprob | 0.20 | 0.1254 | 30.2% [25.2, 35.7] | 0.1633 |
| mean_logprob | 0.30 | 0.0898 | 43.1% [37.5, 48.8] | 0.1266 |
| min_k_logprob | 0.10 | 0.8432 | 13.5% [10.1, 18.0] | 0.9480 |
| min_k_logprob | 0.20 | 0.5701 | 30.9% [25.8, 36.5] | 0.7380 |
| min_k_logprob | 0.30 | 0.4556 | 37.5% [32.1, 43.2] | 0.6209 |
| length | 0.10 | 38.0000 | 16.3% [12.5, 21.0] | 44.0000 |
| length | 0.20 | 33.0000 | 29.2% [24.2, 34.7] | 36.0000 |
| length | 0.30 | 31.0000 | 33.7% [28.5, 39.3] | 32.0000 |
| length_residual | 0.10 | 0.0712 | 20.8% [16.5, 25.9] | 0.1317 |
| length_residual | 0.20 | 0.0416 | 28.5% [23.6, 33.9] | 0.0744 |
| length_residual | 0.30 | 0.0095 | 44.1% [38.5, 49.9] | 0.0379 |
| consistency_k3 | 0.10 | 0.1898 | 16.3% [12.5, 21.0] | 0.2353 |
| consistency_k3 | 0.20 | 0.1400 | 25.7% [21.0, 31.0] | 0.1642 |
| consistency_k3 | 0.30 | 0.0927 | 36.5% [31.1, 42.2] | 0.1293 |
| p_true | 0.10 | 1.0000 | 11.5% [8.3, 15.7] | 1.0000 |
| p_true | 0.20 | 0.9948 | 21.9% [17.5, 27.0] | 0.9968 |
| p_true | 0.30 | 0.7773 | 34.4% [29.1, 40.0] | 0.9399 |

### Cost tiers

What each scorer costs per score beyond the primary answer, from logged tokens and config/pricing.json, and what it buys at the 0.20 budget. Sorted by extra cost.

| scorer | extra calls | extra cost per 1000 | lift at 0.20 |
|---|---:|---|---|
| length | 0 | $0.00 [0.00, 0.00] | 1.14 [0.93, 1.40] |
| length_residual | 0 | $0.00 [0.00, 0.00] | 1.57 [1.35, 1.81] |
| mean_logprob | 0 | $0.00 [0.00, 0.00] | 1.61 [1.38, 1.84] |
| min_k_logprob | 0 | $0.00 [0.00, 0.00] | 1.57 [1.34, 1.80] |
| p_true | 1 | $0.01 [0.01, 0.01] | 1.79 [1.58, 2.03] |
| consistency_k3 | 3 | $0.07 [0.07, 0.07] | 1.71 [1.47, 1.93] |

### Prevalence sensitivity

Every CORRECT test row is kept (149) and INCORRECT rows are subsampled without replacement to the target prevalence, 300 times (seeds 0 to 299). Cells are the median and the 2.5 and 97.5 percentiles across subsamples. These are subsample intervals, not row bootstraps. Budget policy at 0.20 inside each subsample. The observed test prevalence is 48.3%.

![prevalence sensitivity](../figures/prevalence_sensitivity.png)

| scorer | prevalence | n | catch rate | lift | precision | residual error |
|---|---:|---:|---|---|---|---|
| mean_logprob | 0.05 | 157 | 50.0% [25.0, 75.0] | 2.45 [1.23, 3.68] | 12.5% [6.2, 18.8] | 3.2% [1.6, 4.8] |
| mean_logprob | 0.10 | 166 | 41.2% [23.5, 64.7] | 2.01 [1.15, 3.16] | 20.6% [11.8, 32.4] | 7.6% [4.5, 9.8] |
| mean_logprob | 0.15 | 175 | 42.3% [26.9, 57.7] | 2.12 [1.35, 2.88] | 31.4% [20.0, 42.9] | 10.7% [7.9, 13.6] |
| mean_logprob | 0.25 | 199 | 40.0% [32.0, 48.0] | 1.99 [1.59, 2.39] | 50.0% [40.0, 60.0] | 18.9% [16.4, 21.4] |
| min_k_logprob | 0.05 | 157 | 50.0% [12.5, 75.0] | 2.45 [0.61, 3.68] | 12.5% [3.1, 18.8] | 3.2% [1.6, 5.6] |
| min_k_logprob | 0.10 | 166 | 47.1% [23.5, 64.7] | 2.30 [1.15, 3.16] | 23.5% [11.8, 32.4] | 6.8% [4.5, 9.8] |
| min_k_logprob | 0.15 | 175 | 46.2% [30.8, 53.8] | 2.31 [1.54, 2.69] | 34.3% [22.9, 40.0] | 10.0% [8.6, 12.9] |
| min_k_logprob | 0.25 | 199 | 40.0% [34.0, 44.0] | 1.99 [1.69, 2.19] | 50.0% [42.5, 55.0] | 18.9% [17.6, 20.8] |
| length | 0.05 | 157 | 25.0% [0.0, 56.6] | 1.23 [0.00, 2.78] | 6.2% [0.0, 14.1] | 4.8% [2.8, 6.4] |
| length | 0.10 | 166 | 29.4% [11.8, 41.2] | 1.44 [0.57, 2.01] | 14.7% [5.9, 20.6] | 9.1% [7.6, 11.4] |
| length | 0.15 | 175 | 26.9% [15.4, 38.5] | 1.35 [0.77, 1.92] | 20.0% [11.4, 28.6] | 13.6% [11.4, 15.7] |
| length | 0.25 | 199 | 26.0% [18.0, 30.0] | 1.29 [0.90, 1.49] | 32.5% [22.5, 37.5] | 23.3% [22.0, 25.8] |
| length_residual | 0.05 | 157 | 50.0% [25.0, 87.5] | 2.45 [1.23, 4.29] | 12.5% [6.2, 21.9] | 3.2% [0.8, 4.8] |
| length_residual | 0.10 | 166 | 47.1% [26.3, 64.7] | 2.30 [1.29, 3.16] | 23.5% [13.2, 32.4] | 6.8% [4.5, 9.5] |
| length_residual | 0.15 | 175 | 42.3% [32.6, 57.7] | 2.12 [1.63, 2.88] | 31.4% [24.2, 42.9] | 10.7% [7.9, 12.5] |
| length_residual | 0.25 | 199 | 40.0% [32.0, 48.0] | 1.99 [1.59, 2.39] | 50.0% [40.0, 60.0] | 18.9% [16.4, 21.4] |
| consistency_k3 | 0.05 | 157 | 62.5% [25.0, 87.5] | 3.07 [1.23, 4.29] | 15.6% [6.2, 21.9] | 2.4% [0.8, 4.8] |
| consistency_k3 | 0.10 | 166 | 52.9% [29.4, 70.6] | 2.58 [1.44, 3.45] | 26.5% [14.7, 35.3] | 6.1% [3.8, 9.1] |
| consistency_k3 | 0.15 | 175 | 53.8% [36.4, 67.4] | 2.69 [1.82, 3.37] | 40.0% [27.1, 50.1] | 8.6% [6.1, 11.8] |
| consistency_k3 | 0.25 | 199 | 46.0% [41.0, 52.0] | 2.29 [2.04, 2.59] | 57.5% [51.2, 65.0] | 17.0% [15.1, 18.6] |
| p_true | 0.05 | 157 | 62.5% [37.5, 87.5] | 3.07 [1.84, 4.29] | 15.6% [9.4, 21.9] | 2.4% [0.8, 4.0] |
| p_true | 0.10 | 166 | 64.7% [47.1, 82.4] | 3.16 [2.30, 4.02] | 32.4% [23.5, 41.2] | 4.5% [2.3, 6.8] |
| p_true | 0.15 | 175 | 57.7% [46.2, 69.2] | 2.88 [2.31, 3.46] | 42.9% [34.3, 51.4] | 7.9% [5.7, 10.0] |
| p_true | 0.25 | 199 | 50.0% [44.0, 56.0] | 2.49 [2.19, 2.79] | 62.5% [55.0, 70.0] | 15.7% [13.8, 17.6] |

Lift for `p_true` is 2.49 [2.19, 2.79] at prevalence 0.25 and 3.07 [1.84, 4.29] at 0.05, so it holds above 1 across the whole range as prevalence falls. At 10 percent prevalence a reviewer working the flagged slice from `p_true` would see precision 32.4% [23.5, 41.2]: roughly 3 rows read per error found, against 10 at random.

## Cascade frontier

![cascade frontier](../figures/cascade_frontier.png)

Endpoints: all cheap error 47.7% [41.8, 53.5] at $0.02 [0.02, 0.02] per 1000; all strong error 26.4% [21.5, 31.8] at $0.39 [0.38, 0.41] per 1000. The strong model's error rate is the ceiling of any cascade. Budget points (circles) escalate the top ceil(budget * n) rows of the batch; risk points (triangles) escalate rows at or above a fixed cutoff from results/calibration.json, so their escalated fraction floats.

| scorer | policy | point | cutoff | escalated | error | cost per 1000 | gap closed | cost fraction | fix rate | break rate |
|---|---|---|---:|---|---|---|---|---|---|---|
| mean_logprob | budget | budget_0.1 | 0.2048 | 10.1% [7.1, 14.2] | 43.3% [37.2, 49.1] | $0.06 [0.05, 0.07] | 20.3% [9.3, 32.3] | 9.7% [6.4, 13.5] | 56.5% [36.8, 74.4] | 20.0% [3.6, 62.4] |
| mean_logprob | budget | budget_0.2 | 0.1596 | 20.2% [15.9, 25.3] | 40.8% [35.0, 46.6] | $0.11 [0.09, 0.13] | 32.2% [18.5, 46.0] | 23.1% [17.3, 29.0] | 48.8% [34.6, 63.2] | 15.4% [4.3, 42.2] |
| mean_logprob | budget | budget_0.3 | 0.1219 | 30.3% [25.2, 36.0] | 38.6% [32.9, 44.4] | $0.15 [0.13, 0.18] | 42.4% [27.6, 56.9] | 34.9% [28.6, 41.3] | 47.5% [35.3, 60.0] | 12.0% [4.2, 30.0] |
| mean_logprob | risk | risk_calcut_0.1 | 0.1641 | 18.8% [14.6, 23.8] | 40.8% [35.0, 46.6] | $0.10 [0.08, 0.13] | 32.2% [18.9, 46.0] | 21.3% [15.9, 27.1] | 48.8% [34.3, 63.5] | 9.1% [1.6, 37.7] |
| mean_logprob | risk | risk_calcut_0.2 | 0.1254 | 28.9% [23.9, 34.5] | 39.0% [33.2, 44.8] | $0.15 [0.12, 0.17] | 40.7% [26.0, 55.4] | 33.0% [26.7, 39.4] | 48.2% [35.7, 61.0] | 12.5% [4.3, 31.0] |
| mean_logprob | risk | risk_calcut_0.3 | 0.0898 | 41.5% [35.9, 47.4] | 35.0% [29.6, 40.4] | $0.20 [0.17, 0.23] | 59.3% [44.6, 73.3] | 47.5% [40.6, 54.1] | 50.6% [39.7, 61.5] | 10.5% [4.2, 24.1] |
| min_k_logprob | budget | budget_0.1 | 0.9311 | 10.1% [7.1, 14.2] | 44.0% [38.3, 49.8] | $0.07 [0.05, 0.09] | 16.9% [7.8, 27.3] | 11.9% [7.7, 16.7] | 41.7% [24.5, 61.2] | 0.0% [0.0, 49.0] |
| min_k_logprob | budget | budget_0.2 | 0.7263 | 20.2% [15.9, 25.3] | 41.9% [36.1, 47.7] | $0.11 [0.09, 0.14] | 27.1% [14.3, 40.0] | 24.1% [18.6, 30.0] | 42.9% [29.1, 57.8] | 14.3% [4.0, 39.9] |
| min_k_logprob | budget | budget_0.3 | 0.5545 | 30.3% [25.2, 36.0] | 38.3% [32.9, 44.0] | $0.16 [0.13, 0.18] | 44.1% [29.3, 58.5] | 36.2% [29.4, 42.8] | 49.2% [37.1, 61.4] | 17.4% [7.0, 37.1] |
| min_k_logprob | risk | risk_calcut_0.1 | 0.8432 | 13.0% [9.5, 17.5] | 43.7% [37.9, 49.5] | $0.08 [0.06, 0.10] | 18.6% [8.1, 30.2] | 15.4% [10.6, 20.8] | 41.4% [25.5, 59.3] | 14.3% [2.6, 51.3] |
| min_k_logprob | risk | risk_calcut_0.2 | 0.5701 | 30.0% [24.9, 35.6] | 38.3% [32.9, 44.0] | $0.15 [0.13, 0.18] | 44.1% [29.3, 58.5] | 35.4% [28.8, 42.1] | 49.2% [37.1, 61.4] | 18.2% [7.3, 38.5] |
| min_k_logprob | risk | risk_calcut_0.3 | 0.4556 | 35.7% [30.3, 41.5] | 36.8% [31.4, 42.6] | $0.18 [0.15, 0.21] | 50.8% [36.1, 64.8] | 42.1% [35.5, 48.7] | 50.0% [38.4, 61.6] | 12.9% [5.1, 28.9] |
| length | budget | budget_0.1 | 44.0000 | 10.1% [7.1, 14.2] | 45.1% [39.4, 50.9] | $0.08 [0.06, 0.10] | 11.9% [4.3, 21.0] | 15.6% [10.2, 21.3] | 41.2% [21.6, 64.0] | 0.0% [0.0, 25.9] |
| length | budget | budget_0.2 | 36.0000 | 20.2% [15.9, 25.3] | 44.0% [38.6, 50.2] | $0.12 [0.10, 0.15] | 16.9% [4.3, 29.6] | 27.4% [21.1, 34.2] | 45.2% [29.2, 62.2] | 16.0% [6.4, 34.7] |
| length | budget | budget_0.3 | 32.0000 | 30.3% [25.2, 36.0] | 42.2% [36.5, 48.0] | $0.17 [0.14, 0.20] | 25.4% [10.8, 39.3] | 39.5% [32.9, 46.5] | 45.5% [31.7, 59.9] | 12.5% [5.5, 26.1] |
| length | risk | risk_calcut_0.1 | 38.0000 | 15.9% [12.1, 20.7] | 44.0% [38.3, 49.8] | $0.11 [0.08, 0.13] | 16.9% [6.9, 28.0] | 22.6% [16.6, 29.0] | 45.8% [27.9, 64.9] | 5.0% [0.9, 23.6] |
| length | risk | risk_calcut_0.2 | 33.0000 | 28.9% [23.9, 34.5] | 43.0% [37.5, 48.7] | $0.16 [0.14, 0.19] | 22.0% [7.5, 36.0] | 37.8% [31.3, 44.8] | 42.9% [29.1, 57.8] | 13.2% [5.8, 27.3] |
| length | risk | risk_calcut_0.3 | 31.0000 | 33.2% [27.9, 39.0] | 41.5% [35.7, 47.3] | $0.18 [0.15, 0.22] | 28.8% [14.3, 42.9] | 43.4% [36.4, 50.5] | 46.8% [33.3, 60.8] | 11.1% [4.8, 23.5] |
| length_residual | budget | budget_0.1 | 0.1301 | 10.1% [7.1, 14.2] | 43.3% [37.5, 49.1] | $0.06 [0.04, 0.07] | 20.3% [9.3, 32.1] | 8.7% [5.8, 12.0] | 54.2% [35.1, 72.1] | 25.0% [4.6, 69.9] |
| length_residual | budget | budget_0.2 | 0.0722 | 20.2% [15.9, 25.3] | 40.1% [34.6, 45.8] | $0.10 [0.08, 0.12] | 35.6% [21.8, 50.0] | 20.4% [15.4, 25.6] | 52.3% [37.9, 66.2] | 16.7% [4.7, 44.8] |
| length_residual | budget | budget_0.3 | 0.0351 | 30.3% [25.2, 36.0] | 38.3% [32.9, 44.0] | $0.14 [0.12, 0.17] | 44.1% [29.0, 58.6] | 32.7% [26.7, 39.0] | 46.8% [34.9, 59.0] | 13.6% [4.7, 33.3] |
| length_residual | risk | risk_calcut_0.1 | 0.0712 | 20.2% [15.9, 25.3] | 40.1% [34.6, 45.8] | $0.10 [0.08, 0.12] | 35.6% [21.8, 50.0] | 20.4% [15.4, 25.6] | 52.3% [37.9, 66.2] | 16.7% [4.7, 44.8] |
| length_residual | risk | risk_calcut_0.2 | 0.0416 | 27.1% [22.2, 32.6] | 39.0% [33.2, 44.8] | $0.13 [0.11, 0.16] | 40.7% [26.1, 55.4] | 29.8% [23.9, 36.1] | 47.3% [34.7, 60.2] | 10.0% [2.8, 30.1] |
| length_residual | risk | risk_calcut_0.3 | 0.0095 | 42.6% [36.9, 48.5] | 33.9% [28.5, 39.4] | $0.19 [0.17, 0.22] | 64.4% [49.2, 79.2] | 45.5% [38.9, 52.3] | 51.9% [41.1, 62.6] | 7.7% [2.7, 20.3] |
| consistency_k3 | budget | budget_0.1 | 0.2343 | 10.1% [7.1, 14.2] | 43.3% [37.5, 49.1] | $0.13 [0.12, 0.15] | 20.3% [10.8, 32.7] | 29.1% [25.4, 33.3] | 48.0% [30.0, 66.5] | 0.0% [0.0, 56.2] |
| consistency_k3 | budget | budget_0.2 | 0.1642 | 20.2% [15.9, 25.3] | 39.4% [33.6, 45.5] | $0.17 [0.15, 0.19] | 39.0% [26.4, 53.6] | 39.1% [34.1, 44.4] | 48.9% [35.3, 62.8] | 0.0% [0.0, 29.9] |
| consistency_k3 | budget | budget_0.3 | 0.1282 | 30.3% [25.2, 36.0] | 35.7% [30.3, 41.5] | $0.21 [0.19, 0.24] | 55.9% [42.4, 71.2] | 50.9% [45.0, 57.0] | 52.2% [40.5, 63.7] | 11.8% [3.3, 34.3] |
| consistency_k3 | risk | risk_calcut_0.1 | 0.1898 | 16.2% [12.4, 21.0] | 40.4% [34.7, 46.2] | $0.15 [0.14, 0.17] | 33.9% [21.7, 47.5] | 35.2% [30.6, 40.3] | 52.6% [37.3, 67.5] | 0.0% [0.0, 35.4] |
| consistency_k3 | risk | risk_calcut_0.2 | 0.1400 | 25.6% [20.8, 31.1] | 38.3% [32.9, 44.0] | $0.19 [0.17, 0.21] | 44.1% [30.2, 58.6] | 45.3% [39.8, 51.3] | 49.1% [36.6, 61.7] | 14.3% [4.0, 39.9] |
| consistency_k3 | risk | risk_calcut_0.3 | 0.0927 | 36.5% [31.0, 42.3] | 34.3% [29.2, 40.1] | $0.23 [0.21, 0.26] | 62.7% [48.2, 77.1] | 56.7% [50.4, 63.0] | 52.6% [41.6, 63.5] | 12.0% [4.2, 30.0] |
| p_true | budget | budget_0.1 | 1.0000 | 10.1% [7.1, 14.2] | 44.0% [38.3, 49.8] | $0.08 [0.06, 0.09] | 16.9% [7.8, 27.5] | 14.5% [10.7, 18.9] | 43.5% [25.6, 63.2] | 0.0% [0.0, 43.4] |
| p_true | budget | budget_0.2 | 0.9968 | 20.2% [15.9, 25.3] | 39.0% [33.2, 44.8] | $0.12 [0.10, 0.14] | 40.7% [27.6, 54.9] | 25.5% [20.5, 30.9] | 51.1% [37.2, 64.7] | 0.0% [0.0, 29.9] |
| p_true | budget | budget_0.3 | 0.9047 | 30.3% [25.2, 36.0] | 34.7% [29.2, 40.1] | $0.16 [0.13, 0.18] | 61.0% [46.8, 76.6] | 35.8% [29.9, 41.8] | 52.9% [41.2, 64.3] | 0.0% [0.0, 19.4] |
| p_true | risk | risk_calcut_0.1 | 1.0000 | 10.8% [7.7, 15.0] | 43.3% [37.5, 49.1] | $0.08 [0.06, 0.10] | 20.3% [10.5, 32.1] | 15.1% [11.2, 19.6] | 48.0% [30.0, 66.5] | 0.0% [0.0, 43.4] |
| p_true | risk | risk_calcut_0.2 | 0.9948 | 21.3% [16.9, 26.5] | 38.3% [32.5, 44.0] | $0.12 [0.10, 0.14] | 44.1% [31.0, 58.6] | 26.5% [21.4, 32.0] | 52.0% [38.5, 65.2] | 0.0% [0.0, 29.9] |
| p_true | risk | risk_calcut_0.3 | 0.7773 | 33.9% [28.6, 39.7] | 33.9% [28.5, 39.4] | $0.17 [0.15, 0.19] | 64.4% [50.0, 79.6] | 39.3% [33.3, 45.5] | 52.0% [40.9, 62.9] | 5.3% [0.9, 24.6] |

Fix rate: among escalated rows the cheap model got wrong, the fraction the strong model got right. Break rate: among escalated rows the cheap model got right, the fraction the strong model got wrong. Gap closed: (error all cheap minus error at point) over (error all cheap minus error all strong). Cost fraction: (cost at point minus cost all cheap) over (cost all strong minus cost all cheap). Both bootstrapped jointly over rows. `risk_calcut_b` is the fixed cutoff that flagged the top b fraction on calibration; `risk_alpha_a` points appear only when calibrate_risk finds a threshold.

## Risk-coverage

![risk coverage](../figures/risk_coverage.png)

| scorer | AURC | oracle AURC |
|---|---|---:|
| mean_logprob | 0.308 [0.244, 0.379] | 0.143 |
| min_k_logprob | 0.301 [0.241, 0.371] | 0.143 |
| length | 0.395 [0.328, 0.472] | 0.143 |
| length_residual | 0.337 [0.266, 0.413] | 0.143 |
| consistency_k3 | 0.352 [0.270, 0.421] | 0.143 |
| p_true | 0.303 [0.218, 0.353] | 0.143 |

## Length

Length tertile edges (output tokens): 22, 31. Rows per tertile: [86, 105, 97]. Budget-policy lift at 0.20 applied within each tertile.

| scorer | short | medium | long |
|---|---|---|---|
| mean_logprob | 1.64 [1.14, 2.30] (n=86, err=35) | 1.60 [1.31, 2.04] (n=105, err=53) | 1.43 [1.10, 1.83] (n=97, err=51) |
| min_k_logprob | 2.05 [1.50, 2.57] (n=86, err=35) | 1.51 [1.23, 2.00] (n=105, err=53) | 1.62 [1.28, 2.03] (n=97, err=51) |
| length | 1.37 [0.87, 1.98] (n=86, err=35) | 0.85 [0.49, 1.30] (n=105, err=53) | 1.33 [0.97, 1.69] (n=97, err=51) |
| length_residual | 1.50 [1.06, 2.17] (n=86, err=35) | 1.70 [1.32, 2.04] (n=105, err=53) | 1.33 [1.04, 1.78] (n=97, err=51) |
| consistency_k3 | 1.64 [1.16, 2.30] (n=86, err=35) | 1.60 [1.28, 2.00] (n=105, err=53) | 1.71 [1.44, 2.14] (n=97, err=51) |
| p_true | 2.32 [1.73, 2.80] (n=86, err=35) | 1.60 [1.33, 2.04] (n=105, err=53) | 1.62 [1.32, 2.05] (n=97, err=51) |

## Conformal and required n

Learn-Then-Test with a Hoeffding bound at delta 0.1, calibrated on n=160.

| scorer | alpha | threshold | coverage (cal) | coverage (test) | selective risk (test) |
|---|---:|---|---|---|---|
| mean_logprob | 0.05 | none qualifies | | | |
| mean_logprob | 0.1 | none qualifies | | | |
| mean_logprob | 0.2 | none qualifies | | | |
| min_k_logprob | 0.05 | none qualifies | | | |
| min_k_logprob | 0.1 | none qualifies | | | |
| min_k_logprob | 0.2 | none qualifies | | | |
| length | 0.05 | none qualifies | | | |
| length | 0.1 | none qualifies | | | |
| length | 0.2 | none qualifies | | | |
| length_residual | 0.05 | none qualifies | | | |
| length_residual | 0.1 | none qualifies | | | |
| length_residual | 0.2 | none qualifies | | | |
| consistency_k3 | 0.05 | none qualifies | | | |
| consistency_k3 | 0.1 | none qualifies | | | |
| consistency_k3 | 0.2 | none qualifies | | | |
| p_true | 0.05 | none qualifies | | | |
| p_true | 0.1 | none qualifies | | | |
| p_true | 0.2 | none qualifies | | | |

Required calibration size for a threshold to pass the test (n_acc = ln(1/delta) / (2 (alpha - risk_hat)^2), divided by target coverage):

| alpha | target coverage | n if risk_hat = 0 | n if risk_hat = alpha/2 |
|---:|---:|---:|---:|
| 0.05 | 0.5 | 922 | 3685 |
| 0.05 | 0.7 | 658 | 2632 |
| 0.05 | 0.9 | 512 | 2047 |
| 0.1 | 0.5 | 231 | 922 |
| 0.1 | 0.7 | 165 | 658 |
| 0.1 | 0.9 | 128 | 512 |
| 0.2 | 0.5 | 58 | 231 |
| 0.2 | 0.7 | 42 | 165 |
| 0.2 | 0.9 | 32 | 128 |

## Cost table

From results/calls.jsonl. Total calls 4620, total cost $1.5855.

| purpose | calls | input tokens | output tokens | USD |
|---|---:|---:|---:|---:|
| calibration:judge_a | 324 | 64057 | 10597 | 0.2661 |
| calibration:judge_b | 324 | 64057 | 9628 | 0.2051 |
| calibration:primary | 162 | 6758 | 4116 | 0.0035 |
| calibration:ptrue | 162 | 12641 | 192 | 0.0020 |
| calibration:resample_1 | 162 | 6758 | 4132 | 0.0035 |
| calibration:resample_2 | 162 | 6758 | 4123 | 0.0035 |
| calibration:resample_3 | 162 | 6758 | 4143 | 0.0035 |
| calibration:strong | 162 | 6758 | 4552 | 0.0624 |
| test:judge_a | 600 | 120198 | 19467 | 0.4952 |
| test:judge_b | 600 | 120198 | 18527 | 0.3886 |
| test:primary | 300 | 12580 | 8503 | 0.0070 |
| test:ptrue | 300 | 24358 | 362 | 0.0039 |
| test:resample_1 | 300 | 12580 | 8637 | 0.0071 |
| test:resample_2 | 300 | 12580 | 8524 | 0.0070 |
| test:resample_3 | 300 | 12580 | 8521 | 0.0070 |
| test:strong | 300 | 12580 | 8868 | 0.1201 |

## Limitations

- Short-answer QA (HaluEval) only. Long multi-claim outputs need claim-level scoring (ROADMAP.md).
- OpenAI models only for generation. Logprob scorers need a provider that returns logprobs; consistency and P(True) do not.
- Labels are from two LLM judges from one provider, not humans. AMBIGUOUS rows are excluded and counted above.
- Conformal thresholds are calibrated on 160 rows. The required-n table shows what a guarantee would need.
- Budget routing needs the whole batch to take a quantile. Streaming needs a running quantile with a warmup window (ROADMAP.md).
- Lift intervals are wide at n of a few hundred. Treat differences between scorers whose intervals overlap as unresolved.

## Reproduce

```bash
python -m routing.cli score --split calibration
python -m routing.cli score --split test
python -m routing.cli gates
python -m routing.cli calibrate
python -m routing.cli evaluate
```
