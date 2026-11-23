# Validity gates on the test split

Brackets are 95 percent bootstrap intervals (2000 resamples, seed 42). Positives are INCORRECT rows.

## Gates

- Template contamination: 0 answers match the v1 warning banner (PASS).
- Split disjointness: calibration and test qids disjoint (PASS).
- Rows with a consensus label: 287 (135 INCORRECT, 152 CORRECT). AMBIGUOUS rows excluded: 13.
- Power to detect AUROC 0.65 versus 0.5 at the actual n: 0.99
- n required for 80 percent power at the observed prevalence: 115 (54 positives, 61 negatives). PASS.

## Scorers

| scorer | n | AUROC | vs length: diff, p | vs mean_logprob: diff, p | flag |
|---|---:|---|---|---|---|
| mean_logprob | 287 | 0.728 [0.668, 0.784] | 0.152 [0.077, 0.225], p=0.000 | n/a |  |
| min_k_logprob | 287 | 0.736 [0.677, 0.791] | 0.160 [0.103, 0.218], p=0.000 | 0.008 [-0.015, 0.033], p=0.500 |  |
| length | 287 | 0.576 [0.508, 0.642] | n/a | -0.152 [-0.225, -0.077], p=0.000 | control |
| length_residual | 287 | 0.717 [0.657, 0.774] | 0.141 [0.050, 0.227], p=0.001 | -0.011 [-0.030, 0.008], p=0.241 |  |
| consistency_k3 | 287 | 0.727 [0.664, 0.787] | 0.151 [0.075, 0.227], p=0.000 | -0.002 [-0.055, 0.055], p=0.952 |  |
| p_true | 287 | 0.819 [0.770, 0.864] | 0.243 [0.162, 0.320], p=0.000 | 0.091 [0.024, 0.158], p=0.009 |  |

A scorer is flagged when its AUROC does not beat `length` at p < 0.05 (DeLong). Flagged scorers carry a marker in every later figure.
