# Validity gates on the test split

Brackets are 95 percent bootstrap intervals (2000 resamples, seed 42). Positives are INCORRECT rows.

## Gates

- Template contamination: 0 answers match the v1 warning banner (PASS).
- Split disjointness: calibration and test qids disjoint (PASS).
- Rows with a consensus label: 288 (139 INCORRECT, 149 CORRECT). AMBIGUOUS rows excluded: 12.
- Power to detect AUROC 0.65 versus 0.5 at the actual n: 0.99
- n required for 80 percent power at the observed prevalence: 114 (55 positives, 59 negatives). PASS.

## Scorers

| scorer | n | AUROC | vs length: diff, p | vs mean_logprob: diff, p | flag |
|---|---:|---|---|---|---|
| mean_logprob | 288 | 0.735 [0.679, 0.789] | 0.155 [0.082, 0.229], p=0.000 | n/a |  |
| min_k_logprob | 288 | 0.741 [0.685, 0.796] | 0.162 [0.107, 0.219], p=0.000 | 0.007 [-0.016, 0.030], p=0.574 |  |
| length | 288 | 0.580 [0.517, 0.645] | n/a | -0.155 [-0.229, -0.082], p=0.000 | control |
| length_residual | 288 | 0.721 [0.664, 0.782] | 0.141 [0.056, 0.226], p=0.001 | -0.014 [-0.033, 0.005], p=0.157 |  |
| consistency_k3 | 288 | 0.730 [0.668, 0.786] | 0.150 [0.073, 0.224], p=0.000 | -0.005 [-0.061, 0.049], p=0.857 |  |
| p_true | 288 | 0.817 [0.771, 0.861] | 0.238 [0.163, 0.312], p=0.000 | 0.082 [0.019, 0.147], p=0.017 |  |

A scorer is flagged when its AUROC does not beat `length` at p < 0.05 (DeLong). Flagged scorers carry a marker in every later figure.
