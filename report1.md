# HalluScope Audit Report (report1)

Audit date: 2026-09-13. Scope: README.md, config/*.json, the HRI engine and metrics, the adapters, and everything under eval/.
No source file was modified. All new code is under `eval/audit/`. All computation ran on CPU from committed files.

Conventions. Brackets are 95 percent intervals. AUROC, AURC, correlations and AUROC differences use a percentile bootstrap with 2000 resamples and seed 42. Proportions use Wilson intervals. Raw counts are exact counts of rows in committed files. They are not estimates, so they carry no interval. Power figures are analytic (Hanley-McNeil). They have no sampling uncertainty.

## 1. Verdict

The core claim does not hold. On the answers that can still be graded, local-mode HRI is at chance (AUROC 0.438 [0.000, 1.000], n=20). API-mode HRI has weak signal (AUROC 0.644 [0.519, 0.768], n=82, audit labels). The reported figures were 0.918 and 0.785.

Confidence in any detection claim is low. Nothing was measured on held-out data, and neither mode reaches the sample size needed for 80 percent power. The five-feature ensemble also never beats length-normalized mean logprob alone (API difference -0.043 [-0.109, 0.022]; gold set -0.052 [-0.106, -0.009]).

The load-bearing weakness is label leakage. The committed runs graded the intervention template instead of the model's answer whenever HRI was at or above 0.46. That makes 18 of 62 API positives and 80 of 97 local positives hallucinations by construction.

## 2. Claim verification (Phase 1)

### 2.1 Findings that drive the table

**F1. Label leakage.** The runs behind `eval/results_api.csv` and `eval/results_local.csv` had the intervention active during grading.
- At the time, `intervention_service.py` fired at HRI 0.46 (commits f25feeb5 and fdd1f176).
- Every fired row's `model_answer` is the fixed text "⚠️ High hallucination risk detected...". SBERT then scored that text against the reference.
- API: 18 rows hold the template, and all 18 are labelled mismatch. The lowest template-row HRI is 0.4642. The highest real-answer HRI is 0.4444.
- Local: 80 rows hold the template, and all 80 are labelled mismatch. The lowest template-row HRI is 0.5001. The highest real-answer HRI is 0.4209.
- So the top of the HRI ranking is a hallucination by definition in both modes.
- The current `run_benchmark.py` sends `apply_intervention: False`, but the committed CSVs were never regenerated after that fix.
- The real answers for those 98 rows are not stored anywhere in the repo.

**F2. Local results came from a formula that is no longer in the code.**
- The local CSV logs `mean_semantic_energy` as the raw free energy, between -32.99 and -22.49. The as-run feature was abs(E)/35. The repo's own fix commit 6e9a83ed calls this sign-inverted.
- The as-run `rcd_inv = 1 - min(2*RCD, 1)` is exactly 0 on 100 of 100 rows, because RCD runs from 0.534 to 0.785. One of the five local features contributed nothing.

**F3. The weights are hand-set.**
- `config/weights.json` has `"source": "heuristic-default"`.
- `eval/fit_weights.py` exists, but no output of it exists: no `eval/weights_fitted.json`, no train or test result CSV, and the source field never changed.
- The same literals (0.4/0.3/0.3, scale 10, bias -3) are hard-coded in commit f25feeb5, dated 2026-03-05, before any benchmark.

**F4. No held-out split was used.**
- `eval/pick_thresholds.py` reads `eval/results_api.csv`, the same file `eval/analyze.py` reports on.
- The SBERT threshold of 0.35 was chosen after viewing this run's cosine histogram (eval/REPORT.md, Methodology).
- `build_subset.py --split` produced `halueval_qa_train.json` and `halueval_qa_test.json` (1000 questions each), but no results exist on either.
- The 100-question eval subset lies entirely inside the train split: 100 of 100 questions overlap, and 0 fall in test.

**F5. The normalization constants are not computed by any code.**
- `rcd_scale = 2.0` carries the comment "typical RCD is ~0 to 0.5". The eval data contradicts this: every RCD is above 0.5.
- The removed divisor of 35 carries the comment "observed TinyLlama range |E| ~25 to 35". The evaluation run's |E| spans 22.5 to 33.0, which matches. Nothing records which data it was read from, so it may have come from the evaluation set.
- `prompt_perplexity_divisor = 100` is hand-set.

**F6. The grader was never calibrated against humans.**
- `eval/gold/gold_annotatorA.csv` and `gold_annotatorB.csv` have 0 of 80 labels filled in.
- `config/grader.json` still says 0.60, while every reported number uses 0.35.

### 2.2 Direct answers to the Phase 1 questions

| Question | Answer |
|---|---|
| Where did `config/weights.json` come from? | Hand-set. No fitting code was ever run. The values predate the benchmark (f25feeb5). |
| Were thresholds selected on the reported data? | Yes. The HRI threshold, the best-F1 threshold and the SBERT grader threshold were all chosen on the same 100 rows the results report. No held-out split exists in any result. The 0.30 threshold is a hard-coded fallback constant, because no threshold met the selection constraints. |
| n, class balance, positives per mode | API: n=100, 62 positives (mismatch), 38 negatives. Of the 62 positives, 18 are template rows. Local: n=100, 97 positives, 3 negatives. Of the 97 positives, 80 are template rows. |
| Correct answers in local mode (raw count) | 3 by the SBERT label. The audit judged 4 of the 20 readable local answers correct. The other 80 answers cannot be judged. |
| Were norm.json constants computed from the eval set? | No code computes them. The energy divisor of 35 matches the eval run's observed range, but its source data is unrecorded. The RCD scale is contradicted by the eval data. |

### 2.3 Claim table

| # | Claim (source) | Verdict | Evidence |
|---|---|---|---|
| 1 | API mode uses gpt-4o-mini, logprobs, top_logprobs=5, 3 features (README L20) | VERIFIED | `openai_adapter.py:52-59`; `FEATURE_NAMES_API` |
| 2 | Local mode uses TinyLlama with hidden states and raw logits, 5 features (README L22) | VERIFIED | `local_adapter.py`; `FEATURE_NAMES_LOCAL` |
| 3 | HRI = sigmoid(s * sum w_i f_i + b), constants read from config, s=10, b=-3 (README L31-38) | VERIFIED (current code) | `halluscope_core/risk/hri_engine.py`. The committed results come from older code with the same API literals hard-coded. |
| 4 | Weights are a learned combination (`eval/fit_weights.py` docstring) | UNSUPPORTED | Hand-set: `source: heuristic-default`, and no fitted file or train CSV exists (F3) |
| 5 | `logit_flatness = (logsumexp - max)/log V` drives local risk (README L45) | CONTRADICTED for the reported results | The code matches, but the reported local run used abs(-logsumexp)/35 (F2) |
| 6 | RCD reads the final layer (README L46-48) | VERIFIED | `local_adapter.py` `step_hs[-1]`. As run, `rcd_inv` was 0 on 100 of 100 rows. |
| 7 | Bands and intervention share one cutoff (README L55-57) | VERIFIED | Current config. At run time both were 0.46: 18 HIGH rows equals 18 intervened rows (API). |
| 8 | Abstains at a "calibrated threshold" (README L5) | CONTRADICTED | `operating_point.json`: `used_fallback: true`, `n_qualifying: 0`. 0.30 is `FALLBACK_HIGH` (`pick_thresholds.py:36`). |
| 9 | 100 questions sampled with seed 42 from HaluEval qa (README L71) | VERIFIED | The subset regenerates identically from the 10,000 cached rows |
| 10 | Correctness graded by SBERT cosine >= 0.35 (README L73-75) | VERIFIED as used | Stored cosines reproduce exactly (max abs diff 0.0). The threshold was picked on the eval run, and grader.json says 0.60. |
| 11 | API "62 / 38" hallucination / correct (README table) | CONTRADICTED as a hallucination count | The counts match the CSV, but 18 of the 62 are template text (F1) |
| 12 | API AUROC 0.785 (README table) | VERIFIED as arithmetic | 0.785 [0.687, 0.871] from the CSV |
| 13 | That AUROC measures detection of the model's own errors (README L217) | CONTRADICTED | Leakage. Real answers only: 0.696 [0.578, 0.809] (SBERT), 0.644 [0.519, 0.768] (audit). If the 18 lost answers were all correct: 0.473 [0.359, 0.591]. |
| 14 | API best-F1 threshold 0.086, P 0.743, R 0.887, F1 0.809 (README table) | VERIFIED as arithmetic | At 0.0858: P 0.743 [0.633, 0.829] (55/74), R 0.887 [0.785, 0.944] (55/62). Selected on the evaluation set. |
| 15 | Local "97 / 3" (README table) | CONTRADICTED as a hallucination count | 80 of the 97 are template text. SBERT correct count is 3. |
| 16 | Local AUROC 0.918, threshold 0.186, P 0.980, R 1.000, F1 0.990 (README table) | VERIFIED as arithmetic; CONTRADICTED as measurement | 0.918 [0.798, 1.000]; P 0.980 [0.929, 0.994]; R 1.000 [0.962, 1.000]. Real answers only: 0.529 [0.000, 1.000] (SBERT), 0.438 [0.000, 1.000] (audit). |
| 17 | Deployed 0.30: P 0.879, R 0.468, F1 0.611, flag rate 33% (README L85-86) | VERIFIED as arithmetic | P 0.879 [0.727, 0.952] (29/33), R 0.468 [0.349, 0.590], flag rate 0.33 [0.246, 0.427]. 18 of the 29 true positives are template rows. |
| 18 | 0.086 would flag about 80% of live traffic (README L84) | UNSUPPORTED | No live traffic was measured. On the benchmark it flags 73 of 100 (0.73 [0.636, 0.807]). |
| 19 | No committed local P/R at 0.30, and no baseline run (README L86-91) | VERIFIED | No `baseline_*` columns in any CSV |
| 20 | Local AUROC is dominated by class imbalance and has a wide CI (README L93-95) | VERIFIED, incomplete | CI [0.798, 1.000]; power 0.10 (Phase 2). Leakage is the larger defect. |
| 21 | Confident fabrication is the structural blind spot (README L99-105) | VERIFIED, larger than stated | 26 of 34 audit-incorrect API answers fall below the 0.30 trigger (0.76 [0.60, 0.88]). 13 of 34 are LOW (0.38 [0.24, 0.55]). Gold: 29 of 32 fall below 0.30 (0.91 [0.76, 0.97]). |
| 22 | Answerable questions score LOW, unknowable ones HIGH (README L120-122) | UNSUPPORTED | Three hand-written prompts in eval/REPORT.md, with no stored output |
| 23 | Experimental signals do not feed HRI (README L175-178) | VERIFIED | `compute_hri` reads only the 5 HRI features |
| 24 | Threshold tuned on the evaluated set; no held-out split (README L293-294) | VERIFIED | F4 |
| 25 | Reproduce commands regenerate the results (README L229-234, eval/REPORT.md) | CONTRADICTED | The current scripts write different columns, keep real answers (so labels change), and the committed CSVs have no seed column |
| 26 | Mean HRI 0.133 (correct) vs 0.359 (hallucination) (eval/REPORT.md L44) | VERIFIED as arithmetic | 0.133 [0.104, 0.167] vs 0.359 [0.290, 0.427]. The gap is partly by construction. |
| 27 | All 38 match rows below HRI 0.40; precision 1.000 at 0.46 (eval/REPORT.md L45) | VERIFIED as arithmetic; CONTRADICTED as evidence | P 1.000 [0.824, 1.000] (18/18). Those 18 rows are exactly the template rows. |
| 28 | 0.35 confirmed by manually inspecting 24 gray-zone rows (eval/REPORT.md L29) | UNSUPPORTED | No record of that inspection exists |
| 29 | Label noise is about 5 to 7 debatable rows (eval/REPORT.md L125) | CONTRADICTED | SBERT and audit disagree on 18 of 82 (agreement 0.780 [0.679, 0.856]) |
| 30 | Local mean HRI 0.277 (match) vs 0.782 (mismatch) (eval/REPORT.md L53) | VERIFIED as arithmetic | 0.277 [0.175, 0.421] vs 0.782 [0.726, 0.837], by construction |
| 31 | Internal-signal terms contribute meaningfully; energy clamps to 1.0 (eval/REPORT.md L54) | CONTRADICTED | `rcd_inv` is 0 on 100 of 100 rows. Energy spans 0.643 to 0.943 and clamps on 0 rows. Energy alone scores AUROC 0.189 [0.000, 0.547]. |
| 32 | v1 candidate evaluation: F1 0.349 at 0.46, mean HRI 0.416 (correct) > 0.392 (hallucinated) (eval/REPORT.md L16) | VERIFIED | F1 0.349; 0.416 [0.347, 0.491] vs 0.392 [0.314, 0.473]; AUROC 0.468 [0.354, 0.581] |
| 33 | No threshold meets all constraints; at 0.27, P 0.838, R 0.500, flag 37% (eval/REPORT.md L88) | VERIFIED | P 0.838 [0.689, 0.923], R 0.500 [0.379, 0.621], flag rate 0.37 [0.282, 0.468] |
| 34 | Normalization constants are principled (config/norm.json) | UNSUPPORTED | F5 |
| 35 | Grader threshold is justified against a gold subset (config/grader.json `_doc`) | CONTRADICTED | 0 of 80 gold rows are labelled (F6) |

## 3. Statistical power (Phase 2)

Script: `eval/audit/p2_power.py`. Figure: `eval/audit/figures/p2_auroc_ci.png`.

| Set | n | pos / neg | AUROC [95% CI] | CI width | Contains 0.5? |
|---|---:|---:|---|---:|---|
| API, as reported | 100 | 62 / 38 | 0.785 [0.687, 0.871] | 0.184 | No |
| API, real answers, SBERT label | 82 | 44 / 38 | 0.696 [0.578, 0.809] | 0.231 | No |
| API, real answers, audit label | 82 | 34 / 48 | 0.644 [0.519, 0.768] | 0.248 | No |
| API, bound if all 18 lost answers were correct | 100 | 44 / 56 | 0.473 [0.359, 0.591] | 0.232 | Yes |
| Local, as reported | 100 | 97 / 3 | 0.918 [0.798, 1.000] | 0.202 | No |
| Local, real answers, SBERT label | 20 | 17 / 3 | 0.529 [0.000, 1.000] | 1.000 | Yes |
| Local, real answers, audit label | 20 | 16 / 4 | 0.438 [0.000, 1.000] | 1.000 | Yes |
| Local, bound if all 80 lost answers were correct | 100 | 17 / 83 | 0.019 [0.000, 0.051] | 0.051 | No (at the bottom) |

Some bootstrap resamples contained a single class and were skipped. Valid resamples: 1913 of 2000 for local as reported, 1921 and 1978 for the two local real-answer sets, and 2000 for every other set.

The two bound rows show what the lost answers could mean. Leakage alone can move API AUROC anywhere between 0.473 and 0.785. For local, the range runs from 0.019 to 0.918. The committed data cannot say where the truth lies.

**Minimum n to detect AUROC 0.65 against 0.5, 80 percent power, two-sided alpha 0.05 (Hanley-McNeil):**

| Prevalence scenario | Required n (pos / neg) | Actual n | Power at actual n |
|---|---|---:|---:|
| Balanced classes | 114 (57 / 57) | n/a | n/a |
| API as reported (62% positive) | 119 (74 / 45) | 100 | 0.72 |
| API real answers, SBERT (54%) | 114 (61 / 53) | 82 | 0.65 |
| API real answers, audit (41%) | 118 (49 / 69) | 82 | 0.64 |
| Local as reported (97%) | 917 (889 / 28) | 100 | 0.10 |
| Local real answers, SBERT (85%) | 210 (178 / 32) | 20 | 0.10 |
| Local real answers, audit (80%) | 172 (138 / 34) | 20 | 0.12 |

- No evaluated set meets the required n.
- API is short by roughly 20 to 36 rows.
- Local mode needs about 28 to 34 negatives. It has 3 or 4, and power is 0.10 to 0.12.
- A local-mode AUROC from this data cannot distinguish a real detector from chance.

## 4. The three deciding experiments

Data used:
- The committed CSVs.
- `eval/gold/gold_results_api.csv`: 80 gpt-4o-mini answers generated with the current code (`apply_intervention` False, seed 42). It is the only committed run that logs every HRI feature. It was never part of a reported result. I labelled it here with SBERT at 0.35 and with the audit judgements.

The audit labels are in `eval/audit/judgements.py`. A single rater made them: the auditing model. They were not blind, because the SBERT score was visible while judging. Treat them as a second noisy label, not ground truth.

### E1. Ablation against the trivial baseline

Script: `eval/audit/e1_ablation.py`. Figures: `eval/audit/figures/e1_auroc_forest.png`, `eval/audit/figures/e1_spearman.png`.

Data limit:
- The reported API CSV logged only `hri` and `mean_logprob`. `logprob_variance` and `confidence_decay` were never logged, so on that run they cannot be scored separately.
- The combined term 0.4*var + 0.3*decay can be recovered by inverting the sigmoid. Checked on the gold set, where the true features exist, the recovery is exact to 0.00009.
- To score var and decay separately on the reported questions, the API run must be regenerated with the current `run_benchmark.py`, which logs every feature.

**Reported API run**

| Label set | n (pos/neg) | HRI | abs mean logprob | 0.4 var + 0.3 decay | answer tokens | DeLong HRI vs mean logprob |
|---|---|---|---|---|---|---|
| SBERT, all 100 as reported | 100 (62/38) | 0.785 [0.687, 0.871] | 0.783 [0.683, 0.871] | 0.771 [0.670, 0.861] | not available | p=0.952; diff 0.001 [-0.043, 0.042] |
| SBERT, 82 real | 82 (44/38) | 0.696 [0.578, 0.809] | 0.709 [0.591, 0.820] | 0.678 [0.556, 0.795] | 0.569 [0.446, 0.691] | p=0.666; diff -0.012 [-0.069, 0.044] |
| Audit, 82 real | 82 (34/48) | 0.644 [0.519, 0.768] | 0.686 [0.569, 0.800] | 0.617 [0.496, 0.742] | 0.500 [0.371, 0.630] | p=0.200; diff -0.043 [-0.109, 0.022] |

**Gold set, current code, all features logged**

| Label set | n (pos/neg) | HRI | abs mean logprob | logprob_variance | confidence_decay | answer tokens | DeLong HRI vs mean logprob |
|---|---|---|---|---|---|---|---|
| SBERT at 0.35 | 80 (37/43) | 0.701 [0.584, 0.809] | 0.737 [0.620, 0.839] | 0.728 [0.612, 0.835] | 0.591 [0.463, 0.713] | 0.629 [0.501, 0.744] | p=0.132; diff -0.036 [-0.086, 0.008] |
| Audit | 80 (32/48) | 0.780 [0.668, 0.870] | 0.833 [0.731, 0.916] | 0.797 [0.688, 0.891] | 0.639 [0.513, 0.754] | 0.671 [0.543, 0.789] | p=0.039; diff -0.052 [-0.106, -0.009] |

**Reported local run (as-run formula)**

| Label set | n (pos/neg) | HRI | abs mean logprob | energy abs(E)/35 | rcd_inv | 0.3 var + 0.2 decay | tokens |
|---|---|---|---|---|---|---|---|
| SBERT, all 100 | 100 (97/3) | 0.918 [0.798, 1.000] | 0.942 [0.837, 1.000] | 0.189 [0.000, 0.547] | 0.500 (constant 0) | not recoverable (HRI rounds to 1.0) | not available |
| SBERT, 20 real | 20 (17/3) | 0.529 [0.000, 1.000] | 0.686 [0.211, 1.000] | 0.333 [0.000, 0.941] | 0.500 (constant 0) | 0.588 [0.000, 1.000] | 0.353 [0.000, 0.947] |
| Audit, 20 real | 20 (16/4) | 0.438 [0.000, 1.000] | 0.453 [0.093, 1.000] | 0.406 [0.080, 0.842] | 0.500 (constant 0) | 0.438 [0.000, 1.000] | 0.453 [0.137, 0.892] |

Local DeLong, HRI against the best single feature:
- SBERT, all 100: p=0.150, diff -0.024 [-0.071, 0.000].
- SBERT, 20 real: p=0.093, diff -0.157 [-0.389, 0.000].
- Audit, 20 real: the best single feature is the constant `rcd_inv`, so the test is meaningless (p=0.773).

**Spearman correlations, gold set (n=80)**

| | abs mean logprob | logprob_variance | confidence_decay | answer tokens | SBERT cosine |
|---|---|---|---|---|---|
| HRI | 0.92 [0.87, 0.95] | 0.90 [0.84, 0.94] | 0.77 [0.63, 0.85] | 0.64 [0.46, 0.77] | -0.27 [-0.46, -0.06] |
| abs mean logprob | | 0.95 [0.90, 0.97] | 0.52 [0.34, 0.66] | 0.61 [0.43, 0.75] | -0.31 [-0.50, -0.09] |
| logprob_variance | | | 0.47 [0.27, 0.62] | 0.59 [0.41, 0.73] | -0.29 [-0.48, -0.07] |
| confidence_decay | | | | 0.45 [0.23, 0.62] | -0.15 [-0.37, 0.07] |
| answer tokens | | | | | -0.21 [-0.41, 0.00] |

**Spearman correlations, reported API run (82 real answers)**

| | abs mean logprob | 0.4 var + 0.3 decay | answer tokens | SBERT cosine |
|---|---|---|---|---|
| HRI | 0.89 [0.80, 0.93] | 0.98 [0.97, 0.99] | 0.45 [0.25, 0.61] | -0.21 [-0.41, -0.01] |
| abs mean logprob | | 0.81 [0.70, 0.88] | 0.53 [0.35, 0.67] | -0.26 [-0.47, -0.05] |
| 0.4 var + 0.3 decay | | | 0.40 [0.19, 0.57] | -0.19 [-0.39, 0.01] |
| answer tokens | | | | -0.13 [-0.33, 0.08] |

Local real answers (n=20): HRI vs abs mean logprob is 0.65 [0.27, 0.86], and energy vs tokens is 0.65 [0.23, 0.91]. `rcd_inv` is constant, so every correlation involving it is undefined. The full matrix is in `out/e1_ablation.json`.

**Answer: the ensemble adds nothing.**
- In every label set, the point AUROC of mean logprob alone is equal to or higher than HRI's.
- The one significant difference (gold, audit labels) favours mean logprob, by 0.052 [0.009, 0.106].
- HRI correlates 0.89 to 0.92 with mean logprob, and mean logprob correlates 0.95 with variance.
- In practice HRI is a rescaled mean logprob. Confidence decay is the weakest feature, and the local-only features are inert or inverted.

### E2. Length confound

Script: `eval/audit/e2_length.py`. Figures: `eval/audit/figures/e2_length_scatter.png`, `eval/audit/figures/e2_bucket_auroc.png`.

Token counts are re-tokenized from the logged answer text: o200k_base for gpt-4o-mini, and the TinyLlama tokenizer for local. Template rows have no answer text, so they are excluded. Generation was capped at 80 tokens. 8 of 82 API answers and 10 of 80 gold answers sit at or near the cap (78 tokens or more).

**Spearman correlation of answer token count with each quantity**

| Quantity | API, 82 real answers | Gold, 80 answers | Local, 20 real answers |
|---|---|---|---|
| HRI | 0.45 [0.25, 0.61] | 0.64 [0.46, 0.77] | -0.08 [-0.54, 0.38] |
| abs mean logprob | 0.53 [0.35, 0.67] | 0.61 [0.43, 0.75] | -0.20 [-0.61, 0.27] |
| logprob_variance | not logged | 0.59 [0.41, 0.73] | not logged |
| confidence_decay | not logged | 0.45 [0.23, 0.62] | not logged |
| 0.4 var + 0.3 decay (derived) | 0.40 [0.19, 0.57] | n/a | n/a |
| energy abs(E)/35 (as run) | n/a | n/a | 0.65 [0.23, 0.91] |
| SBERT cosine (the label source) | -0.13 [-0.33, 0.08] | -0.21 [-0.41, 0.00] | 0.16 [-0.36, 0.64] |
| SBERT label (1 = mismatch) | 0.12 [-0.09, 0.33] | 0.22 [0.00, 0.42] | -0.18 [-0.63, 0.46] |
| Audit label (1 = incorrect) | 0.00 [-0.22, 0.22] | 0.29 [0.07, 0.49] | -0.07 [-0.48, 0.41] |

**Answer length alone as a detector (AUROC)**
- API: 0.569 [0.446, 0.691] (SBERT label), 0.500 [0.371, 0.630] (audit label).
- Gold: 0.629 [0.501, 0.744] (SBERT), 0.671 [0.543, 0.789] (audit).

**AUROC of HRI within length tertiles**

| Set | Tokens | n | Error rate, SBERT | HRI AUROC, SBERT | Error rate, audit | HRI AUROC, audit |
|---|---|---:|---|---|---|---|
| API | 12 to 34 | 27 | 0.48 [0.31, 0.66] | 0.742 [0.512, 0.933] | 0.41 [0.25, 0.59] | 0.744 [0.513, 0.935] |
| API | 35 to 52 | 25 | 0.44 [0.27, 0.63] | 0.597 [0.365, 0.820] | 0.36 [0.20, 0.55] | 0.653 [0.400, 0.877] |
| API | 53 to 80 | 30 | 0.67 [0.49, 0.81] | 0.715 [0.496, 0.895] | 0.47 [0.30, 0.64] | 0.585 [0.360, 0.813] |
| Gold | 13 to 36 | 25 | 0.32 [0.17, 0.52] | 0.838 [0.631, 0.985] | 0.24 [0.11, 0.43] | 0.781 [0.533, 0.968] |
| Gold | 37 to 56 | 28 | 0.50 [0.33, 0.67] | 0.628 [0.406, 0.846] | 0.39 [0.24, 0.58] | 0.695 [0.490, 0.888] |
| Gold | 57 to 80 | 27 | 0.56 [0.37, 0.72] | 0.561 [0.335, 0.808] | 0.56 [0.37, 0.72] | 0.822 [0.626, 0.988] |

Local tertiles hold 6 or 7 answers each, with 0 to 2 negatives. They give no usable estimate. The values are in `out/e2_length.json`.

**Answer: HRI partly measures length, but not only length.**
- HRI correlates with token count (0.45 and 0.64), more strongly than with either correctness label.
- Length alone predicts the audit label on gold (0.671 [0.543, 0.789]).
- HRI keeps some signal inside length buckets, but every bucket interval is wide and most reach down to or below 0.5 at their lower end.
- In the longest API bucket, SBERT calls 0.67 [0.49, 0.81] of answers wrong, while the audit calls 0.47 [0.30, 0.64] wrong. Long answers dilute the cosine, so part of the SBERT label is itself a length artifact.
- With n=25 to 30 per bucket, this data cannot separate "length plus a small true signal" from "true signal only".

### E3. Oracle noise

Script: `eval/audit/e3_oracle.py`.
- Sample for human review: `eval/audit/oracle_sample.csv`. It covers all 100 API rows in a seed-42 random order, because the API eval set has exactly 100 rows.
- Companions: `oracle_sample_local.csv` and `oracle_sample_gold.csv`.
- Columns: question, model answer, reference answer, SBERT cosine, SBERT label at 0.35, audit judgement, ambiguity flag, note.
- Figures: `eval/audit/figures/e3_threshold_sweep.png`, `eval/audit/figures/e3_cosine_by_audit_label.png`.

18 of the 100 API sample rows are unjudgeable: the stored answer is the intervention template. SBERT labelled all 18 of them mismatch.

**Agreement between the SBERT label (0.35) and the audit judgement**

| Set | n judged | SBERT match, audit correct | SBERT match, audit incorrect | SBERT mismatch, audit correct | SBERT mismatch, audit incorrect | Agreement | Cohen's kappa | Agreement without ambiguous rows |
|---|---:|---:|---:|---:|---:|---|---|---|
| API | 82 | 34 | 4 | 14 | 30 | 0.780 [0.679, 0.856] | 0.566 [0.383, 0.732] | 0.782 [0.678, 0.859] |
| Gold | 80 | 36 | 7 | 12 | 25 | 0.762 [0.659, 0.842] | 0.518 [0.320, 0.698] | 0.787 [0.681, 0.864] |
| Local | 20 | 2 | 1 | 2 | 15 | 0.850 [0.640, 0.948] | 0.483 [-0.111, 1.000] | 0.842 [0.624, 0.945] |

**SBERT errors run mostly one way.** Most disagreements are short correct answers inside long explanations that score below 0.35.
- Yes/no answers: halu_q_0014, 0050, 0080, 0091.
- Dates or names stated inside a long sentence: halu_q_0028, 0043, 0047, 0062, 0097.

False matches are wrong answers that share topic words with the reference:
- halu_q_0001 names a different sanctuary.
- halu_q_0003 names a different Filmfare award.
- halu_q_0040 names a different singer.
- halu_q_0044 gives the wrong full name.

On API, SBERT labels 38 answers correct. The audit labels 48 correct.

**Headline AUROC against the SBERT threshold**

| Threshold | API as reported: pos/neg, AUROC | API 82 real: AUROC | Gold 80: AUROC | SBERT vs audit agreement (API real) |
|---|---|---|---|---|
| 0.25 | 52/48, 0.745 [0.641, 0.841] | 0.609 [0.484, 0.738] | 0.647 [0.523, 0.765] | 0.68 [0.58, 0.77] |
| 0.30 | 60/40, 0.769 [0.665, 0.859] | 0.669 [0.548, 0.786] | 0.667 [0.541, 0.784] | 0.76 [0.65, 0.84] |
| 0.35 | 62/38, 0.785 [0.687, 0.871] | 0.696 [0.578, 0.809] | 0.701 [0.584, 0.809] | 0.78 [0.68, 0.86] |
| 0.40 | 65/35, 0.742 [0.637, 0.838] | 0.643 [0.517, 0.761] | 0.632 [0.505, 0.750] | 0.74 [0.64, 0.83] |
| 0.45 | 75/25, 0.731 [0.616, 0.835] | 0.646 [0.512, 0.774] | 0.670 [0.549, 0.788] | 0.70 [0.59, 0.78] |
| 0.50 | 86/14, 0.721 [0.587, 0.846] | 0.648 [0.482, 0.809] | 0.658 [0.522, 0.785] | 0.59 [0.48, 0.69] |
| 0.55 | 92/8, 0.615 [0.444, 0.761] | 0.521 [0.335, 0.709] | 0.660 [0.462, 0.845] | 0.51 [0.41, 0.62] |
| 0.60 | 97/3, 0.703 [0.525, 0.838] | 0.635 [0.432, 0.796] | 0.613 [0.301, 0.909] | 0.45 [0.35, 0.56] |
| 0.65 | 97/3, 0.703 [0.525, 0.838] | 0.635 [0.432, 0.796] | 0.420 [0.000, 0.734] | 0.45 [0.35, 0.56] |
| 0.70 | 98/2, 0.658 [0.485, 0.826] | 0.581 [0.375, 0.778] | 0.420 [0.000, 0.734] | 0.44 [0.34, 0.55] |
| 0.75 | 100/0, undefined | undefined | 0.628 [0.481, 0.759] | 0.41 [0.31, 0.52] |

Local as reported swings from 0.862 [0.647, 0.977] at 0.25 to 0.929 [0.865, 0.981] at 0.30. It is undefined from 0.55 up, because every row is then a mismatch.

**Answer: the label is not trustworthy enough for the reported precision.**
- It disagrees with an independent reading on about 1 row in 4 or 5 (API 0.780, gold 0.762).
- Kappa is only moderate (0.566 [0.383, 0.732]).
- The chosen 0.35 is where both the headline AUROC and the agreement peak. That makes the choice best-available, not arbitrary. Even so, the best available SBERT label is wrong about 22 percent of the time.
- At the configured default of 0.60, agreement falls to 0.45 [0.35, 0.56] and only 3 negatives remain.
- Across the sweep, the API headline moves between 0.615 and 0.785 (point estimates).

## 5. Risk-coverage (Phase 4)

Script: `eval/audit/p4_risk_coverage.py`. Figure: `eval/audit/figures/p4_risk_coverage.png`.

Rule evaluated: answer if HRI < t, otherwise suppress. Coverage is the fraction answered. Selective risk is the error rate among answered rows.

| Mode, label set | n | AURC | Oracle AURC | Excess AURC | Coverage at 0.30 | Selective risk at 0.30 | Risk answering everything | Risk reduction at 0.30 | Wrong withheld / correct suppressed at 0.30 |
|---|---:|---|---|---|---|---|---|---|---|
| API, SBERT, all 100 as reported | 100 | 0.438 [0.309, 0.584] | 0.255 [0.170, 0.367] | 0.182 [0.098, 0.272] | 0.670 [0.573, 0.754] | 0.493 [0.377, 0.609] | 0.620 [0.522, 0.709] | 0.127 [0.069, 0.197] | 29 / 4 (18 of the 29 are template rows) |
| API, SBERT, 82 real | 82 | 0.406 [0.269, 0.565] | 0.183 [0.110, 0.283] | 0.223 [0.119, 0.327] | 0.817 [0.720, 0.886] | 0.493 [0.377, 0.609] | 0.537 [0.429, 0.640] | 0.044 [-0.005, 0.099] | 11 / 4 |
| API, audit, 82 real | 82 | 0.320 [0.196, 0.470] | 0.104 [0.059, 0.174] | 0.216 [0.116, 0.324] | 0.817 [0.720, 0.886] | 0.388 [0.280, 0.508] | 0.415 [0.314, 0.523] | 0.027 [-0.025, 0.082] | 8 / 7 |
| Local, SBERT, all 100 as reported | 100 | 0.907 [0.803, 1.000] | 0.869 [0.748, 1.000] | 0.038 [0.000, 0.086] | 0.140 [0.085, 0.221] | 0.857 [0.601, 0.960] | 0.970 [0.915, 0.990] | 0.113 [-0.020, 0.303] | 85 / 1 (80 of the 85 are template rows) |
| Local, SBERT, 20 real | 20 | 0.774 [0.517, 1.000] | 0.585 [0.356, 1.000] | 0.188 [0.000, 0.422] | 0.700 [0.481, 0.855] | 0.857 [0.601, 0.960] | 0.850 [0.640, 0.948] | -0.007 [-0.129, 0.086] | 5 / 1 |
| Local, audit, 20 real | 20 | 0.763 [0.503, 0.997] | 0.497 [0.248, 0.820] | 0.266 [0.000, 0.503] | 0.700 [0.481, 0.855] | 0.786 [0.524, 0.924] | 0.800 [0.584, 0.919] | 0.014 [-0.100, 0.123] | 5 / 1 |

Net of wrong answers withheld minus correct answers suppressed at 0.30:
- API as reported: 25 [15, 35].
- API real, SBERT: 7 [0, 14].
- API real, audit: 1 [-7, 8].
- Local real, both label sets: 4 [0, 8].

Because 18 API answers and 80 local answers were lost, the true risk of answering everything is only bounded: between 0.44 and 0.62 for API, and between 0.17 and 0.97 for local.

Correct answers suppressed at 0.30:
- API: 4 by the SBERT label, or 7 by the audit label, among real answers. The 18 template rows are also suppressed, with unknown correctness.
- Local: 1 among real answers. The 80 template rows are also suppressed, with unknown correctness.
- Local mode answers only 14 of 100 questions at this threshold.

The current 0.30 operating point is not measurably better than answering everything in either mode. Its apparent API benefit comes from the 18 template rows. On gradable answers with audit labels, it withholds 8 wrong answers at the cost of 7 correct ones.

## 6. What survives

**Claims the repo can still make honestly**
1. HalluScope is a working post-hoc pipeline. It scores logprob-derived uncertainty, bands the result, and can suppress an answer. The architecture claims in the README are accurate for the current code.
2. On gpt-4o-mini free-form HaluEval QA answers, HRI carries weak-to-moderate signal about correctness. That is 0.644 [0.519, 0.768] on the 82 surviving answers and 0.780 [0.668, 0.870] on the 80-question gold set, both with audit labels. These results are exploratory, single-rater, not blind, and not held out.
3. Length-normalized mean logprob alone performs at least as well as the HRI ensemble.
4. The v1 candidate-evaluation protocol produced an inverted, chance-level signal (AUROC 0.468 [0.354, 0.581]). Abandoning it was correct.
5. Confident fabrication is a real blind spot. 76 percent [60, 88] of wrong API answers pass below the 0.30 trigger.
6. The 100-question subset is deterministic and regenerable.

**Claims that must be removed or rescoped**
1. Remove local-mode AUROC 0.918, the local best-F1 row, and the local means. Nothing valid supports them.
2. Remove API AUROC 0.785, the best-F1 row, and P 0.879 / R 0.468 / F1 0.611 at 0.30. If kept at all, restate them as leakage-contaminated.
3. Remove "precision 1.000 above HRI 0.40" and "all rows above 0.40 were hallucinations".
4. Remove "calibrated threshold" and "precision-floor selection". 0.30 is a hard-coded fallback.
5. Remove "internal-signal terms contribute meaningfully" and the claim that energy clamps to 1.0.
6. Remove any wording that implies learned or fitted weights, or that the ensemble adds value over mean logprob.
7. Rescope "~80% of live traffic" to "73 of 100 benchmark questions".
8. Replace "about 5 to 7 debatable labels" with the measured disagreement of 18 of 82.
9. Mark the reproduce commands as not regenerating the committed CSVs.
10. Rescope "answerable questions score LOW, unknowable ones HIGH" to "three hand-picked prompts".
11. State in `config/grader.json` that 0.35, not 0.60, produced every reported number, and that it was chosen on the evaluation data.

## 7. Cheapest path to a defensible result

These estimates cover engineering and labelling time. API cost is small (about $0.02 per 100 questions, per eval/REPORT.md).

1. **Withdraw the contaminated numbers (0.5 h).** Put a notice in README.md and eval/REPORT.md that points to this report.
2. **Drop local mode from the claims for now (0 h).** Revisit it only with a model that yields at least about 35 correct answers in the evaluation set: Qwen2.5-1.5B or 3B, not TinyLlama. That rerun is roughly 3 h of mostly unattended GPU time plus 1 h of checks, and is optional.
3. **Regenerate API answers on disjoint splits (1.5 h).** Run `run_benchmark.py --mode api --eval-deterministic --seed 42` on 300 questions from `halueval_qa_train.json` and 300 from `halueval_qa_test.json`. The current script already keeps real answers and logs every feature. 300 test rows exceed the 114 to 119 needed for 80 percent power at AUROC 0.65. Detecting an HRI vs mean-logprob difference would need more rows, and that n was not computed here.
4. **Fix the label before using it (5 h).** Two humans independently label 150 train rows and 150 test rows (the gold workflow already exists). Choose the grader rule on the train labels only. Report grader agreement and kappa on the test labels. If SBERT agreement stays near 0.78, grade with the human labels on the test set instead of SBERT.
5. **Fit and select on train only (1 h).** Run `fit_weights.py` on the train CSV. Point `pick_thresholds.py` at the train CSV; it currently hard-codes `results_api.csv`, so it needs a path argument, about 0.5 h of that time.
6. **Evaluate once on test (2 h).** Generalize the `eval/audit/` scripts to take a CSV path. Report AUROC with CIs, DeLong for HRI vs mean logprob, length buckets, and risk-coverage with AURC at the train-selected threshold.
7. **Add the cheap baselines (2 h).** Run the text-similarity confound control and P(True) from `eval/baselines/` on the test split, and DeLong them against HRI.
8. **Rewrite the README results from the test split only (1 h).** Every number gets an interval and a script reference.

Total: about 13 h without local mode, or about 17 h with it.

## Appendix: reproduce this audit

Everything runs on CPU from the repo root. Outputs go to `eval/audit/out/` and figures to `eval/audit/figures/`.

```bash
python3 eval/audit/p1_verify.py        # Phase 1 numbers (also builds caches)
python3 eval/audit/p2_power.py         # Phase 2
python3 eval/audit/e1_ablation.py      # E1
python3 eval/audit/e2_length.py        # E2
python3 eval/audit/e3_oracle.py        # E3, writes oracle_sample*.csv
python3 eval/audit/p4_risk_coverage.py # Phase 4
```

Two caches are committed alongside the scripts:
- `out/token_counts.csv` was built once with `tiktoken` (o200k_base) and the cached TinyLlama tokenizer. If it is deleted, `tiktoken` must be installed to rebuild it.
- `out/gold_sbert_cos.csv` uses the same SBERT model as `eval/match.py`.

Data that does not exist, and would have to be generated to close the remaining gaps:
- (a) The real model answers for the 18 API and 80 local template rows. They are unrecoverable, so the runs must be regenerated.
- (b) Per-row `logprob_variance` and `confidence_decay` for the reported API run.
- (c) Any local-mode run under the current feature formula.
- (d) Any human correctness labels.
