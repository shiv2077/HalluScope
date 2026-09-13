# Handoff

Written 2026-09-14 after phases 0 to 5 ran to completion. Every number below is copied from
`reports/routing_report.md` or `results/phase1_notes.md`. Brackets are 95 percent intervals.

## The three headline numbers

| headline | value |
|---|---|
| Lift at a 20 percent review budget, best scorer (`p_true`) | 1.78 [1.56, 2.03]. Catches 38.8% [30.3, 47.2] of errors against 21.9% [17.5, 27.0] for random. |
| Residual error among unreviewed rows at that budget | 37.8% [31.4, 44.0], from a base error rate of 48.3% [42.6, 54.0]. |
| Cascade at the best operating point (`p_true`, 30 percent budget) | Error 33.9% [28.5, 39.4] at $0.17 [0.15, 0.19] per 1000 queries. All cheap: 47.7% [41.8, 53.5] at $0.02. All strong: 26.4% [21.5, 31.8] at $0.39. That is 44 percent of strong-model cost for 61 percent of the error reduction. |

Pitch with the blanks filled: "On a held-out set of 300 questions with blind two-judge labels,
reviewing the top 20 percent by risk catches 39 percent of errors against 22 for random, and the
cascade brings error from 48 to 34 percent at 44 percent of strong-model cost."

## The worst number, and why it is there

**Conditional fix rate 52.0% [40.9, 62.9]** (`p_true`, 30 percent budget). Half of the wrong
answers the cascade escalates stay wrong, because the strong model is also wrong on 26.4 percent
of test questions and hard questions are hard for both. This is the ceiling of any cascade on
this data and it is reported on purpose, next to the break rate (5.3% [0.9, 24.6]), so the
frontier cannot be read as "escalation fixes what it catches".

Runner up: **the budget cutoffs drift.** The 20 percent cutoff calibrated on 160 rows flags
21.9 percent of test rows for `p_true` but 30.2 percent for `mean_logprob`. The calibration
questions come from the v1 100-question subset and the gold set; the test questions come from
`halueval_qa_test.json`. Same dataset, different draw, and the logprob distribution moved. This
is the drift monitor on the roadmap, observed rather than hypothesized.

Also honest: the conformal step returns **no risk-controlled threshold** for any scorer at any
alpha with n=160. The required-n table says alpha 0.10 at 70 percent coverage needs 165 rows if
the empirical risk were zero and 658 rows if it were 0.05. Our empirical risk is nowhere near
zero, so no threshold qualifies. A guarantee is a function plus a sample size, and we have the
function.

## Regenerate everything

From the committed CSVs, CPU only, no keys, about 90 seconds, byte-identical report on rerun:

```bash
python3 -m routing.cli score --split calibration && python3 -m routing.cli score --split test && python3 -m routing.cli gates && python3 -m routing.cli calibrate && python3 -m routing.cli evaluate
```

To redo the API phases from scratch (resumable, needs `OPENAI_API_KEY`):

```bash
nohup scripts/run_routing_phase1.sh > /dev/null 2>&1 &
```

## Total API cost

From `results/calls.jsonl`, priced on 2026-09-14: 4620 calls, 502199 input tokens, 122892 output
tokens, **$1.59**. Judging was 86 percent of it ($1.36); all generation for both splits was $0.23.

## Demo command

```bash
python3 -c "import pandas as pd; d=pd.read_csv('data/test_generations.csv'); d[d.role=='primary'].to_csv('/tmp/answers.csv', index=False)" && python3 -m routing.cli route --scorer min_k_logprob --budget 0.20 --escalate-budget 0.30 --in /tmp/answers.csv --out /tmp/routed.csv
```

Output on the 300 test rows: 184 return, 23 escalate, 93 review, with the cutoff and the
calibration hash printed. It refuses to run without `results/calibration.json`.

## Ten likely interview questions

1. **What is the lift at a 20 percent budget and is it worth infrastructure?** The best scorer,
   P(True), gives lift 1.78 [1.56, 2.03] at one extra cheap call per row; the free logprob
   scorers give 1.43 to 1.49. The number a dataset team ships is residual error: 37.8% [31.4,
   44.0] after review against 48.3% [42.6, 54.0] without. The interval on the free scorers
   excludes 1.0 and excludes the length control, so the lift is real and cheap, but it is not
   large.

2. **How do you know the lift is not just answer length?** Length alone has AUROC 0.580 [0.517,
   0.645] and lift 1.11 [0.93, 1.30], and every scorer beats it under a DeLong test at p at or
   below 0.001. The length-residualized logprob keeps AUROC 0.721 [0.664, 0.782] after regressing
   length out. Within length tertiles the lift stays above 1 for every scorer except length
   itself, though mean logprob weakens to 1.20 [1.01, 1.41] on the longest third.

3. **Where did the thresholds come from?** Every cutoff was computed on the 160-row calibration
   split and applied once to the 300-row test split; `results/calibration.json` carries the file
   hash and n. There is no threshold in config, and the `route` command refuses to run without
   the calibration file. Splits are disjoint by question id and by question text, asserted at
   load time.

4. **How good are the labels?** Two blind judges, `gpt-4o` and `gpt-4.1`, agree on 96.3% [93.6,
   97.9] of cheap-model test answers with kappa 0.927 [0.881, 0.967]; 12 of 300 rows are
   AMBIGUOUS and excluded. Both judges are OpenAI models, which is written down as a limitation
   because two judges from one provider share failure modes. Human labels on 150 rows are the
   first roadmap item.

5. **Does escalation actually fix anything?** Only about half the time: the conditional fix rate
   is 52.0% [40.9, 62.9] at the best operating point. The strong model's own error rate on test
   is 26.4% [21.5, 31.8], and that is the floor no router can beat. The break rate, where
   escalation turns a right answer wrong, is 5.3% [0.9, 24.6].

6. **What does the cascade cost?** Cost is computed from logged token counts and
   `config/pricing.json` dated 2026-09-14, never typed in. At the best P(True) point it is $0.17
   [0.15, 0.19] per 1000 queries, against $0.02 for all cheap and $0.39 for all strong. That
   buys 61 percent of the possible error reduction for 44 percent of the strong-model cost.

7. **Why not just improve the detector?** The clean v1 signal was AUROC 0.64 on 82 rows and the
   five-feature ensemble never beat mean logprob alone (report1.md). On this build the best
   scorer reaches AUROC 0.799 [0.750, 0.847], which still cannot decide correctness for a single
   row. It can order rows, and ordering is what a review budget or an escalation budget needs.

8. **Is the conformal threshold a guarantee?** No, and the report says so in every row: at n=160
   no scorer has a threshold whose Hoeffding bound clears alpha 0.05, 0.10 or 0.20. The
   required-n table shows alpha 0.10 at 70 percent coverage needs 165 rows at zero empirical
   risk and 658 at risk 0.05. The function is correct and shipped; the sample size is not.

9. **What happens with a provider that has no logprobs?** Two of the six scorers are text-only.
   Self-consistency over three resamples reaches AUROC 0.730 [0.668, 0.786] and lift 1.65 [1.45,
   1.87]; P(True) is the best scorer overall at 0.799. Their cost per score is on the same figure
   as the free scorers, so the buyer sees what each tier buys.

10. **What would you build next?** Claim-level scoring first, because mean logprob weakens on the
    longest tertile (1.20 [1.01, 1.41]) and production content is long. Then 150 human labels to
    check the LLM consensus, then the drift monitor, because the calibrated 20 percent cutoff
    already flagged 30 percent of test rows for mean logprob. Nothing on that list was started.
