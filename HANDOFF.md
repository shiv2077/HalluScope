# Handoff

Written 2026-09-14 at the end of the overnight build session.

## Status

The code for every phase is written, committed, and verified end to end on synthetic files
(`tests/test_routing_synthetic.py`, which never touches `data/` or `results/`). The question files
for both splits are committed. No generation, judging, figure, or number exists yet, because the
`OPENAI_API_KEY` in `.env` was rejected by OpenAI with a 401 at every attempt during the session.
Nothing was simulated or estimated.

## What to do first

1. Put a working key in `.env` (the only key; every model is OpenAI).
2. Run the background job. It verifies every model id in `config/models.json` against the models
   endpoint, then generates, judges, scores, gates, calibrates, and evaluates. It is resumable:
   every generation and label row is appended to its CSV as it lands, so a rerun continues from
   the last row.

```bash
nohup scripts/run_routing_phase1.sh > /dev/null 2>&1 &
```

```bash
tail -f logs/routing_phase1.log
```

If that job is already running from the overnight session, it retries every 10 minutes and picks
up the new key on the next attempt.

3. When `reports/routing_report.md` exists, fill the three headline numbers below from its
   Summary, commit with `routing: phase 1 data, generations, blind two-judge labels`,
   `routing: phase 2 scorers and validity gates`, and `routing: phase 3 evaluation, figures, report`
   (the CSVs, `results/*.md|json`, `figures/`, `reports/`), then update this file.

## Headline numbers

From `reports/routing_report.md`, Summary. Not available until the job above completes.

| headline | value |
|---|---|
| Lift at a 20 percent review budget, best scorer | pending |
| Residual error among unreviewed rows at 20 percent | pending |
| Cascade error and cost per 1000 at the best operating point, against all cheap and all strong | pending |

## The worst number

Pending the report. Read the Conformal section first: at n=162 the Learn-Then-Test threshold at
alpha 0.05 is expected to return no qualifying threshold, and the required-n table beside it says
how many calibration rows a guarantee would need. That is the honest result, not a bug.

## Regenerate everything from committed CSVs, no keys, CPU

```bash
python3 -m routing.cli score --split calibration && python3 -m routing.cli score --split test && python3 -m routing.cli gates && python3 -m routing.cli calibrate && python3 -m routing.cli evaluate
```

## Total API cost

From `results/calls.jsonl`, printed in the Cost table of the report and in `results/phase1_notes.md`.
Pending.

## Demo command

```bash
python3 -c "import pandas as pd; d=pd.read_csv('data/test_generations.csv'); d[d.role=='primary'].to_csv('/tmp/answers.csv', index=False)" && python3 -m routing.cli route --scorer mean_logprob --budget 0.20 --in /tmp/answers.csv --out /tmp/routed.csv
```

## Ten likely interview questions

Answers must be drawn only from numbers in `reports/routing_report.md`. Each answer below names the
section to read; fill in the numbers once the report exists.

1. **What is the lift at a 20 percent budget and is it worth infrastructure?** Review lift table,
   best unflagged scorer at budget 0.20, with its interval. Compare to the `length` control row.
   Lead with residual error, which is what a dataset team ships.
2. **How do you know the lift is not just answer length?** Validity gates table: the DeLong test
   against `length` per scorer, and the Length section: lift within tertiles. Any scorer flagged
   "not distinguishable from length" is marked in every figure.
3. **Where did the thresholds come from?** `results/calibration.json`: computed on the 162-row
   calibration split, applied once to the 300-row test split, file hash and n recorded. No
   threshold exists in config. The `route` command refuses to run without the calibration file.
4. **How good are the labels?** `results/phase1_notes.md`: agreement and kappa per split and model,
   AMBIGUOUS counts excluded from every headline number. Both judges are OpenAI models, which is
   recorded as a limitation.
5. **Does escalation actually fix anything?** Cascade table: fix rate and break rate with Wilson
   intervals, and the strong model's own error rate as the ceiling.
6. **What does the cascade cost?** Cascade table: cost per 1000 at each cutoff, computed from
   logged token counts and `config/pricing.json` with its date, against the all-cheap and
   all-strong endpoints.
7. **Why not just improve the detector?** `report1.md` and ADR-014: the clean v1 signal was
   AUROC 0.64 on 82 rows and the ensemble never beat mean logprob. Ordering is enough for a
   budget; a verdict is not available.
8. **Is the conformal threshold a guarantee?** Conformal section: threshold, coverage and realized
   risk per alpha, or "none qualifies", with the required-n table. At n=162 it is a function with
   a table beside it, not a guarantee.
9. **What happens with a provider that has no logprobs?** Scorer table: consistency and P(True)
   are text-only; their lift and their cost per score are on the same figure as the free scorers.
10. **What would you build next?** `ROADMAP.md`: claim-level scoring first, then 150 human labels,
    then the drift monitor. Nothing on that list was started.
