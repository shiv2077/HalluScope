# Roadmap

Everything here is deliberately not built. The interview build is scoped to short-answer QA,
OpenAI generation, LLM judges, and six scorers. Each item below has a reason it waits.

## 1. Claim-level scoring

Mean logprob over a 300-token answer washes out one hallucinated sentence. Dataset items at
production scale are long and multi-claim. Plan: extract atomic claims from the answer, map each
claim to its token span, score each span with the same scorer interface, and route at the claim
level. The extraction step is the hard part and needs its own validity gates.

## 2. Human labels on 150 rows

The current labels come from two blind LLM judges with a consensus rule. The gold annotator
files under `eval/gold/` exist and are empty. Plan: two people label a 150-row subsample of the
test split blind to every score, report kappa against the LLM consensus, and re-run the report
on the human-labelled rows.

## 3. Drift monitor

A cutoff calibrated on one score distribution is wrong once the model, prompt style, or domain
moves. Plan: log the score distribution per deployment, compare it to the calibration
distribution with a two-sample test on a rolling window, and flag recalibration when it moves.

## 4. Additional providers

Only OpenAI generation was run. Consistency and P(True) are text-only and already work with any
provider; the logprob scorers need a provider that returns logprobs. Plan: an Anthropic and a
local HuggingFace adapter behind the same `generate.py` role interface, with the router picking
the cheapest scorer the provider supports.

## 5. Hidden-state scorers behind the same interface

The v1 local adapter captures per-layer hidden states. The HalluScope-v2 study in this
repository is about which hallucinations only hidden states can see. Plan: a hidden-state
scorer that returns a higher-is-riskier float and a cost record like the other six, so it
lands on the same lift curve and the same gates.

## 6. Required n for a conformal guarantee

The Learn-Then-Test function is implemented and run. At n=162 it either returns no threshold or
a coverage too low to use. The required-n table in the report says how many calibration rows a
guarantee at each alpha and coverage would need. Plan: label that many rows, then and only then
present the risk-controlled threshold as a guarantee.

## 7. Streaming budget policy

The budget policy (ADR-018) takes a quantile over the whole batch, so it cannot flag a row until
the batch is complete. Plan: a running quantile estimator with a warmup window of a few hundred
rows, during which the layer either holds rows or falls back to the risk policy, plus a report of
the realized flag rate against the budget over time.

## Streaming and production

Design only. Nothing here is built.

- **Batched routing**: the budget policy as shipped. Take the quantile over each batch, flag
  the top ceil(budget * n), and log the batch cutoff beside the flag count so drift is visible
  batch to batch.
- **Streams**: a running quantile estimator (P-squared or a fixed-size reservoir) with a
  warmup window of a few hundred rows. During warmup the layer either holds rows until the
  window fills or falls back to the risk policy with a calibrated cutoff. Report the realized
  flag rate against the budget over a sliding window.
- **Calibration refresh**: re-run `calibrate` on a fixed cadence (weekly, or every N thousand
  routed rows) on a fresh labelled sample, and keep every `calibration.json` with its hash, n,
  and date so a cutoff can always be traced to the sample that produced it.
- **Drift monitor**: a two-sample test (Kolmogorov-Smirnov on the score distribution, or a
  chi-square on the flagged fraction) between the live window and the calibration
  distribution, with the alert threshold chosen for a fixed false-alarm rate. An alert means
  refresh calibration, not tune the cutoff.

## itemtriage follow-ups

- **API panel.** The preregistered panel included gpt-4o and gpt-4o-mini. The key was invalid, so the study
  ran on five local models (PREREG Amendment 1). With a working key, `python3 -m itemtriage.cli panel-api --pilot`
  then `--full` costs about $4.30 in total. Its results are a second analysis, reported separately as post-unblinding.
- **Human spot check.** 150 blind items are frozen in `data/itemtriage/human_sample.json`. Label them with
  `python3 -m itemtriage.cli label --annotator <name>`, then `python3 -m itemtriage.cli report`.
- **Routing risk lane.** `routing.calibrate.calibrate_risk` cannot qualify for scorers without ties at their minimum
  (ADR-028). Start its fixed sequence at a fixed coverage and regenerate the routing outputs.
