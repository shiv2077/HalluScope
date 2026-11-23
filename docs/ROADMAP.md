# Roadmap

Everything here is deliberately not built. Each item has a reason it waits. The first section is itemtriage,
the second is routing.

## itemtriage

- **API panel point.** The preregistered `gpt-4o` and `gpt-4o-mini` panel never ran because the key was invalid
  (PREREG Amendment 1). With a working key, `python3 -m itemtriage.cli panel-api --pilot` then `--full` costs
  a projected $4.53 (`reports/phase1_findings.md`). The result is a separate post-unblinding analysis, plotted on the Study 2 axes.
- **A second benchmark with item-level expert labels.** Platinum GSM8K gives 300 items. A larger multiple-choice
  benchmark with expert error labels and a public response matrix would let H5 and H6 run with real power.
- **Human spot check.** 150 blind items are frozen in `data/itemtriage/human_sample.json`. Label them with
  `python3 -m itemtriage.cli label --annotator <name>`, then `python3 -m itemtriage.cli report`.
- **Reasoning-prompted panels.** Out of scope: every panel here answers directly. Reasoning may change which
  scorer wins.

## routing

## 1. Claim-level scoring

Mean logprob over a 300-token answer washes out one hallucinated sentence. Dataset items at
production scale are long and multi-claim. Plan: extract atomic claims from the answer, map each
claim to its token span, score each span with the same scorer interface, and route at the claim
level. The extraction step is the hard part and needs its own validity gates.

## 2. Human labels on 150 rows

The current labels come from two blind LLM judges with a consensus rule, and the strong model in the cascade
is also judge A. Plan: draw and label a 150-row sample with `routing/label_tool.py`, blind to every score,
report kappa against each judge and the consensus, and re-run the report on the human-labelled rows.

## 3. Drift monitor

A cutoff calibrated on one score distribution is wrong once the model, prompt style, or domain
moves. Plan: log the score distribution per deployment, compare it to the calibration
distribution with a two-sample test on a rolling window, and flag recalibration when it moves.

## 4. Additional providers

Only OpenAI generation was run. Consistency and P(True) are text-only and already work with any
provider; the logprob scorers need a provider that returns logprobs. Plan: an Anthropic and a
local HuggingFace adapter behind the same `generate.py` role interface, with the router picking
the cheapest scorer the provider supports.

## 5. Required n for a conformal guarantee

The Learn-Then-Test function is implemented and run at n=160, and returns no threshold. ADR-028 shows its fixed
sequence could not qualify for four of five scorers on any data, so the first step is fixing the sequence start. The required-n table in the report says how many calibration rows a
guarantee at each alpha and coverage would need. Plan: label that many rows, then and only then
present the risk-controlled threshold as a guarantee.

## 6. Streaming budget policy

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
