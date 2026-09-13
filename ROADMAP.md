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
