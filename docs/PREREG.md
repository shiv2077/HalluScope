# itemtriage preregistration

Committed before any panel output exists. Any later change to this file is a deviation, recorded with its
reason in `results/itemtriage/evaluation.json` and in the report. The first evaluation on test writes
`results/itemtriage/unblinding.json` with the commit hash of this file and a timestamp.

## Question

When a dataset builder cannot run a large model panel, how much item-flaw detection can a panel of 5 to 8
models recover when each model's confidence signals are added to its answers? Detection is measured
against expert labels at fixed review budgets.

## Data and classes

MMLU-Redux 2.0: 5,700 four-option items in 57 subjects, expert-annotated into `ok`, `bad_question_clarity`,
`bad_options_clarity`, `no_correct_answer`, `multiple_correct_answers`, `wrong_groundtruth` and `expert`.

- **Primary positive class:** ground-truth error, meaning `wrong_groundtruth`, `no_correct_answer` or
  `multiple_correct_answers`.
- **Secondary positive class:** any category other than `ok`.
- **Unknown, excluded from both classes and every metric:** `expert`, items the annotator could not verify.
  The excluded count is always reported.
- **Item identity** is question plus options under NFKC, lowercase and collapsed whitespace, with
  punctuation kept. Identical items never cross folds. This is checked at split time.

## Split

Frozen in `data/itemtriage/splits/mmlu_redux.json` (sha256 prefix `a030abe5c2ae9db2`). The procedure:

1. Each subject's primary-class error rate is computed over its known items. Subjects are sorted by that
   rate ascending, ties broken by subject name.
2. The sorted list is cut into 11 consecutive blocks of 5 and a final block of 2.
3. In each block of 5, 2 subjects go to calibration and 3 to test. In the final block, 1 goes to each. That
   gives 23 calibration and 34 test subjects. `numpy.random.RandomState(42)` chooses only which positions
   inside each block are calibration, block by block in sorted order.
4. The 23 calibration subjects are sorted the same way and cut into 5 blocks of 4 and a final block of 3.
   One subject per block becomes validation, with positions chosen by `RandomState(43)`.

Result: calibration fit has 17 subjects, 1,691 known items and 49 primary positives. Validation has 6
subjects, 597 and 15. Test has 34 subjects, 3,380 and 117.

## Panel

The panel answers with direct letters and no reasoning. This is a deliberate low-cost regime: one forward
pass or one API token per item.

| Model | Family | Access |
|---|---|---|
| gpt-4o | OpenAI | API, first-token top 20 logprobs, temperature 0, seed 42 |
| gpt-4o-mini | OpenAI | API, same settings |
| Qwen2.5-3B-Instruct | Qwen | local, NF4 |
| Qwen2.5-1.5B-Instruct | Qwen | local, fp16 |
| Phi-3.5-mini-instruct | Microsoft | local, NF4 |
| SmolLM2-1.7B-Instruct | Hugging Face | local, fp16 |
| granite-3.1-2b-instruct | IBM | local, NF4 |

- **Pilot.** 300 items drawn from calibration fit with `RandomState(44)`. A model stays in the panel only if
  its letter parse rate is at least 95% and the Wilson lower bound of its accuracy against the key exceeds
  0.25. A failing local model is replaced by OLMo-2-0425-1B-Instruct, checked the same way. If fewer than
  5 models pass, the study stops and reports that.
- **Letter distributions.** API models pool the first-token logprobs of tokens equal to a letter after
  stripping whitespace. A letter absent from the top 20 takes the lowest returned logprob, an upper bound.
  Local models take the log-softmax at the answer position, pooled over each letter's single-token
  variants (`A` and ` A`). Distributions are renormalized over the 4 letters.
- **Option order.** Local models run on the original order and on one cyclic shift (options 2, 3, 4, 1).
  Each model's per-option probability is the mean over the two orders after mapping back to the original
  options. API models run the original order only.
- **P(True).** Each API model gets a second call showing the keyed answer and asking for one word, Yes or
  No. The score is log P(No) − log P(Yes), with case and space variants pooled and absent tokens floored
  as above.

## Scorers

All scorers are label-free, and higher means riskier. Each has a declared secondary tie-break key.

| Scorer | Definition | Tie-break key |
|---|---|---|
| `disagree` | Fraction of panel models whose most probable option is not the key | `conf` |
| `conf` | Mean over models of 1 − P_m(key) | `disagree` |
| `conf_w` (primary) | Σ_m w_m (1 − P_m(key)), with w_m = exp(θ_m) / Σ exp(θ), where θ_m comes from the stage-1 IRT fit | `conf` |
| `irt_small` | 4PL forced-ceiling Δℓ = ℓ(d=0) − ℓ(d=1), following Land and Bikel. Stage 1: on items with item-total r > 0.2, a constrained 3PL with one discrimination per subject, c = 0.25, d = 1, b in [-2, 2], a in [1/15, 15], and anchor penalty λ(θ̄² + (σ_θ − 1)²) with λ = 10, fit by L-BFGS-B. Stage 2: θ fixed, each item refit with a, b, c free under d = 0 and d = 1. Correctness is argmax equal to key. | `conf` |
| `irt_small_c` | The same contrast with c fixed at 0.25 and a fixed at the subject value from stage 1. Only b is refit per item. | `conf` |
| `ptrue_key` | Mean over the API models of the P(True) log-odds | `conf` |
| `length` (control) | Characters in the question plus options | `conf` |
| `dup_options` (control) | Maximum pairwise `difflib` similarity among the normalized options | `conf` |
| `lr` (supervised, secondary) | L2 logistic regression on standardized features `disagree`, `conf`, `conf_w`, the selected IRT variant, `ptrue_key`, `length` and `dup_options`. Fit on calibration fit, with C chosen from {0.01, 0.1, 1, 10} by average precision on validation. | `conf_w` |

- **IRT variant.** Exactly one of `irt_small` and `irt_small_c` is carried forward: whichever has the
  higher average precision for the primary class on validation. IRT is fit transductively on the full
  Redux response matrix, which is label-free.
- **Ties.** The budget policy flags the top ceil(budget × n) rows by (score, tie-break key). Rows still tied
  on both keys and straddling the cutoff are not ordered by row or qid. Metrics use the expected value
  under uniform resolution within the tied block. The residual tie count is reported.

## Metrics

All metrics are computed on the known test items.

- **Primary metric:** precision of the flagged slice at a 10% budget, primary class.
- **Secondary metrics:** precision at 5% and 20%, recall, lift over the base rate, errors found per 100
  reviewed, average precision, AUROC, and API dollars per error found from logged tokens. Each is given
  for both classes. The best possible precision at each budget is reported beside every precision.
- **Reviewer minutes per item** is an assumed parameter, labelled as an assumption wherever it appears.
- **Intervals:** 2000-resample bootstrap with seed 42 for differences between scorers, paired over items,
  with the budget quantile recomputed inside each resample. Proportions use Wilson intervals.

## Hypotheses and decision rules

- **H1a.** At a 10% budget, `conf_w` beats both `disagree` and the selected IRT variant on precision for
  the primary class. It holds only if both paired differences, `conf_w` minus `disagree` and `conf_w` minus
  the IRT variant, have intervals entirely above 0. Otherwise it is reported as a negative result.
- **H1b.** Not computable. The Writer release has 21 multiple-choice weak positives outside GPQA, below the
  required 100 (`reports/phase1_findings.md`). No external benchmark is run.
- **H2.** A linear probe on one local model's hidden states adds lift over the API tier. The model is
  Qwen2.5-1.5B-Instruct. It reads the multiple-choice prompt with the keyed letter appended as the
  assistant turn. Hidden states are captured at the keyed letter token and at the last prompt token.
  - **Features:** for each position, the mean over the middle third of layers, plus the layers at relative
    depth 0.5 and 0.75 at the key token. There is no other layer search.
  - **Fit:** PCA to 64 dimensions fitted on calibration fit, then an L2 logistic regression with C chosen
    on validation by average precision.
  - **Contrast:** `lr` plus the probe output against `lr` with the API-derived features only. H2 holds only
    if the paired interval of the precision difference at a 10% budget lies entirely above 0. Otherwise
    the tier is dropped and reported.
- **H3.** Fixed cutoffs, the 5%, 10% and 20% score quantiles of calibration fit plus validation, are
  applied to test. Their realized flag rates get Wilson intervals and are compared with the budget
  policy's exact rate. Learn-Then-Test with a Hoeffding bound is run on calibration fit plus validation at
  α in {0.05, 0.10, 0.20} and δ = 0.10, accepting items with scores at or below the threshold, with risk
  defined as the primary flaw rate among accepted items. For each α the report says whether it qualifies,
  the test coverage and risk with intervals, and the required n.

## Exploratory analyses

These carry no decision rule.

- Memorization: for each panel model, agreement with the key on flawed versus ok calibration items, with
  Wilson intervals.
- Every H1 metric with the virology subject excluded.
- Per-family leave-one-out: the primary metric with each family removed from the panel.

## Threats to validity

- **Memorized wrong keys.** MMLU is in pretraining data, so a model may reproduce a wrong key from memory
  and lower every response-based signal.
- **Correlated panel families.** Two OpenAI models and two Qwen models share training lineage, so their
  errors are correlated and the effective panel is smaller than 7.
- **The direct-letter regime.** No reasoning prompts are used. Land and Bikel's panel produced answers with
  justifications, so results here do not speak to reasoning-prompted panels.
- **Letter-position bias.** Local models are averaged over two orders. API models are not and keep a
  residual bias.
- **Base-rate skew.** Virology holds many errors. The stratified split spreads high-error subjects across
  folds, and a virology-excluded analysis is reported.
- **Label noise.** Expert labels are not perfect, and `expert` items are excluded rather than resolved.
- **Weak labels elsewhere.** Writer's weak labels came from an aggregator reading strong-model arguments.
  That favours a full panel and is why H1b's threshold was not relaxed.
