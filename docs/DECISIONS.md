# Architecture decision records

Format per entry: Context (the forcing constraint) → Decision → Alternatives rejected (with the
reason they lost under these constraints) → Consequences (including the downside) → Revisit if →
Evidence. Where no evidence exists, that is stated plainly rather than implied.

---

## ADR-001 to ADR-013: v1 detector (superseded)

**Status:** superseded. The v1 detector was removed from the tree. The code is at git tag `v1` and the full records are in `docs/history/v1_decisions.md`.

- ADR-001: Use uncertainty proxies, not a supervised probe or an LLM-judge verifier
- ADR-002: Score post-hoc from cached logits, not streaming during decode
- ADR-003: One adapter interface (`StandardizedResult`) over two provider SDKs
- ADR-004: Feature set: 3 shared + 2 local-only, redundancy left in
- ADR-005: Weighted sigmoid over fitted logistic regression or a single feature
- ADR-006: Config-as-single-source-of-truth for the risk threshold
- ADR-007: Free-form generation in eval, not scoring a supplied candidate answer
- ADR-008: SBERT cosine similarity for correctness grading
- ADR-009: Operating point HIGH = 0.30, and the recall it gives up
- ADR-010: SQLite + per-mode singleton risk services, not a stateless service
- ADR-011: Suppress-and-clarify, not hedging or a raw score
- ADR-012: TinyLlama-1.1B-Chat as the local-mode default
- ADR-013: One product, two detectors, ship both modes behind one interface

---

## ADR-014: Route by risk rather than judge correctness

**Status:** accepted

**Context:** The audit in `report1.md` found the v1 detection numbers contaminated by a grading
leak, and the clean signal at AUROC 0.64 [0.52, 0.77] on 82 rows. A signal that weak cannot decide
whether an answer is correct. It can still order answers by risk.

**Decision:** The product is a routing layer. Every output gets a score, a calibrated cutoff maps
the score to one of three lanes (return, escalate, review), and the reported metrics are the ones a
budget holder cares about: errors caught and residual error at a review budget, and error against
cost along a cheap-to-strong cascade.

**Alternatives rejected:**
- *Continue improving the detector*: more features and better weights were the v1 plan, and the
  audit showed the five-feature ensemble never beat mean logprob alone. No amount of ensembling
  turns an ordering signal into a verdict, and the interview is in three days.

**Consequences:** Lift at a 20 percent budget can be modest and still be useful. The cascade
ceiling is the strong model's own error rate, so conditional fix and break rates are always
reported beside the frontier.

**Revisit if:** a scorer reaches an AUROC where a fixed operating point has precision a reviewer
would accept without a budget.

**Evidence:** `report1.md` sections 1 and 2, `routing/evaluate.py`.

---

## ADR-015: Scorer is a replaceable interface with a logged cost

**Status:** accepted

**Context:** Mean logprob is free but weak, needs a provider that returns logprobs, and hides
localized errors in long answers. Other signals cost extra calls. The right scorer differs by
provider and by budget.

**Decision:** A scorer is any function that returns a higher-is-riskier float per output plus a
cost record measured from logged token counts. Six ship in three cost tiers: mean logprob, min-k
logprob, length (control), length-residualized logprob at zero extra calls; self-consistency over
three resamples; P(True) at one extra call. Every figure shows all of them and the gates flag any
scorer that does not beat length.

**Alternatives rejected:**
- *A single fixed score*: the v1 HRI was one number, and the audit found it was a rescaled mean
  logprob that tracked answer length. One score cannot be honest about its cost or its confound.

**Consequences:** The report is a lift-per-cost-tier table rather than one headline. Providers
without logprobs still get the text-only scorers.

**Revisit if:** a claim-level scorer arrives; it should slot into the same interface.

**Evidence:** `routing/scorers.py`, `routing/gates.py`, `config/pricing.json`.

---

## ADR-016: Calibration is a per-deployment function, never a shipped constant

**Status:** accepted

**Context:** The v1 thresholds (`HIGH = 0.30`, SBERT 0.35, the 0.46 intervention cutoff) were all
chosen on the rows the results were reported on. Logprob distributions also differ by model,
prompt style, and domain, so any constant is wrong somewhere.

**Decision:** No threshold lives in config. `routing/calibrate.py` takes a labelled calibration
CSV and writes cutoffs to `results/calibration.json` with the file hash and n. Budget cutoffs
flag exactly the top fraction on calibration; risk cutoffs come from Learn-Then-Test with a
Hoeffding bound and a required-n table beside them. `route` refuses to run without that file.

**Alternatives rejected:**
- *Thresholds in config*: the v1 approach. A constant cannot carry its own coverage, risk, or the
  data it came from, and it invites tuning on the reporting set.

**Consequences:** Every deployment must label a small sample first. The number in config is not
the product; the function that produces it is.

**Revisit if:** never for the constant. The drift monitor on the roadmap decides when to re-run
the function.

**Evidence:** `routing/calibrate.py`, `routing/cli.py::cmd_route`, `report1.md` F4.

---

## ADR-017: Correctness labels come from two blind judges with a consensus rule

**Status:** accepted

**Context:** The v1 labels came from SBERT cosine at a threshold picked after viewing the
histogram, and the audit's human relabel disagreed with SBERT on 18 of 82 rows. The audit rater
could also see the score. A lift curve on labels that are 20 percent wrong has a 20 percent noise
floor.

**Decision:** Two LLM judges (`gpt-4o` and `gpt-4.1`, both OpenAI because that is the only API available) each see only question, reference
answer, and candidate answer, and return CORRECT, INCORRECT, or AMBIGUOUS with one sentence. A
row is labelled only when both agree; any disagreement or AMBIGUOUS excludes the row from every
headline number, and the excluded count is always reported with agreement and kappa.

**Alternatives rejected:**
- *SBERT similarity*: measures phrasing overlap, not truth. Kept only inside the consistency
  scorer, where it measures agreement between the model's own answers.
- *Single-rater labels*: no way to measure label noise, and the rater saw the scores.

**Consequences:** Fewer usable rows, the labels are still not human, and both judges come from one provider, so they share failure modes. Human labels on 150 rows
are the first roadmap item.

**Revisit if:** human labels show kappa against the LLM consensus below 0.6.

**Evidence:** `routing/judge.py`, `routing/data.py::consensus`, `report1.md` F6 and E3.

---

## ADR-018: Budget policies use batch quantiles, risk policies use calibrated cutoffs

**Status:** accepted

**Context:** A cutoff fixed as a score quantile on the 160-row calibration split does not hold its
flag rate on another draw. Applied to the 300-row test split, the 20 percent calibration quantile
flags 30.2% [25.2, 35.7] of rows for `mean_logprob`, 30.9% [25.8, 36.5] for `min_k_logprob`, 28.5%
[23.6, 33.9] for `length_residual`, 25.7% [21.0, 31.0] for `consistency_k3` and 21.9% [17.5, 27.0]
for P(True) (reports/routing_report.md, Cutoff drift). A dataset team sets a review budget and needs
that budget delivered.

**Decision:** Two policy families, never mixed (`routing/policy.py`). A budget policy fixes the
flag rate and lets the score cutoff float: it flags exactly ceil(budget * n) rows of the batch
being routed by batch quantile, ties broken by qid, using only the batch's own scores and no
calibration or label input. A risk policy fixes the score cutoff from `results/calibration.json`
and lets the flag rate float. Every review-lift number uses the budget policy, and the bootstrap
re-applies the quantile inside each resample. `calibrate_budget()` survives only to report drift.

**Alternatives rejected:**
- *A single calibrated cutoff for both*: a cutoff fixed on one draw does not deliver a fixed flag
  rate on another, and a dataset team sets a budget, not a score.

**Consequences:** Budget routing needs the whole batch before it can flag anything. Streaming
needs a running quantile with a warmup window; that is on docs/ROADMAP.md and not built. The
`route` command takes `--policy` with no default and refuses `--cutoff` under the budget policy.

**Revisit if:** the drift monitor shows the calibration and serving distributions agree closely
enough that a fixed cutoff delivers its budget within the interval.

**Evidence:** `routing/policy.py`, `tests/test_policy.py`, `routing/evaluate.py::cutoff_drift`,
reports/routing_report.md (Cutoff drift).

---

## ADR-019: P(True) is scored as No-over-Yes log-odds with a mean_logprob tie term

**Status:** accepted

**Context:** A P(True) call returns the first-token top 5 logprobs. Two scorings are possible: one
minus the probability of "Yes", or the log-odds of "No" over "Yes". Scores are stored in the scores
CSV at 6 decimals, and the budget policy flags exactly ceil(budget × n) rows, so the scorer must
separate rows finely enough that the flagged set never depends on row order.

**Decision:** Score P(True) as log P(No) - log P(Yes). Case and space variants are pooled by
logsumexp. A token absent from the top 5 takes the lowest returned logprob, an upper bound on its
probability. Add 1e-7 times the `mean_logprob` risk value as a tie term, asserted below 3e-7 on every
row, and store the `p_true` column at 1e-10 so the term survives. Log-odds are rounded to 1e-6, so
the term separates exact ties and never reorders distinct values.

**Alternatives rejected:**
- *One minus P(Yes) stored at 6 decimals*: probabilities near 0 and 1 collapse to the same stored
  value, leaving 81 distinct values on the 288 labelled test rows and a tied group at the 0.20 cutoff.
- *A secondary sort key inside `flag_by_budget`*: it would change tie handling for every routing
  scorer, and the determinism contract covers all of them.

**Consequences:** All 288 test scores are distinct, and no tie straddles any budget cutoff. AUROC
is 0.817 [0.771, 0.861], against 0.799 [0.750, 0.847] for one minus P(Yes) at 6 decimals and
0.813 [0.766, 0.858] at full precision. Lift at the 0.20 budget is 1.79 [1.58, 2.03] under every
variant. Dropping the 19 rows where "No" was imputed gives AUROC 0.791 [0.732, 0.842] on 269 rows.

**Revisit if:** a provider returns more than 5 top logprobs, which would make imputation rare.

**Evidence:** `routing/scorers.py::p_true_logodds`, `tests/test_ptrue.py`,
`scripts/ptrue_variants.py`, `results/ptrue_variants.json`.

---

## ADR-020: The length control keeps qid tie-breaks

**Status:** accepted

**Context:** The `length` control is an integer token count with 57 distinct values on the 288
labelled test rows. The budget policy orders tied rows by qid. At the 0.10, 0.20 and 0.30 budgets
the tied group at the cutoff holds 5, 8 and 9 rows (`results/ptrue_variants.json`, `length_ties`).

**Decision:** Keep qid tie-breaks for `length` and record the counts. `length` is a control, not a
candidate. Its role is to show what a naive signal achieves.

**Alternatives rejected:**
- *A secondary tie key for `length`*: it adds a second signal to the control, which then no longer
  measures length alone.

**Consequences:** `length` lift carries a small dependence on row order. Work under `itemtriage/`
breaks every tie by a declared secondary key.

**Revisit if:** a routing result ever rests on `length` beating another scorer by a margin that the
tied rows could close.

**Evidence:** `scripts/ptrue_variants.py::straddling_ties`, `results/ptrue_variants.json`.

---

## ADR-021: itemtriage uses a small direct-letter panel with confidence, not a large reasoning panel

**Status:** accepted

**Context:** A dataset builder auditing items before release cannot run a 114-model panel. Land and
Bikel (arXiv 2605.30504) report that small-panel IRT precision at the top 100 falls from 98.0% with 114
models to 71.0% with 7, using answers only.

**Decision:** Seven models in five families, each answering with a single letter and no reasoning: two
API models with first-token logprobs, and five local models from 1.5B to 3.8B parameters on a 6 GB GPU.
Each model contributes its full letter distribution and, for the API models, a P(True) call. The study
asks whether that confidence recovers what the small panel loses.

**Alternatives rejected:**
- *A reasoning-prompted panel*: tens to hundreds of output tokens per item instead of one, which is the
  cost the question is trying to avoid.

**Consequences:** Results do not speak to reasoning-prompted panels. Correlated families (two OpenAI, two
Qwen) make the effective panel smaller than seven.

**Evidence:** `docs/PREREG.md`, `itemtriage/panel_api.py`, `itemtriage/panel_local.py`, `results/itemtriage/pilot.json`.

---

## ADR-022: Ground truth is MMLU-Redux expert annotation; the weak-label comparison is not run

**Status:** accepted

**Context:** The comparison with a published full-panel Δℓ needs a benchmark with weak labels for every
item. The Writer release holds 1,001 flagged items only, with 12 strict weak positives for GSM-MC and 9
for MATH-MC, and GPQA treated as all correct.

**Decision:** Evaluate only on MMLU-Redux 2.0 expert labels. The primary class is a ground-truth error
(`wrong_groundtruth`, `no_correct_answer`, `multiple_correct_answers`). `expert` items are unknown and
excluded. H1b is reported as not computable.

**Alternatives rejected:**
- *Use the Writer weak labels anyway*: 21 positives cannot carry an interval worth reporting, and the
  labels came from an aggregator reading strong-model arguments, which favours the full panel.

**Consequences:** One benchmark, a 3.5% test base rate, and a precision ceiling of about a third at a 10%
budget.

**Evidence:** `reports/phase1_findings.md`, `docs/related_work.md`.

---

## ADR-023: Splits are by subject, stratified by base rate, and frozen with a hash

**Status:** accepted

**Context:** Error rates differ widely by subject (virology is far above the rest), and identical items
recur within subjects.

**Decision:** Sort subjects by primary error rate, cut them into blocks of five, and assign two of each
block to calibration and three to test with a fixed seed. Validation is chosen the same way inside
calibration. The split file is written once with a sha256 and refuses to change. Item identity is
question plus options, and no identical item crosses folds.

**Alternatives rejected:**
- *A random item-level split*: the same question could sit in both folds, and per-subject error rates
  would leak into both.

**Consequences:** 17 calibration-fit, 6 validation and 34 test subjects, with 49, 15 and 117 primary
positives.

**Evidence:** `itemtriage/splits.py`, `data/itemtriage/splits/mmlu_redux.json`.

---

## ADR-024: Ranking ties are resolved in expectation, never by row order

**Status:** accepted

**Context:** Panel scores tie often. With seven models, `disagree` takes eight values. A budget cutoff
inside a tied block makes the flagged set depend on how rows happen to be ordered.

**Decision:** Every scorer has a declared secondary key. Rows still tied on both keys and straddling a
cutoff count their expected share of positives under uniform resolution. Average precision uses the
exact expectation, computed per positive.

**Alternatives rejected:**
- *Break ties by qid or by a seeded shuffle*: the result would then depend on an arbitrary order or on
  one seed's luck.

**Consequences:** Metrics are invariant to row permutation (`tests/itemtriage/test_rank_and_irt.py`).
Routing's older scorers keep qid tie-breaks (ADR-020).

**Evidence:** `itemtriage/rank.py`.

---

## ADR-025: Labels are readable only through an allowlisted loader

**Status:** accepted

**Context:** MMLU-Redux ships its annotations (`error_type`, `correct_answer`, `potential_reason`) in the
same rows as the questions.

**Decision:** Ingest splits each row into a label-free item table and a separate label table. Only
`itemtriage.labels.load` reads the label table, and it accepts only named callers: splits, evaluation,
calibration, the supervised tier, the audit and the human check. Every prompt passes a guard that checks
the options verbatim and rejects category identifiers and annotation text. A test asserts that the
scorer, prompt and panel modules never import the loader.

**Alternatives rejected:**
- *Keep labels in the item table and rely on discipline*: one stray column in a prompt or a feature
  silently invalidates every result.

**Consequences:** 17,100 rendered prompts were audited against 302 annotation strings with no violation
(`results/itemtriage/prompt_audit.json`).

**Evidence:** `itemtriage/labels.py`, `itemtriage/prompts.py`, `itemtriage/audit.py`.

---

## ADR-026: Local models read two option orders; API models read one

**Status:** accepted

**Context:** Small models have strong letter-position bias. In the pilot, the two orders gave the same
answer on 56.3% to 77.3% of items depending on the model.

**Decision:** Each local model reads the original order and one cyclic shift. Probabilities are mapped
back to the original options and averaged. API models read the original order only.

**Alternatives rejected:**
- *Permute API calls too*: it doubles the API spend for the two models with the least position bias.

**Consequences:** Local inference time doubles at no API cost. API letter bias remains, and is listed as a
threat in the preregistration.

**Evidence:** `itemtriage/prompts.py::permute`, `itemtriage/collect.py`, `results/itemtriage/pilot.json`.

---

## ADR-027: The preregistration is committed before any model output exists

**Status:** accepted

**Context:** Choices made after seeing outputs (which scorer, which IRT variant, which layers) would
inflate any reported difference.

**Decision:** `docs/PREREG.md` was committed before the first panel call. It fixes hypotheses, scorers,
tie keys, metrics and decision rules. The evaluator refuses to touch test unless that file is tracked and
unmodified. The first test evaluation writes `results/itemtriage/unblinding.json`, and any later run under
a different commit is flagged as a deviation.

**Alternatives rejected:**
- *Register only before test evaluation*: pilot and validation results could then shape the registered
  design.

**Consequences:** Anything the preregistration did not specify is reported as a clarification, such as
the out-of-fold probe scores in H2.

**Evidence:** `docs/PREREG.md`, `itemtriage/evaluate.py::prereg_commit`, `tests/itemtriage/test_guards.py`.

---

## ADR-028: The routing risk lane's fixed sequence is a known issue; itemtriage starts it at 10% coverage

**Status:** accepted

**Context:** `routing.calibrate.calibrate_risk` runs Learn-Then-Test as a fixed sequence over 200 score
quantiles, starting at the lowest score and stopping at the first threshold whose Hoeffding bound
exceeds alpha. For a score without ties at its minimum, the first threshold accepts one item, so its
bound, risk + sqrt(ln(1/δ) / 2), is above 1 at δ = 0.10 and the sequence stops there. On routing's calibration
split, four of the five scorers accept exactly one item at their first threshold, so for them the
function could not return a threshold whatever the error rate. `consistency_k3` accepts 31 tied items
there, which still puts the bound at least 0.19 above the observed risk. The function can only qualify
when many items tie at the lowest score. On itemtriage's calibration set the required-n table says alpha 0.05 at 90% coverage needs 1,509
items, fewer than the 2,288 available, yet the function returned nothing.

**Decision:** Routing's function stays as it is, because changing routing behaviour is out of scope, and
its reported risk-lane results are read as "not tested" rather than "no threshold exists". itemtriage
reports the preregistered call unchanged, and beside it a declared post-unblinding deviation that starts
the same fixed sequence at 10% coverage (`itemtriage.calibrate.ltt_from_coverage`). The coverage targets do
not depend on labels, so the sequence is still fixed in advance and FWER control holds.

**Alternatives rejected:**
- *Silently swap in the corrected sequence*: it would hide that the preregistered implementation could not
  qualify.

**Consequences:** The routing report's statement that the risk lane found no cutoff because of the base
rate is incomplete. For four of its five scorers the sequence could not have qualified at any base rate. Fixing routing means changing
`calibrate_risk` and regenerating the routing outputs, which is left as a separate change.

**Evidence:** `routing/calibrate.py::calibrate_risk`, `itemtriage/calibrate.py`, `results/itemtriage/calibration.json`.
