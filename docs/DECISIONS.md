# HalluScope — Architecture Decision Records

Format per entry: Context (the forcing constraint) → Decision → Alternatives rejected (with the
reason they lost under these constraints) → Consequences (including the downside) → Revisit if →
Evidence. Where no evidence exists, that is stated plainly rather than implied.

---

## ADR-001: Use uncertainty proxies, not a supervised probe or an LLM-judge verifier

**Status:** accepted

**Context:** No labelled in-domain hallucination data exists for this project — no per-domain
corpus of true/false statements, no human-graded correctness at serving time.

**Decision:** Score risk from the generating model's own token-level uncertainty (logprobs,
logit flatness, representation collapse) — signals available with zero labelled data, computed
directly from what the model already outputs.

**Alternatives rejected:**
- *Supervised hidden-state probe (SAPLMA-style)*: needs a labelled corpus of true/false
  statements to train a classifier on hidden states. Building that corpus is the bottleneck this
  project exists to work around. Also tied to the specific checkpoint it was trained on —
  brittle across model swaps.
- *LLM-judge verifier*: doubles API cost per turn, adds a second model's own hallucination risk
  to the fact-checking step, and gives nothing in local mode without a second model loaded.

**Consequences:** HRI can only ever proxy uncertainty, not truth. The [known blind
spot](../README.md#known-blind-spot) — confident fabrication — is a direct structural
consequence of this decision, not an unrelated gap.

**Revisit if:** in-domain labelled data becomes available (e.g. a production feedback loop) —
a probe could then be fit and blended in as an additional feature.

**Evidence:** `config/weights.json`, `backend/app/risk/hri_engine.py` — no labelled-data
dependency anywhere in the scoring path.

---

## ADR-002: Score post-hoc from cached logits, not streaming during decode

**Status:** accepted

**Context:** Several signals need the *complete* answer before they're defined:
`confidence_decay` compares the first-N and last-N token logprobs; RCD walks the full hidden-state
sequence. FastAPI request/response is single-shot, not a token stream, in the current UI.

**Decision:** Capture the full per-token trajectory during generation
(`token_logprobs`, `hidden_states_per_token`, `logits_per_token`), then compute every signal and
classify once, after generation finishes and before the response is returned.

**Alternatives rejected:**
- *Token-by-token scoring*: `confidence_decay` and RCD are undefined on a partial sequence;
  would need different formulas or a discontinuous "provisional" score.
- *Early-exit on rising per-token risk*: would require validated sub-answer hallucination
  detection, which does not exist here, and would produce truncated partial answers — worse UX
  than the current suppress-and-clarify design.

**Consequences:** No latency benefit from early abstention — full generation cost is paid even
for answers that get suppressed. Not compatible with a token-streaming UI without redesign.

**Revisit if:** the UI needs live token streaming, or generation cost dominates enough that an
early-exit accuracy tradeoff becomes worth making.

**Evidence:** `backend/app/services/risk_service.py::analyze_message` runs entirely after
`llm.send_message` returns.

---

## ADR-003: One adapter interface (`StandardizedResult`) over two provider SDKs

**Status:** accepted

**Context:** OpenAI Chat Completions exposes only text + top-5 logprobs — no raw logits, no
hidden states, ever. The local HuggingFace adapter can expose both. There is no way to get API
mode to parity with local mode's internals.

**Decision:** `BaseLLMAdapter.generate_chat()` returns one `StandardizedResult` dataclass for
both adapters; local-only fields (`logits_per_token`, `hidden_states_per_token`,
`per_layer_hidden_per_token`, `unembed`) are `Optional` and `None` from the OpenAI adapter.

**Alternatives rejected:**
- *Two separate pipelines per mode*: would duplicate `risk_service.py`, `hri_engine.py`, and the
  frontend, defeating the point of a shared reliability engine.
- *Force API-mode's 3-feature set on local mode too*: throws away the entire reason local mode
  exists.

**Consequences:** HalluScope is really **two detectors sharing one interface and one name** — the
3-feature API-mode HRI and the 5-feature local-mode HRI use different weight blocks and are not
the same calibrated score. The README's results table already keeps them as separate rows for
this reason.

**Revisit if:** a hosted provider starts exposing full logprobs/logits, converging the two
feature sets.

**Evidence:** `backend/app/adapters/schemas.py`, `backend/app/adapters/base_adapter.py`.

---

## ADR-004: Feature set — 3 shared + 2 local-only, redundancy left in

**Status:** accepted

**Context:** Both modes expose per-token logprobs; only local mode exposes full logits and
hidden states.

**Decision:** `logprob_variance`, `confidence_decay`, `abs_mean_logprob` shared; `logit_flatness`
(formerly "semantic energy") and `rcd_inv` local-only.

**Alternatives rejected:**
- *Top-K truncated entropy as a 4th shared feature*: computed and returned in
  `metrics.entropy`/`normalized_entropy`, deliberately excluded from `compute_hri` — top-5
  truncated entropy is a weak proxy for the real distribution shape, and largely redundant with
  `abs_mean_logprob` + `logprob_variance`.
- *Folding Stage 1 (prompt risk) / Stage 3 (semantic checks) into HRI*: implemented, computed on
  every turn, returned in the response — but not fed to `compute_hri`; no validated weight for
  them exists (see README's Experimental section).

**Consequences:** `logprob_variance` and `confidence_decay` are both derived from the same
logprob sequence and likely correlated; the heuristic weights were hand-set, not fit for
orthogonality. `eval/ablation.py` can measure this (leave-one-out AUROC) but has not been run to
a committed result.

**Revisit if:** `eval/ablation.py` is run and shows near-zero marginal AUROC for a shared feature.

**Evidence:** `backend/app/risk/hri_engine.py` (`FEATURE_NAMES_API`/`FEATURE_NAMES_LOCAL`),
`eval/ablation.py`.

---

## ADR-005: Weighted sigmoid over fitted logistic regression or a single feature

**Status:** accepted

**Context:** `config/weights.json` explicitly documents itself as `"source":
"heuristic-default"` — hand-set weights. `eval/fit_weights.py` exists to overwrite the file with
logistic-regression-fitted weights, and `hri_from_features` consumes either transparently.

**Decision:** ship the hand-set heuristic weights as the default:
`sigmoid(scale · Σ w_i f_i + bias)`.

**Alternatives rejected:**
- *Single best feature alone*: simpler, possibly close on `eval/ablation.py`'s single-signal
  mode, but throws away the ability to combine local-only signals and drops the shared structure
  across modes.
- *Ship `fit_weights.py`'s logistic-regression output as default*: would mean fitting on the
  n=100 calibration set and baking a low-n overfit into the shipped default — the same
  small-sample risk already flagged for the thresholds.

**Consequences:** **the sigmoid is a monotone transform of a linear score — it does not change
ranking or AUROC.** It bounds the score to `[0,1]` for display. Absent a Platt/isotonic fit
against real labels, HRI is a *score*, not a calibrated probability. "HRI = 0.72" does not mean
"72% chance of hallucination."

**Revisit if:** `fit_weights.py` is run on a proper held-out split and beats the heuristic
weights by a meaningful AUROC margin.

**Evidence:** `config/weights.json` (`_doc` field), `eval/fit_weights.py`,
`backend/app/risk/hri_engine.py::hri_from_features`.

---

## ADR-006: Config-as-single-source-of-truth for the risk threshold

**Status:** accepted

**Context:** `eval/REPORT.md` documents a historical bug: the HIGH threshold existed as separate
literals in different places and drifted (0.30 / 0.46 / 0.086 disagreeing with each other).

**Decision:** `config/thresholds.json` owns `risk_low`/`risk_high`; both
`hri_engine.classify_risk` and `intervention_service.should_intervene` read the *same*
`risk_high` through `config_store`, so "classified HIGH" and "intervention fired" are identical
by construction, not by convention.

**Alternatives rejected:**
- *Separate `INTERVENTION_THRESHOLD` constant equal to the HIGH cutoff*: exactly the bug being
  fixed — two literals that must be kept in sync manually will eventually drift, and did.
- *Threshold passed as a function parameter at each call site*: still lets a caller pass the
  wrong value; moves the two-sources problem, doesn't remove it.

**Consequences:** this is a genuine correctness invariant, with a real cost: you cannot display a
different risk_level than what triggers suppression without adding a second config key
(not currently supported) — experimentation on display-only thresholds isn't possible today.

**Revisit if:** product wants to A/B test displayed risk level independent of intervention
behavior.

**Evidence:** `backend/app/config_store.py`, `config/thresholds.json` (`_doc` field),
`backend/app/services/intervention_service.py`.

---

## ADR-007: Free-form generation in eval, not scoring a supplied candidate answer

**Status:** accepted

**Context:** HaluEval's default protocol supplies a candidate answer (dataset-correct or
dataset-hallucinated) and asks the system to judge it. This was tried first and the result is
preserved in `eval/results_api_v1_candidate_eval.csv`.

**Decision:** the harness sends only the question, captures the model's own free-form answer,
and grades *that* answer against the reference — HRI scores generation, not verdict.

**Alternatives rejected:**
- *Score the supplied candidate directly (HaluEval default)*: measured failure, not a
  hypothetical — HaluEval's correct answers are short fragments and hallucinated answers are
  full fluent sentences, so the model's uncertainty when judging a fragment vs. a sentence
  tracked the candidate's surface fluency, not its truth (F1 = 0.349, mean HRI inverted:
  correct > hallucinated).

**Consequences:** this changes what's measured — from "is HRI sensitive to candidate fluency" to
"does HRI correlate with the model's own ability to answer correctly," the actually-relevant
question for real deployment. It also means these numbers are **not directly comparable** to
published HaluEval-QA leaderboard entries using the standard protocol.

**Revisit if:** comparing against published HaluEval baselines becomes a goal — would need to
also report the standard-protocol number with its confound disclosed.

**Evidence:** `eval/REPORT.md` (Methodology), `eval/results_api_v1_candidate_eval.csv`.

---

## ADR-008: SBERT cosine similarity for correctness grading

**Status:** accepted

**Context:** No human-labelled correctness exists for the 100-question calibration set; an
automatic match/mismatch label is needed for AUROC.

**Decision:** cosine similarity (`all-MiniLM-L6-v2`) between the model answer and the dataset
`reference_answer`; threshold 0.35, picked from the bimodal histogram trough on this run and
spot-checked on 24 gray-zone rows.

**Alternatives rejected:**
- *Exact-match / substring against the reference*: HalluEval references are short fragments,
  model answers are full sentences — near-guaranteed false negatives on every correct free-form
  answer, the same fluency mismatch that broke the v1 methodology (ADR-007).
- *LLM-as-judge grading*: introduces a second model's hallucination/inconsistency risk into the
  grading step of a hallucination-detection benchmark — circular.

**Consequences:** **known weakness on short-form QA.** SBERT cosine is a semantic-similarity
proxy, not a correctness judgment — `eval/REPORT.md` itself reports ~5–7 of 100 rows sit in a
genuinely ambiguous 0.20–0.35 band. The 0.35 cutoff was chosen after seeing this run's
distribution, not blind — a mild version of the "don't tune on what you evaluate" risk flagged
elsewhere in this doc.

**Revisit if:** `eval/grader_calibration.py` / `grader_sensitivity.py` is run against
`eval/gold/` (two independent human-annotator CSVs already exist) to validate 0.35 against real
labels instead of a histogram shape.

**Evidence:** `config/grader.json`, `eval/match.py`, `eval/gold/gold_annotatorA.csv`,
`eval/gold/gold_annotatorB.csv`.

---

## ADR-009: Operating point HIGH = 0.30, and the recall it gives up

**Status:** accepted

**Context:** `eval/pick_thresholds.py` swept thresholds on the n=100 API calibration set under
precision ≥ 0.85, recall ≥ 0.50, flag-rate ≤ 0.50. No threshold satisfied all three at once
(`eval/operating_point.json`: `"n_qualifying": 0, "used_fallback": true`).

**Decision:** fall back to the closest defensible point near the constraint corner:
`HIGH = 0.30` (precision 0.879, recall 0.468, flag-rate 33%).

**Alternatives rejected:**
- *Best-F1 threshold (0.086)*: mathematically the strongest single number (F1 = 0.809), but
  flags ~80% of normal traffic on the calibration data, and breaks the LOW/HIGH ordering outright
  — `LOW = max(0.10, HIGH−0.15)` evaluates to `LOW = 0.10 > HIGH = 0.086`, an operational bug,
  not a UX tradeoff.
- *Pre-calibration heuristic (0.46)*: very conservative (precision 1.00) but recall 0.29 — misses
  most hallucinations, the opposite failure mode from what a reliability layer should default to.

**Consequences:** recall of 0.468 means roughly half of actual hallucinations in the calibration
distribution are **not** flagged and pass through unintervened — a real, accepted false-negative
rate, the price of keeping flag-rate low enough for a usable demo.

**Revisit if:** a larger held-out calibration set narrows the current wide confidence intervals,
or the product's flag-rate tolerance changes.

**Evidence:** `eval/operating_point.json`, `eval/pick_thresholds.py`, `eval/REPORT.md`
(Operating Point Selection).

---

## ADR-010: SQLite + per-mode singleton risk services, not a stateless service

**Status:** accepted — scoped to single-user demo

**Context:** Single-machine interview demo. `backend/app/api/chat.py` holds two `RiskService`
instances (`api`, `local`) as module-level globals, each with its own in-process conversation
history. SQLite (`storage.py`) persists only per-message analysis results, not conversation
state.

**Decision:** keep conversation history in-process, one singleton pair per server process;
persist analysis results to SQLite so `/metrics/{id}` survives a restart.

**Alternatives rejected:**
- *Per-session state keyed by session/user ID*: the correct fix for multi-user correctness, not
  built because there is exactly one user in this deployment target and no current consumer of
  session isolation.
- *Stateless service (client resends full history each turn)*: removes the singleton problem
  entirely, but requires the frontend to manage and resend growing conversation state — a larger
  frontend change than this build's scope.

**Consequences:** two concurrent users in different browser tabs **share the same two
`RiskService` singletons** — user A's messages become part of user B's conversation context.
This breaks under any real concurrency; SQLite's per-request open/use/close connection pattern
also becomes a contention point under concurrent writers.

**Revisit if:** a second concurrent user needs to use the deployed demo.

**Evidence:** `backend/app/api/chat.py` (`_risk_services` module dict), `backend/app/storage.py`.

---

## ADR-011: Suppress-and-clarify, not hedging or a raw score

**Status:** accepted

**Context:** `intervention_service.decide_answer` — when HRI ≥ `risk_high` and
`apply_intervention` is true, the real model answer is discarded and replaced by a fixed
clarification template (`generate_clarification`).

**Decision:** full suppression + deterministic clarification, not a hedged version of the
original answer and not a raw score left for the user to interpret.

**Alternatives rejected:**
- *Hedge the existing answer ("I'm not fully sure, but...")*: keeps the possibly-wrong content in
  front of the user, who may anchor on it regardless of the disclaimer — doesn't remove the risk.
- *Surface only the HRI score/badge, let the user judge*: assumes every user understands what an
  uncertainty score means and will act on it correctly; defeats the purpose of an automated
  intervention.

**Consequences:** false-positive suppressions (HIGH-flagged answers that were actually correct —
real, given precision 0.879, so roughly 1 in 8 flagged answers) show the user nothing of the real
answer, only a generic clarification. That usability cost is accepted in exchange for never
leaking a suppressed answer's content.

**Revisit if:** false-positive rate on a larger calibration set is high enough that hedging (a
middle option) becomes worth the added complexity.

**Evidence:** `backend/app/services/intervention_service.py`.

---

## ADR-012: TinyLlama-1.1B-Chat as the local-mode default

**Status:** accepted

**Context:** `local_adapter.py` loads the model with no quantization
(`AutoModelForCausalLM.from_pretrained(target)`, no `torch_dtype`/4-bit path) — it must fit in
float32 on whatever device is available. Target hardware is a single consumer laptop GPU (RTX
3060 class, 6GB VRAM), which a fp32 3B+ model does not comfortably fit.

**Decision:** default `LOCAL_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0`; overridable via env var
to any `AutoModelForCausalLM`-compatible repo (README documents `Qwen/Qwen2.5-3B-Instruct` as an
example override).

**Alternatives rejected:**
- *A stronger fp32 default (e.g. Qwen2.5-3B-Instruct, ~12GB in fp32)*: would not load on the
  class of GPU this was built against without 4-bit quantization (`bitsandbytes`) — a dependency
  and dtype path this build does not have.
- *Add quantization to unlock a bigger default*: real engineering work (dtype handling, a new
  dependency, likely accuracy impact on `logit_flatness`) not undertaken in this pass.

**Consequences:** **local-mode result quality is known to be bad.** `eval/REPORT.md` reports
TinyLlama answering correctly on only 3 of 100 HaluEval questions — a statement about the
model's competence, not the HRI signal's validity, but local-mode AUROC = 0.918 cannot be read
as "local mode detects hallucination better than API mode."

**Revisit if:** quantization support is added and a 3B+ class model becomes the default.

**Evidence:** `backend/app/adapters/local_adapter.py`, `backend/app/config.py`
(`DEFAULT_LOCAL_MODEL`), `eval/REPORT.md`.

---

## ADR-013: One product, two detectors — ship both modes behind one interface

**Status:** accepted, with the framing conceded

**Context:** API mode (3 features, OpenAI-only) and local mode (5 features, any local HF causal
LM) sit behind one chat UI toggle, sharing one HRI formula shape, one config directory, and one
intervention threshold key.

**Decision:** ship both under one product surface rather than splitting into two tools or repos.

**Alternatives rejected:**
- *API mode only*: loses the entire local-mode feature set (`logit_flatness`, `rcd_inv`) and the
  demonstration that internal-signal access adds detection power — a core part of the point.
- *Local mode only*: API mode is the realistic deployment target for most teams (few run their
  own inference), and is the only no-GPU path to demo the concept at all.

**Consequences:** **conceding the point** — under scrutiny, "one product" overstates it. It is
one interface serving two independently-weighted formulas with different result distributions
(ADR-003) and a UI toggle that quietly changes what's being measured. Defensible as "the honest
way to demo both without duplicating the engine"; not as "a single calibrated detector."

**Revisit if:** the two feature sets converge (ADR-003), or user research shows the toggle
confuses more than it clarifies.

**Evidence:** `frontend/src/pages/ChatPage.jsx` (mode picker), `config/weights.json` (separate
`api`/`local` blocks), README.md Results table.

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
needs a running quantile with a warmup window; that is on ROADMAP.md and not built. The
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
