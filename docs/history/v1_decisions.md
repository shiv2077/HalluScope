# v1 architecture decision records (withdrawn)

These ADRs describe the v1 HRI detector. Its AUROC figures were withdrawn after the audit in `report1.md`. The code is at git tag `v1`.

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
