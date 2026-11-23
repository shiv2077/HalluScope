# Architecture

itemtriage decides which benchmark items an expert should read first. It never changes a key itself. This
document explains the design and the reasoning behind each choice. Numbers live in the generated reports. The
diagram is `docs/itemtriage_architecture.excalidraw`, made by `python3 scripts/make_itemtriage_diagram.py`.

## Why rank instead of fixing keys

A model that disagrees with a key is evidence, not a verdict. The key may be wrong, or the item may be hard,
ambiguous, or out of the model's knowledge. Replacing keys automatically would turn every model error into a
dataset error. So the output is an ordering for a fixed review budget, and the expert keeps the decision. The
budget is the unit dataset QA actually spends: hours of expert time. Every metric is therefore
precision and recall of the flagged slice at 5, 10 and 20% budgets, not AUROC.

## Data flow

1. **Ingest.** MMLU-Redux 2.0 becomes an items table (question, options, key) and a separate labels table.
   Items get an identity hash from normalised question plus options (`itemtriage/ingest.py`).
2. **Split.** Subjects are split into calibration-fit, validation and test, stratified by base rate and frozen
   with a sha256 (`itemtriage/splits.py`). Splitting by subject means a test result is about unseen subjects.
3. **Responses.** Three sources give per-model answers on every item:
   - the own panel: five local models answering zero-shot by letter, in two option orders, on a 6 GB GPU
     (`itemtriage/panel_local.py` under the lock in `itemtriage/gpu.py`)
   - the API panel (`itemtriage/panel_api.py`), which never ran because the key was invalid
   - a public matrix: 395 Open LLM Leaderboard models, joined to MMLU-Redux by item text
     (`itemtriage/matrix.py`)
4. **Panels.** Study 2 draws panels from the public matrix by size and strength (`itemtriage/panels.py`).
5. **Scores.** Label-free scorers turn a panel's responses into one risk score per item
   (`itemtriage/scorers.py`, `itemtriage/panels.py`, `itemtriage/irt.py`).
6. **Rank.** Items are ranked by score. Ties resolve in expectation, never by row order (`itemtriage/rank.py`).
7. **Evaluate.** Labels join here, and only here (`itemtriage/evaluate.py`, `explore.py`, `study2.py`).
8. **Report.** `itemtriage/report.py` writes `reports/itemtriage_report.md` and the README results block from
   the results JSON. CI regenerates both and fails on any byte difference.

## Keeping labels out of the scoring path

- `itemtriage/labels.py` is the only reader of the labels files. Its allowlist names every module that may
  call it. Scorers, prompts, panels and panel runners are not on it. A test checks this.
- Prompts are rendered and checked before any model call: options verbatim, no category names, no annotation
  text (`itemtriage/audit.py`).
- Study 1 was preregistered in `docs/PREREG.md` before any model output existed (ADR-027). The first test
  evaluation wrote `results/itemtriage/unblinding.json` with the PREREG commit and a timestamp, and the
  evaluator refuses to run if PREREG is modified.
- After the test fold was unblinded on 2 October 2026, every new analysis on it is labelled exploratory,
  post-unblinding. Study 2 was preregistered in `docs/PREREG_2.md` before any Study 2 metric existed. Its
  confirmatory test runs on a second benchmark whose labels had not been evaluated (platinum GSM8K).

## Difficulty is not a wrong key

The cheapest scorer, `disagree`, is the share of the panel that misses the key. A wrongly keyed item makes every
model miss. So does a hard item. Correctness alone cannot tell them apart, so on its own `disagree` mostly
measures difficulty.

The two cases separate on who misses and how:

- **On a hard item**, weak models miss and strong models get it right. Misses spread across the wrong options.
- **On a wrong key**, strong models miss too, because they pick the true answer. They converge on one other
  option, and they are confident about it.

Every scorer that does better than difficulty uses that contrast:

- `conf_w` weights each model's probability mass off the key by its IRT ability, so strong models count more.
- `irt_small` (Δℓ) asks whether the item fits better if even the strongest models are allowed to fail it. That
  is a forced ceiling, and an item that only fits with a ceiling below 1 is suspect.
- `strong3` asks only the panel's three strongest models.

All of these need strong models in the panel. A panel of weak models can only report that an item is hard.
That is why Study 2 measures panel strength separately from panel size.

The contrast breaks in one known way. A strong model that memorised the wrong key from pretraining agrees with
it confidently, and then looks like evidence that the key is right. The memorisation tables in the report
measure this: agreement with the key on flawed items against sound items, per model and per ability decile.

## Bad items or bad subjects

Splits are by subject, and most errors sit in a few subjects. A score that only recognises bad subjects would
look useful. `itemtriage/explore.py` adds two label-free controls:

- `subject_mean`, which gives every item its subject's mean score
- `within_subject`, which is the item's score minus that mean

`subject_mean` ranks hard subjects, not flawed ones, so the panel signal is not a subject effect. The
within-subject residual is no better than the length control, so it is not a clean item effect either. Most
of Study 1's precision comes from virology. See the exploratory section of the report and ADR-030.

## Why IRT, and why it needs many models

IRT models each model's ability and each item's difficulty and discrimination together. With enough models
spread in ability, an item whose response pattern contradicts the ability ordering stands out: strong models
fail it while weak ones pass. That is the signal of a wrong key. With five models, stage 1 keeps only items
whose item-total correlation exceeds 0.2, and each per-item fit has five data points for three free
parameters. The fit is close to unidentified. In Study 1, `irt_small` ranked well below plain disagreement
(H1 table in the report). Study 2's H6 found no panel size in range where IRT or item-total correlation
overtakes disagreement. Both trail it clearly up to 20 models and only draw level at the full 335-model panel.
On these benchmarks, IRT is a large-panel tool.

## Using it on a new private benchmark

A team that builds a benchmark has no leaderboard matrix before release. The procedure is:

1. Pick a panel by strength first. On MMLU-Redux (exploratory), the strongest panels had higher precision than
   random ones at every size, and three strong models did as well as ten random ones. Weak panels did about as
   well as the item-length control. Two strong API models scored by confidence were the best measured point, at
   well under a dollar per 1,000 items (exploratory). Add models for coverage only after the strong ones are in.
2. Run each model once on every item and record its per-option probabilities. Cost is one forward pass per
   item and model.
3. Score, rank, and send the top slice of the budget to experts. Track precision per batch, because it varies
   with the base rate.
4. Keep the labels the experts produce. They become the calibration set for a cutoff with a guarantee, once
   there are enough of them (see the required-n tables).

## Routing

`routing/` applies the same ranking idea to model-generated answers: return, escalate to a stronger model,
or send to a human. It is engineering on top of known conformal methods, and the related work lists them. Its
risk lane at n=160 returns no threshold. ADR-028 shows the implementation's fixed sequence could not qualify
for four of its five scorers on any data, so the lane is untested rather than refuted.

## History

v1 was a hallucination detector whose AUROCs were withdrawn after an audit found the benchmark graded the
intervention template instead of the answer, and thresholds were set on the reported rows. The audit is
`docs/history/report1.md`. The discipline above (preregistration, label isolation, generated numbers,
determinism in CI) is the response to that audit.
