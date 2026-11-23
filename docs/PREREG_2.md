# itemtriage Study 2 preregistration: panel size and strength

Committed before any Study 2 metric is computed. Any later change to this file is a deviation and is
recorded with its reason in `results/itemtriage/study2.json` and in the report.

## Question

A team building a new private benchmark has no 100-model response matrix before release. How small and how
cheap can the audit panel be, and does panel strength matter more than panel size?

## What has already been seen

- **The MMLU-Redux test labels were seen in Study 1.** The test fold was unblinded on 2 October 2026
  (`results/itemtriage/unblinding.json`). Every Study 2 result on MMLU-Redux is therefore exploratory and
  post-unblinding, even where it has a decision rule below.
- **Platinum GSM8K is the blind confirmatory replication of H4.** Before this file was committed, only the
  platinum file's `cleaning_status` counts (32 `rejected`, 1 `revised`, 300 items) and the text join rate were
  seen. No scorer was evaluated against these labels.
- Before this file was committed, the response matrix was built and the model ability and agreement clusters
  below were computed. These use correctness against the key only, never a flaw label.

## Data

- **Response matrix.** The tinyBenchmarks release of the Open LLM Leaderboard v1 (`lb.pickle` at commit
  `ce9f7024`, sha256 `34f44d6a…6437`): binary correctness against the key for 395 models on all 14,042 MMLU
  rows (5-shot, log-likelihood scoring) and 1,319 GSM8K rows. MIT licence.
- **Join.** Leaderboard rows follow the v1 detail files. Row text from one reference model's detail files
  (`open-llm-leaderboard-old/details_alignment-handbook__zephyr-7b-sft-full`, revision `1325769d`) is parsed
  into question and options and matched to MMLU-Redux with the Study 1 item identity rule (NFKC, lowercase,
  collapsed whitespace). 5,690 of 5,700 items match. The 10 unmatched items are excluded. An item whose text
  appears twice in MMLU uses the first leaderboard row (34 items).
- **Second ground truth.** MadryLab platinum-bench GSM8K (revision `51920a33`, CC-BY-SA-4.0): 300 items, all
  matched to leaderboard rows by question text. Positive class: `rejected` or `revised` (33 items).
- Everything is recorded in `data/itemtriage/matrix/manifest.json`.

## Models

- **Ability.** Accuracy against the key on every MMLU row of the 17 calibration-fit subjects (4,818 rows). No
  test subject enters ability.
- **Model filter.** Models are clustered by pairwise agreement on the same rows (average linkage). One model
  per cluster at 95% agreement is kept, the first by name. That leaves 335 models. Beyond 95% agreement a second
  member adds almost no independent responses.
- **Lineages.** Clusters at 90% agreement (239 among the 335) define the lineage-diverse stratum.
- Ability ties are broken by model name.

## Panels

- Sizes k in {1, 3, 5, 7, 10, 20, 50, all}. "All" is the 335-model panel.
- **top-k:** the k most able models. **bottom-k:** the k least able.
- **random-k:** 50 draws of k models without replacement.
- **lineage-diverse random-k:** 50 draws. Each draws k distinct lineages, then one model uniformly within each.
  Not defined at k = all.
- One `numpy.random.RandomState(42)` draws every random panel in increasing k. A second `RandomState(42)` draws
  every lineage-diverse panel in increasing k.

## Scorers

All are label-free and higher means riskier. They are computed over all 5,690 matched items (all folds), as in
Study 1, with subjects as groups.

| scorer | definition | panels |
|---|---|---|
| `disagree` | share of the panel that misses the key | all |
| `strong3` | share of the panel's 3 most able models that miss the key. Equals `disagree` when k ≤ 3 | all |
| `item_total` | minus the item-total correlation within the subject (`irt.item_total_r`, Fantastic Bugs). An item with a constant response vector gets 0 | k ≥ 5 |
| `irt_small` | the 4PL forced-ceiling contrast Δℓ, exactly as in Study 1 (`irt.fit_stage1`, `irt.delta_full`) | k ≥ 5, see below |
| `subject_mean` | the subject mean of `disagree` | all |
| `length` | characters in question and options, as in Study 1 | panel-free |
| `conf` | mean over the panel of 1 − P(key) | only if per-choice log-likelihoods are fetched |
| `conf_w` | 1 − P(key) weighted by softmax of the stage-1 IRT ability, as in Study 1 | k ≥ 5, same condition |

- **IRT cost.** `irt_small` is fit on top-k and bottom-k for every k ≥ 5, on the all-model panel, and on the
  first 20 draws of each random and lineage-diverse cell for k in {5, 10, 20, 50}.
- **P(key).** It is the softmax of the four option log-likelihoods from the model's newest complete detail run.
  A model's P(key) is used only if that run agrees with the matrix on right versus wrong for at least 95% of
  items. `conf` and `conf_w` use the panel members that pass. Coverage is reported.
- **Ties.** No Study 2 scorer has a secondary key. Ties resolve in expectation (ADR-024). With k = 3, `disagree`
  takes 4 values, so a secondary scorer would decide much of the ranking and bring its own signal into the
  size comparison.

## Metrics

- **Primary:** precision of the flagged slice at a 10% budget, primary class, MMLU-Redux test fold, matched
  items with a known label.
- **Secondary:**
  - precision at 5% and 20%
  - the secondary class at 10%
  - virology excluded at 10%
  - average precision
  - **within-subject average precision:** `rank.average_precision` inside each test subject, averaged over the
    test subjects with at least one positive. Its bootstrap resamples items within each subject, so the subject
    set stays fixed. A subject left with no positive in a resample is skipped, and the skips are counted. It
    separates item-level from subject-level signal. It is not defined on GSM8K.
- **Intervals.** Nested bootstrap:
  - Outer: 2,000 item resamples with seed 42, the Study 1 indices (`evaluate.boot_indices`).
  - Inner: the R draws of a cell are resampled with replacement in each outer replicate, from a separate
    `RandomState(42)`, and the statistic is their mean.
  - The point estimate is the mean over draws on the full item set.
  - Differences are paired: both sides use the same outer resample.

## Hypotheses and decision rules

All use `disagree`, because it exists for every panel without log-likelihoods.

- **H4, strength over size.** Precision at 10%: top-3 minus random-10 (draw mean).
  - "Holds" if the paired interval lies entirely above 0. "Reversed" if it lies entirely below 0. Otherwise
    "inconclusive".
  - It is evaluated twice:
    - **On platinum GSM8K, confirmatory.** Panels are chosen by MMLU ability. Scorers are computed on all
      1,319 GSM8K rows and evaluated on the 300 platinum items, so 30 items are flagged at 10%. Power is low,
      and "inconclusive" is a likely outcome.
    - **On MMLU-Redux, exploratory.**
  - Every other GSM8K number is secondary.
  - Prior: Land and Bikel (arXiv 2605.30504) report P@100 of 71.0% with 7 ability-sorted models, against weak
    labels.
- **H5, minimum panel.** On the random-k curve, report the smallest k whose interval lower bound reaches 80% of
  the all-model panel's point estimate, at that k and at every larger k in the grid. Report k with the curve's
  interval at k, or "none in range". No pass or fail.
- **H6, scorer by size.** For `irt_small` and for `item_total`, on the random-k cells with k ≥ 5 and the all-model
  panel, the paired difference against `disagree` on the same draws. Report the smallest k whose lower bound is
  above 0, or "no crossing in range".

## Cost axis

- Leaderboard models: total parameters of the panel, from model-card safetensors metadata (174 of 395 models
  known). This is a labelled proxy and never converted to dollars.
- The five local Study 1 models: a measured point, with GPU-seconds per 1,000 items if the raw logs carry
  timestamps.
- API models: dollars per 1,000 items from logged tokens. There are none unless a key becomes available.

## Exploratory, no decision rule

- Memorisation by ability decile: agreement with the key on flawed and on sound items, with Wilson intervals,
  on calibration and validation items, and on test items labelled post-unblinding.
- The five-model local panel from Study 1 as a measured point on the same axes. Its regime differs (zero-shot,
  direct letters), so it is never pooled with the leaderboard curves.

## Threats

- Leaderboard models share backbones. Across the 395 models one ability factor dominates the correlation
  matrix (participation ratio 5.4 on the calibration-fit rows), so the effective number of independent models
  is far below the panel size.
- Many top leaderboard entries are merges tuned on the leaderboard itself. A strong panel may have memorised
  MMLU keys, including wrong ones. That would push H4 toward "reversed".
- The leaderboard regime (5-shot, log-likelihood) differs from Study 1's panel (zero-shot, letters).
- MMLU-Redux labels carry annotator noise. Platinum GSM8K is small.
