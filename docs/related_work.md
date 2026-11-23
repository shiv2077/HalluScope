# itemtriage: related work

itemtriage ranks multiple-choice items (question plus answer key) for expert review under a fixed budget,
using a panel of models whose answers and option probabilities are turned into label-free scores. Ground
truth is MMLU-Redux 2.0's expert annotation, with platinum GSM8K as a second ground truth in Study 2. The
first rows were read on 2 October 2026. The rows on Fantastic Bugs, Automated Benchmark Auditing,
HLE-Verified, the Epoch review and routing were read from their abstracts or pages on 24 November 2026.

| Work | What it does | Data | Labels | How itemtriage differs |
|---|---|---|---|---|
| Land and Bikel, "Auditing LLM Benchmarks with Item Response Theory", arXiv 2605.30504, EMNLP 2026 | 4PL IRT with a forced-ceiling contrast Δℓ = ℓ(d=0) − ℓ(d=1) per item, on binary correctness against the key. Two-stage fit: a constrained 3PL on items with item-total r > 0.2 sets θ, then per-item fits with a in [1/15, 15], b in [-2, 2], anchor penalty λ = 10. | 20,986 items in 32 subsets: preference benchmarks plus GPQA Diamond (165), GSM-MC (1,317), MATH-MC levels 1 to 5 | Weak labels from GPT-5.4 acting as an aggregator over strong-model arguments | Uses a 114-model panel and answers only. Its ablation reports P@100 of 71.0% with 7 models, ability-sorted subsamples. itemtriage asks whether confidence signals let 5 to 8 models recover that loss, against expert labels instead of weak ones. |
| Gema et al., "Are We Done with MMLU?", NAACL 2025 | Builds MMLU-Redux and its error taxonomy. Reports 6.49% of MMLU items erroneous, and 57% in Virology. | 5,700 MMLU items, 100 per subject | Expert re-annotation into ok, bad_question_clarity, bad_options_clarity, no_correct_answer, multiple_correct_answers, wrong_groundtruth, expert | Provides the ground truth. It does not test a small-panel detector at a review budget. |
| Song and Park, "Auditing MCQA Benchmarks through Probability Landscapes", arXiv 2608.30372, EMNLP 2026 | Top-option probability, normalized residual entropy and noise injection on option distributions from 3 open models (7B to 9B), with 3 larger models as a supplement | 1,000 items each from MMLU, ARC, HellaSwag and CSQA | 373 items overlap MMLU-Redux, 9 of them flawed. Noise injection flags 2 at precision 1.000 and recall 0.222. | The closest work. It has no disagreement, IRT or Confident Learning baseline, no review-budget metrics, and 9 positives. itemtriage uses 181 primary positives across 5,668 known items, with paired comparisons at fixed budgets. |
| Nahum et al., "Are LLMs Better than Reported? Detecting Label Errors and Mitigating Their Effect on Model Performance", EMNLP 2025 | An ensemble of 4 LLMs with 4 prompts. Averages class probabilities and flags confident disagreement with the gold label. Precision rises with confidence: above 95% confidence, more than two thirds of flags are real errors. | TRUE factual-consistency datasets and SummEval | Expert and crowd re-annotation | The same idea, mean ensemble confidence against the key, applied to binary and ordinal labels. itemtriage applies it to multiple-choice keys, weights models by IRT ability, and compares it with small-panel IRT under budgets. |
| Northcutt et al., Confident Learning, JAIR 2021, and "Pervasive Label Errors in Test Sets", NeurIPS 2021 | Flags examples whose out-of-sample predicted probability for the given label falls below a per-class threshold | Image and text classification test sets | Crowd verification of flagged items | Assumes a trained classifier with calibrated out-of-sample probabilities over a fixed label set. itemtriage has no training: each item has its own options, and the probability of the key comes from zero-shot panel models. `conf` is the panel analogue of self-confidence. |
| Klie et al., "Annotation Error Detection", Computational Linguistics 2023 | Survey and benchmark of 18 annotation-error-detection methods across tasks | Several NLP datasets with injected and real errors | Mixed | Frames the method families (uncertainty, ensembles, training dynamics). itemtriage sits in the ensemble family, without fine-tuning. |
| Vendrow et al., "Do Large Language Model Benchmarks Test Reliability?", 2025 | Manually revises 15 benchmarks into "platinum" versions with label errors removed | 15 public benchmarks | Manual review | Shows the cost of full manual review. itemtriage targets the review budget: which items an expert should read first. |
| "Task Verification Bench" (ICML 2026, as named in the brief) | Not located. The closest match is "Beyond Blind Compliance: Benchmarking Task Verification in OCR Reasoning", arXiv 2609.00232, an OCR task-verification benchmark. | | | The brief's figure, that LLMs overestimate broken-task detection by up to 54 points, could not be verified and is not used. |
| Truong, Tu, Hardy et al., "Fantastic Bugs and Where to Find Them in AI Benchmarks", arXiv 2511.16842, 2025 | Flags likely invalid items from response-pattern statistics under a unidimensional latent-construct assumption (item-total correlation and related item statistics), with an LLM-judge first pass, for expert review | Nine widely used benchmarks | Expert review of flagged items | Reports up to 84% precision on flagged items. It assumes a large response matrix. itemtriage Study 2 measures how item-total correlation degrades as the panel shrinks, beside plain disagreement and IRT, at fixed budgets. |
| Wang, Bianchi, Zhu et al., "Automated Benchmark Auditing for AI Agents and Large Language Models", arXiv 2605.26079, 2026 | An agentic auditor (ABA) that inspects each task for hidden environment dependencies, specification gaps and weak grading logic | 168 benchmarks in nine domains | Agent findings, with reported effects on rankings | Finds critical issues in over 25.7% of evaluated tasks. It reads each task, which costs a strong agent per item. itemtriage uses only model answers to decide which items a human or agent should read first. |
| Zhai, Wang, Wang et al., "HLE-Verified", arXiv 2602.13964, 2026 | Two-stage verification and repair of Humanity's Last Exam with expert review and model cross-checks | HLE: 668 verified, 1,143 revised and certified, 689 uncertain items | Expert review | Reports a link between model confidence and errors in problem statements or reference answers, on frontier models. itemtriage found that confidence adds little over plain disagreement with five small models (Study 1, H1a), which suggests the link needs strong models. Study 2 tests strength directly. |
| Epoch AI, "Humanity's Last Exam: benchmark review", reviewed 11 August 2026 | AI-assisted error search with every error confirmed by hand. A partial review that stops once errors pass its threshold for a flawed benchmark | 48 randomly sampled HLE questions, 6 from each of 8 categories | Human confirmation | Found 22 of 48 questions (46%) defective and rated the benchmark flawed. It shows how much expert time one audit takes. itemtriage is about ordering that time. |

## Routing related work

`routing/` is engineering, not a research claim. Conformal routing and cascades are an active area.

| Work | Relation to `routing/` |
|---|---|
| Uddin, Khider and Bauer, "Proactive Routing to Interpretable Surrogates with Distribution-Free Safety Guarantees", arXiv 2603.14623, 2026 | Derives a feasibility condition linking the base safe rate, the risk budget alpha and the AUC a gate needs, so a scorer of a given AUC can only certify some alphas at a given base rate. Routing's risk lane matches it: no alpha qualified at n=160, and with 828 calibration rows three scorers qualify at alpha 0.2 but none lower (ADR-035). |
| Dou, Lian and Li, "Conformal Cascade", arXiv 2607.25018, 2026 | Uses conformal prediction set size as the deferral rule in multi-tier LLM cascades, with a per-tier union bound. |
| Guo, Wu and Yiu, "RouteNLP", arXiv 2604.23577, ACL 2026 Industry Track | A closed-loop router with conformal cascading and distillation, deployed in a pilot. |
| Sun, Wang, Liu et al., "Compositional Conformal Risk Control for Multi-Stage LLM Cascades", arXiv 2611.11114, 2026 | Decomposes accepted-answer risk into stage-wise terms and certifies a frozen configuration on held-out data. |

## Novelty verdict

**Study 1.** No paper found compared a confidence-weighted small panel against small-panel IRT and plain
disagreement, on expert labels, at fixed review budgets. The comparison ran, and confidence added little
(H1a does not hold).

**Study 2.** Land and Bikel and Fantastic Bugs both work from large response matrices. Land and Bikel report
one small-panel ablation against weak labels (P@100 of 71.0% with 7 ability-sorted models). No paper found maps precision at a fixed review budget against panel size and panel
strength, on expert labels, with a blind replication on a second benchmark. That is the regime a team
building a new private benchmark is in, with no leaderboard matrix before release.

The contribution is narrow and empirical. Each ingredient exists separately: disagreement, item-total
correlation, IRT and a review budget. The value is the controlled comparison, its intervals, and an honest
result either way.

H1b, comparison with the published full-panel Δℓ, cannot be run. The Writer release
(`Writer/IRT-mislabeled-items`, ODC-BY) covers only 1,001 flagged items, not every item. Its multiple-choice
weak positives are 12 for GSM-MC and 9 for MATH-MC, and GPQA counts as all correct. That is below the
preregistered minimum of 100.

## Sources

- https://arxiv.org/abs/2605.30504
- https://arxiv.org/abs/2511.16842
- https://arxiv.org/abs/2605.26079
- https://arxiv.org/abs/2602.13964
- https://epoch.ai/benchmarks/hle/review
- https://arxiv.org/abs/2603.14623
- https://arxiv.org/abs/2607.25018
- https://arxiv.org/abs/2604.23577
- https://arxiv.org/abs/2611.11114
- https://arxiv.org/abs/2406.04127
- https://arxiv.org/abs/2608.30372
- https://aclanthology.org/2025.emnlp-main.1360/
- https://arxiv.org/abs/2103.14749
- https://arxiv.org/abs/2502.03461
- https://arxiv.org/abs/2609.00232
- https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0
- https://huggingface.co/datasets/Writer/IRT-mislabeled-items
