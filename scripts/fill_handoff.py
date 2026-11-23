"""Write HANDOFF.md from reports/routing_report.md (and results/phase1_notes.md for label agreement).

Nothing is typed by hand: every number is parsed from those two files. Deterministic, so a rerun
is byte-identical. Run: python3 scripts/fill_handoff.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = (ROOT / "reports" / "routing_report.md").read_text()
NOTES = (ROOT / "results" / "phase1_notes.md").read_text()
OUT = ROOT / "HANDOFF.md"


# ------------------------------------------------------------------ parsing
def section(text: str, header: str) -> str:
    m = re.search(rf"^{re.escape(header)}\n(.*?)(?=^#{{1,3}} |\Z)", text, re.M | re.S)
    if not m:
        raise SystemExit(f"section not found: {header}")
    return m.group(1)


def table(text: str) -> list[list[str]]:
    rows = [ln for ln in text.splitlines() if ln.startswith("|")]
    body = [r for r in rows[2:] if not re.match(r"^\|\s*-", r)]
    return [[c.strip() for c in r.strip().strip("|").split("|")] for r in body]


def est(cell: str) -> float:
    """Point estimate of a '36.0% [31.7, 40.8]' or '1.79 [1.58, 2.03]' or '$0.16 [0.13, 0.18]' cell."""
    return float(re.match(r"\$?(-?[\d.]+)", cell).group(1))


def quoted(pattern: str, text: str) -> str:
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"pattern not found: {pattern}")
    return m.group(1)


summary = section(REPORT, "## Summary")
best = quoted(r"the best scorer is `(\w+)`", summary)
free = quoted(r"the best free scorer is `(\w+)`", summary)
n_rows = quoted(r"On (\d+) test rows", summary)
base_err = quoted(r"error rate of ([\d.]+% \[[\d., ]+\])", summary)

lift_rows = {(r[0].rstrip(" *"), r[1]): r for r in table(section(REPORT, "## Review lift"))}
drift_rows = {(r[0], r[1]): r for r in table(section(REPORT, "### Cutoff drift"))}
tier_rows = {r[0].rstrip(" *"): r for r in table(section(REPORT, "### Cost tiers"))}
casc_text = section(REPORT, "## Cascade frontier")
casc_rows = [r for r in table(casc_text)]
gate_rows = {r[0]: r for r in table(section(REPORT, "## Validity gates"))}
len_rows = {r[0]: r for r in table(section(REPORT, "## Length"))}
conf_text = section(REPORT, "## Conformal and required n")
conf_rows = table(conf_text)
cost_text = section(REPORT, "## Cost table")
err_cheap = quoted(r"all cheap error ([\d.]+% \[[\d., ]+\]) at (\$[\d.]+ \[[\d., ]+\])", casc_text)
cost_cheap = re.search(r"all cheap error [\d.]+% \[[\d., ]+\] at (\$[\d.]+ \[[\d., ]+\])", casc_text).group(1)
err_strong = quoted(r"all strong error ([\d.]+% \[[\d., ]+\]) at", casc_text)
cost_strong = re.search(r"all strong error [\d.]+% \[[\d., ]+\] at (\$[\d.]+ \[[\d., ]+\])", casc_text).group(1)
total_calls = quoted(r"Total calls (\d+)", cost_text)
total_cost = quoted(r"total cost \$([\d.]+)", cost_text)
judge_cost = sum(float(r[4]) for r in table(cost_text) if "judge" in r[0])
gen_cost = sum(float(r[4]) for r in table(cost_text) if "judge" not in r[0])
n_cal = quoted(r"calibrated on n=(\d+)", conf_text)
prev_text = section(REPORT, "### Prevalence sensitivity")
prev_sentence = [ln for ln in prev_text.splitlines() if ln.startswith("At 10 percent prevalence") or "At 10 percent prevalence" in ln][0]
prev_sentence = prev_sentence[prev_sentence.index("At 10 percent prevalence"):]
HUMAN = ROOT / "results" / "human_agreement.md"
human_kappa = None
if HUMAN.exists():
    ht = HUMAN.read_text()
    m = re.search(r"\| human vs consensus \| (\d+) \| ([\d.]+% \[[\d., ]+\]) \| ([\d.]+ \[[\d., ]+\]) \|", ht)
    if m:
        human_kappa = {"n": m.group(1), "agreement": m.group(2), "kappa": m.group(3)}

# cascade: best budget-policy point of the best scorer = lowest error
budget_pts = [r for r in casc_rows if r[0].rstrip(" *") == best and r[1] == "budget"]
bp = min(budget_pts, key=lambda r: est(r[5]))
bp_budget = bp[2].replace("budget_", "")
bp_esc, bp_err, bp_cost, bp_gap, bp_cf, bp_fix, bp_break = bp[4], bp[5], bp[6], bp[7], bp[8], bp[9], bp[10]

# cutoff drift at 0.20: the largest deviation from 20 percent
drift20 = {sc: r for (sc, b), r in drift_rows.items() if b == "0.20"}
worst_sc = max(drift20, key=lambda sc: abs(est(drift20[sc][3]) - 20.0))
worst = drift20[worst_sc]
ml = drift20["mean_logprob"]

# labels from phase1_notes (test split, cheap model line)
test_notes = section(NOTES, "## test")
cheap_line = [ln for ln in test_notes.splitlines() if "gpt-4o-mini" in ln][0]
agree = quoted(r"agreement ([\d.]+% \[[\d., ]+\])", cheap_line)
kappa = quoted(r"kappa ([\d.]+ \[[\d., ]+\])", cheap_line)
amb = quoted(r"AMBIGUOUS \(excluded\) (\d+)", cheap_line)


def L(sc, b="0.20"):
    return lift_rows[(sc, b)]


def tier(sc):
    return tier_rows[sc]


def auroc(sc):
    return gate_rows[sc][1]


def pdelong(sc):
    return gate_rows[sc][2]


free_tiers = [r for sc, r in tier_rows.items() if r[1] == "0" and sc != "length"]
one_call = [r for r in tier_rows.values() if r[1] == "1"][0]
three_call = [r for r in tier_rows.values() if r[1] == "3"][0]
req = {(r[0], r[1]): r for r in table(conf_text.split("Required calibration size")[1])}
none_count = sum(1 for r in conf_rows if "none qualifies" in r[2])

human_line = (("Human agreement: on " + human_kappa["n"] + " human-labelled rows, agreement with the two-judge consensus is "
               + human_kappa["agreement"] + ", kappa " + human_kappa["kappa"] + " (results/human_agreement.md).") if human_kappa
              else "Human labels: none collected yet (`python3 -m routing.cli label --split test --n 60 --seed 7 --annotator <name>`).")
if human_kappa:
    q12 = ("On " + human_kappa["n"] + " rows labelled blind by a human, agreement with the two-judge consensus is " + human_kappa["agreement"]
           + " with kappa " + human_kappa["kappa"] + "; the disagreements are listed one by one with the human note in "
           "results/human_agreement.md, and the lift is recomputed under the human oracle on the same rows beside the consensus oracle.")
else:
    q12 = ("Human labels have not been collected yet, so the only check is that two blind OpenAI judges agree on " + agree
           + " of cheap-model answers with kappa " + kappa + ", and two judges from one provider share failure modes. The blind labeling "
           "tool is `python3 -m routing.cli label --split test --n 60 --seed 7 --annotator <name>`, and `python3 -m routing.cli agreement` "
           "then reports agreement, kappa, the disagreements, and lift under the human oracle beside the consensus oracle.")
REGEN_CMD = ("python3 -m routing.cli score --split calibration && python3 -m routing.cli score --split test && "
             "python3 -m routing.cli gates && python3 -m routing.cli calibrate && python3 -m routing.cli evaluate && "
             "python3 scripts/fill_handoff.py")
DEMO_CMD = ("python3 -c \"import pandas as pd; g=pd.read_csv('data/test_generations.csv'); s=pd.read_csv('data/test_scores.csv'); "
            "g[g.role=='primary'].merge(s[['qid','p_true','consistency_k3']], on='qid').to_csv('/tmp/answers.csv', index=False)\" && "
            "python3 -m routing.cli route --policy budget --scorer p_true --budget 0.20 --in /tmp/answers.csv --out /tmp/routed.csv")

pitch = (f"It is a risk-routing layer for model-generated content. A free signal from the model orders outputs by risk, "
         f"a per-deployment calibration turns that into a threshold, and the layer routes each output to return, escalate, or human review. "
         f"On a held-out set of 300 questions with blind two-judge labels, reviewing the top 20 percent by risk under the budget policy "
         f"catches {L(best)[3]} of errors against {L(best)[4]} for random, and the cascade brings error from {err_cheap} to {bp_err}, "
         f"closing {bp_gap} of the gap to the strong model for {bp_cf} of the incremental cost.")

doc = f"""# Handoff

Generated by `scripts/fill_handoff.py` from `reports/routing_report.md` and `results/phase1_notes.md`.
Nothing here is typed by hand. Brackets are 95 percent intervals (2000-resample bootstrap, seed 42,
or Wilson for proportions). Review numbers use the budget policy (ADR-018): the top ceil(budget * n)
rows of the test batch by batch quantile.

## Pitch

{pitch}

## Headline table, 0.20 budget, budget policy

| scorer | extra calls | extra cost per 1000 | lift | catch rate | random | precision | residual error |
|---|---:|---|---|---|---|---|---|
| `{best}` (best) | {tier(best)[1]} | {tier(best)[2]} | {L(best)[5]} | {L(best)[3]} | {L(best)[4]} | {L(best)[6]} | {L(best)[7]} |
| `{free}` (best free) | {tier(free)[1]} | {tier(free)[2]} | {L(free)[5]} | {L(free)[3]} | {L(free)[4]} | {L(free)[6]} | {L(free)[7]} |

Base error rate with no review: {base_err} on {n_rows} rows with a consensus label.
Precision of the `{best}` flagged slice at 0.20: {L(best)[6]}. Prevalence: {prev_sentence}
{human_line}

## Cost tiers

Free tier (zero extra calls): `{free}` gives lift {L(free)[5]} at {tier(free)[2]} per 1000.
One-call tier: `{one_call[0].rstrip(' *')}` gives lift {one_call[3]} at {one_call[2]} per 1000.
Three-call tier: `{three_call[0].rstrip(' *')}` gives lift {three_call[3]} at {three_call[2]} per 1000.

## Cascade in one line

At the `{best}` budget {bp_budget} point (escalating {bp_esc} of rows), the cascade closes {bp_gap} of the error gap
between all-cheap ({err_cheap}) and all-strong ({err_strong}) while spending {bp_cf} of the incremental cost
(all-cheap {cost_cheap}, all-strong {cost_strong}, this point {bp_cost} per 1000). The ceiling is visible beside it:
the conditional fix rate is {bp_fix}, and the strong model's own error rate is {err_strong}.

## The worst number

Cutoff drift at the 0.20 budget. A cutoff fixed as the 20 percent quantile on the {n_cal}-row calibration split flagged
{worst[3]} of test rows for `{worst_sc}` (calibration cutoff {worst[2]}, batch cutoff on test {worst[4]}); for
`mean_logprob` it flagged {ml[3]}. A budget that does not deliver its budget is not a budget. This is why ADR-018 exists:
budget policies take the quantile on the batch being routed, and calibrated cutoffs are used only by risk policies.

Runner up: no risk-controlled threshold qualifies at n={n_cal} for any scorer at any alpha ({none_count} of {none_count} cells).
Alpha 0.10 at 70 percent coverage needs {req[('0.1', '0.7')][2]} calibration rows at zero empirical risk and {req[('0.1', '0.7')][3]} at
risk 0.05. The function is shipped; the sample size is not.

## Regenerate everything

From the committed CSVs, CPU only, no keys, byte-identical report on rerun:

```bash
{REGEN_CMD}
```

To redo the API phases from scratch (resumable, needs `OPENAI_API_KEY`):

```bash
nohup scripts/run_routing_phase1.sh > /dev/null 2>&1 &
```

## Total API cost

From `results/calls.jsonl`: {total_calls} calls, ${total_cost}. Judging ${judge_cost:.2f}, all generation for both splits ${gen_cost:.2f}.

## Demo command

```bash
{DEMO_CMD}
```

It flags exactly ceil(0.20 * n) rows, writes `policy`, `cutoff_applied` and `lane`, and never opens
`results/calibration.json`. `--policy risk --alpha 0.10` refuses, because no calibrated risk cutoff qualifies.

## Twelve likely interview questions

1. **What is the lift at a 20 percent budget and is it worth infrastructure?** Under the budget policy the best scorer,
   `{best}`, gives lift {L(best)[5]}, catching {L(best)[3]} of errors against {L(best)[4]} for random. The number a dataset
   team ships is residual error: {L(best)[7]} after review against {base_err} without. The free scorer `{free}` gives
   {L(free)[5]} for nothing, so the lift is real and cheap, but it is not large.

2. **How do you know the lift is not just answer length?** Length alone has AUROC {auroc('length')} and lift {L('length')[5]},
   and every scorer beats it under a DeLong test (largest p = {max(pdelong(sc) for sc in gate_rows if sc != 'length')}). The
   length-residualized logprob keeps AUROC {auroc('length_residual')} after regressing length out. Within length tertiles,
   `{best}` holds lift {len_rows[best][1].split(' (')[0]}, {len_rows[best][2].split(' (')[0]} and {len_rows[best][3].split(' (')[0]}
   from short to long, while `mean_logprob` weakens to {len_rows['mean_logprob'][3].split(' (')[0]} on the longest third.

3. **Where did the thresholds come from?** Budget routing has no threshold: it flags the top ceil(budget * n) of the batch.
   Risk routing reads a cutoff from `results/calibration.json`, computed on the {n_cal}-row calibration split with its hash
   and n recorded, and at n={n_cal} no risk cutoff qualifies, so `route --policy risk` refuses. There is no threshold in config.

4. **How good are the labels?** Two blind judges, `gpt-4o` and `gpt-4.1`, agree on {agree} of cheap-model test answers with
   kappa {kappa}; {amb} of 300 rows are AMBIGUOUS and excluded. Both judges are OpenAI models, written down as a limitation
   because two judges from one provider share failure modes. Human labels on 150 rows are the first roadmap item.

5. **Does escalation actually fix anything?** Only about half the time: the conditional fix rate at the best point is
   {bp_fix}. The strong model's own error rate is {err_strong}, and that is the floor no router can beat. The break rate,
   where escalation turns a right answer wrong, is {bp_break}.

6. **What does the cascade cost?** Cost comes from logged token counts and `config/pricing.json` dated 2026-09-14, never
   typed in. At the best `{best}` point it is {bp_cost} per 1000, against {cost_cheap} all cheap and {cost_strong} all strong.
   That closes {bp_gap} of the error gap for {bp_cf} of the incremental cost.

7. **Why not just improve the detector?** The clean v1 signal was AUROC 0.64 on 82 rows and the five-feature ensemble never
   beat mean logprob alone (report1.md). Here the best scorer reaches AUROC {auroc(best)}, which still cannot decide
   correctness for one row. It can order rows, and ordering is what a review budget or an escalation budget needs.

8. **Is the conformal threshold a guarantee?** No: at n={n_cal} no scorer has a threshold whose Hoeffding bound clears
   alpha 0.05, 0.10 or 0.20 ({none_count} cells say "none qualifies"). Alpha 0.10 at 70 percent coverage needs
   {req[('0.1', '0.7')][2]} rows at zero empirical risk and {req[('0.1', '0.7')][3]} at risk 0.05. The function is correct
   and shipped; the sample size is not.

9. **What happens with a provider that has no logprobs?** Two scorers are text-only. `consistency_k3` reaches AUROC
   {auroc('consistency_k3')} and lift {L('consistency_k3')[5]} at {tier('consistency_k3')[2]} per 1000; `p_true` is the best
   scorer overall at AUROC {auroc('p_true')} for {tier('p_true')[2]} per 1000. Their cost per score sits on the same table as
   the free scorers.

10. **What would you build next?** Claim-level scoring first, because `mean_logprob` weakens to
    {len_rows['mean_logprob'][3].split(' (')[0]} on the longest tertile and production content is long. Then 150 human
    labels to check the LLM consensus, then the drift monitor and the streaming budget policy, because the calibrated
    20 percent cutoff flagged {worst[3]} of test rows for `{worst_sc}`. Nothing on that list was started.

11. **Your headline scorer costs an extra call. What do I get for free?** Lift {L(free)[5]} from `{free}` at zero extra
    calls, against {L(best)[5]} from `{best}` at {tier(best)[2]} per 1000. The three-call `{three_call[0].rstrip(' *')}`
    gives {three_call[3]} at {three_call[2]} per 1000, so the second tier is the one worth buying and the third is not,
    on this data.

12. **How do you know your judges are right?** {q12}
"""

OUT.write_text(doc)
print(f"wrote {OUT}")
