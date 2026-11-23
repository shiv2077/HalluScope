# Phase 1 notes: data, generations, labels

Agreement carries a Wilson interval; kappa a bootstrap interval (2000 resamples, seed 42). Counts are exact.

## calibration

- Questions: 162. Generation rows: 972.
- Rows per role: primary=162, resample_1=162, resample_2=162, resample_3=162, strong=162, ptrue=162.
- `gpt-4o`: 162 judged rows, agreement 98.1% [94.7, 99.4], kappa 0.952 [0.890, 1.000], consensus INCORRECT 41, AMBIGUOUS (excluded) 3. Judge A marked 44 INCORRECT, judge B 41.
- `gpt-4o-mini`: 162 judged rows, agreement 98.8% [95.6, 99.7], kappa 0.974 [0.935, 1.000], consensus INCORRECT 65, AMBIGUOUS (excluded) 2. Judge A marked 67 INCORRECT, judge B 65.

## test

- Questions: 300. Generation rows: 1800.
- Rows per role: primary=300, resample_1=300, resample_2=300, resample_3=300, strong=300, ptrue=300.
- `gpt-4o`: 300 judged rows, agreement 95.0% [91.9, 96.9], kappa 0.877 [0.813, 0.933], consensus INCORRECT 77, AMBIGUOUS (excluded) 15. Judge A marked 91 INCORRECT, judge B 78.
- `gpt-4o-mini`: 300 judged rows, agreement 96.3% [93.6, 97.9], kappa 0.927 [0.881, 0.967], consensus INCORRECT 139, AMBIGUOUS (excluded) 12. Judge A marked 145 INCORRECT, judge B 144.

## Cost so far

From results/calls.jsonl, priced on 2026-09-14: 4620 calls, 502199 input tokens, 122892 output tokens, $1.5855.

- calibration:judge_a: 324 calls, $0.2661
- calibration:judge_b: 324 calls, $0.2051
- calibration:primary: 162 calls, $0.0035
- calibration:ptrue: 162 calls, $0.0020
- calibration:resample_1: 162 calls, $0.0035
- calibration:resample_2: 162 calls, $0.0035
- calibration:resample_3: 162 calls, $0.0035
- calibration:strong: 162 calls, $0.0624
- test:judge_a: 600 calls, $0.4952
- test:judge_b: 600 calls, $0.3886
- test:primary: 300 calls, $0.0070
- test:ptrue: 300 calls, $0.0039
- test:resample_1: 300 calls, $0.0071
- test:resample_2: 300 calls, $0.0070
- test:resample_3: 300 calls, $0.0070
- test:strong: 300 calls, $0.1201

## Limitations

- Two judges from one provider: judge_a is `gpt-4o` and judge_b is `gpt-4.1`, both OpenAI. Two judges from one provider share more failure modes than two providers would, so agreement overstates label reliability.
- All 162 calibration primary rows were regenerated: the v1 CSVs store only a scalar mean logprob, not per-token logprobs.
- Labels are from LLM judges, not humans.
