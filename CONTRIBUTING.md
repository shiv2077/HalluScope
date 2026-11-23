# Contributing

## Rules every change keeps

- Every reported number carries an interval: a 2000-resample bootstrap with seed 42, paired over items for
  differences, and Wilson for proportions. Raw counts carry none.
- Every number in a doc traces to a committed file made by a committed command. CI regenerates the reports
  and fails on any byte difference.
- Scorers never read labels. Labels are read only through `itemtriage/labels.py`, whose allowlist names
  every caller.
- Ties in a ranking resolve by a declared secondary scorer or in expectation, never by row order.
- No thresholds in config. Cutoffs exist only as calibration outputs.
- A new design choice gets an ADR in `docs/DECISIONS.md`.

## Before opening a pull request

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[test,score]"
ruff check .
python3 -m pytest tests
```

GPU work runs in a separate environment with `.[gpu]`, one CUDA process at a time, under
`itemtriage.gpu.GpuLock`.
