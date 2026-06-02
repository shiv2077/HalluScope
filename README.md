# HalluScope

## Overview

HalluScope is a model-agnostic reliability layer for Large Language Models. During inference, it computes a **Hallucination Risk Index (HRI ∈ [0, 1])** from token-level uncertainty signals — log-probability variance, confidence decay, mean log-probability, and (for local models) Semantic Energy and Representation Collapse Detection — and uses that score to classify each response as **LOW**, **MEDIUM**, or **HIGH** risk. When risk crosses the calibrated HIGH threshold, an intervention service replaces the LLM's answer with a clarification request rather than passing potentially hallucinated content to the user.

The system supports two evaluation backends behind a uniform `StandardizedResult` interface: an **API mode** built on the OpenAI Chat Completions API (`gpt-4o-mini`, with `logprobs=True, top_logprobs=5`) and a **local mode** that loads HuggingFace causal-LM checkpoints (default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) and additionally captures hidden states + raw logits per generated token, enabling internal-signal metrics that the OpenAI API cannot expose. Thresholds are calibrated against HalluEval-QA — see `eval/REPORT.md` for AUROC, ROC plots, and the precision-floor operating-point selection rationale.

## Architecture

Three layers, plus a benchmark harness:

- **Frontend (React + Vite)** — Two pages: `ChatPage` (chat UI with mode picker, intervention banner, risk badge per bubble) and `MetricsPage` (per-message dashboard: 11-card metric grid, per-token HRI line chart with threshold lines, token risk heatmap, and — local-mode only — internal-signal trajectory charts for Semantic Energy and RCD).
- **API (FastAPI)** — Three routers: `POST /chat` and `POST /chat/reset` (chat orchestration with per-mode singleton risk services), `GET /metrics/{message_id}` (dashboard lookup), and `GET /health` (liveness + local-model warmup status). Background thread pre-warms the local model at startup so the first chat doesn't block.
- **Reliability Engine** — `metrics/` modules compute the per-token signals, `risk/hri_engine.py` aggregates them via the calibrated weighted-sigmoid HRI formula and classifies into LOW/MEDIUM/HIGH, and `services/intervention_service.py` decides when to suppress the LLM answer and inject a clarification.

A SQLite-backed message store (`backend/app/storage.py`) persists analysed messages so the `/metrics/{id}` page survives backend restarts.

## Tech Stack

- **Backend**: FastAPI 0.110+, Python 3.10+, HuggingFace Transformers 4.40+, OpenAI Python SDK 1.14+, PyTorch 2.2+, Pydantic 2
- **Frontend**: React 18, Vite 5, recharts 2, react-router 6
- **Storage**: SQLite (stdlib `sqlite3`) for message persistence; in-memory dicts for short-lived per-mode risk state
- **Models**: `gpt-4o-mini` (API mode), `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (local mode default; configurable via `LOCAL_MODEL` env var)
- **Evaluation**: scikit-learn 1.x, sentence-transformers 5.x, matplotlib 3.x, HuggingFace `datasets`

## Setup

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env     # then edit ../.env and add your OPENAI_API_KEY
cd ..
uvicorn backend.app.main:app --reload
# serves on http://127.0.0.1:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# opens http://localhost:5173
```

### Benchmark (optional, for re-calibration)

```bash
python3 eval/build_subset.py             # downloads HalluEval QA subset
python3 eval/run_benchmark.py --mode api    # ~5 min, ~$0.02 on gpt-4o-mini
python3 eval/run_benchmark.py --mode local  # ~1.5 min on GPU
python3 eval/match.py --csv eval/results_api.csv   --threshold 0.35
python3 eval/match.py --csv eval/results_local.csv --threshold 0.35
python3 eval/analyze.py                  # AUROC, ROC plots, summary.json
python3 eval/pick_thresholds.py          # precision-floor operating-point selection
# results: eval/REPORT.md
```

## Calibration

Thresholds calibrated against HalluEval-QA (n=100, balanced 62 hallucination / 38 correct under our v2 free-form-answer methodology).

- **API-mode AUROC: 0.785**
- Operating point selected to maintain precision ≥ 0.85 on the calibration set
- Live thresholds: `RISK_LOW_THRESHOLD = 0.15`, `RISK_HIGH_THRESHOLD = 0.30`

The selection between best-F1 (rejected: would fire on ~80% of normal traffic) and the precision-floor operating point (chosen) is documented in detail in `eval/REPORT.md` § "Operating Point Selection". The thresholds themselves live in `backend/app/risk/hri_engine.py` with the calibration date, AUROC, and fallback rationale in the source comment.

## Known Limitations

- **Calibration set is small.** n=100 for API, with effectively n=3 in the local-mode "correct" class. AUROC has wide confidence intervals; held-out validation on a separate slice is scoped for the next iteration.
- **Local mode currently uses TinyLlama-1.1B-Chat.** The architecture is model-agnostic; Qwen2.5-3B-Instruct (the originally-specified local model) is supported via the `LOCAL_MODEL` environment variable but requires a stronger GPU than TinyLlama and was deferred from the demo build.
- **Stage 1 pre-hoc prompt analysis is scoped for the next iteration.** All current scoring happens after generation completes; per-token signals are computed post-hoc from cached logits and hidden states.
- **Logit Lens Divergence and Entity Expansion are deferred to the next phase.** Of the six internal-signal metrics in the original spec, two are shipped (Semantic Energy, RCD); the remaining four require either multi-layer hidden-state extraction or an NER pipeline that is out of scope for the current build.
- **Single-user demo deployment.** The `_risk_services` per-mode singleton, in-process model warmup thread, and CORS allow-list are all configured for a single-machine demo; multi-tenant session isolation, authentication, and rate limiting are not implemented.
- **Conversation history corruption was hand-fixed in Phase 1** — when an intervention fires, the suppressed LLM answer is now stripped from history before the clarification is injected. This was a real bug, not a design feature; mentioned here for transparency.

## Project Structure

```
HalluScope/
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py              FastAPI app factory, startup hooks, /health
│       ├── config.py            env vars, model defaults
│       ├── storage.py           SQLite-backed message store
│       ├── api/                 chat.py, metrics.py routers
│       ├── adapters/            base, openai, local — uniform StandardizedResult
│       ├── metrics/             logprob, confidence, internal-signal metrics
│       ├── risk/                hri_engine.py — HRI + thresholds + classification
│       ├── services/            llm, risk, intervention orchestration
│       └── models/              Pydantic request/response schemas
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx, App.jsx
│       ├── api.js               fetch wrappers for /chat, /metrics, /chat/reset
│       ├── pages/               ChatPage, MetricsPage
│       └── components/          ChatWindow, MessageBubble, MetricsPanel,
│                                RiskBadge, TokenHeatmap
├── eval/
│   ├── REPORT.md                Headline benchmark report (read this first)
│   ├── build_subset.py          Download + dedupe HalluEval QA into 100 questions
│   ├── run_benchmark.py         /chat → CSV with HRI per row
│   ├── match.py                 SBERT cosine matcher → fills model_match column
│   ├── analyze.py               AUROC, ROC plots, best-F1 + at-current metrics
│   ├── pick_thresholds.py       Precision-floor operating-point sweep
│   ├── datasets/                Subset JSONs
│   ├── results_api.csv,
│   │   results_local.csv        Per-row benchmark results
│   ├── roc_api.png,
│   │   roc_local.png            ROC plots
│   └── summary.json,
│       operating_point.json     Machine-readable summaries
├── scripts/                     Phase 1/2 verification scripts
├── README.md                    (you are here)
└── .env.example                 Template for secrets file
```

## Authors

- **V Dinessh** — RA2211003011234
- **Shivnarayan S** — RA2211003011216
- **Guide:** Dr. R.I. Minu, Department of Computing Technologies, SRM IST
