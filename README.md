# HalluScope

**A reliability layer for LLMs that scores how likely each answer is to be a hallucination — and steps in before a risky answer reaches the user.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

HalluScope wraps a language model and, for every response it generates, computes a single
**Hallucination Risk Index (HRI ∈ [0, 1])** from the model's own token-level uncertainty
signals. It classifies each answer as **LOW**, **MEDIUM**, or **HIGH** risk, and when risk
crosses the calibrated HIGH threshold it suppresses the answer and returns a clarification
request instead of passing a potentially hallucinated answer to the user.

It is a **real-time, zero-shot, internal-signal** detector: it needs no reference answer, no
extra labelled data, and no second judge model — it reads the uncertainty the generating model
already exposes. It runs in two interchangeable modes behind one uniform interface:

- **API mode** — wraps the OpenAI Chat Completions API (`gpt-4o-mini`, with
  `logprobs=True, top_logprobs=5`) and scores risk from the returned log-probabilities.
- **Local mode** — runs a HuggingFace causal-LM locally (default
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) and additionally captures per-token hidden states and
  raw pre-softmax logits, unlocking internal-signal metrics the hosted API can't expose.

---

## How it works

```
                    user prompt
                        │
          ┌─────────────▼─────────────┐
          │   LLM adapter (api|local) │   generate answer, capture per-token
          │   → StandardizedResult    │   logprobs (+ logits & hidden states locally)
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │   metrics/  (per token)   │   logprob variance · confidence decay ·
          │                           │   mean logprob · semantic energy · RCD
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │   risk/hri_engine.py      │   HRI = sigmoid(scale·Σ wᵢ·featureᵢ + bias)
          │   → HRI + LOW/MED/HIGH    │   weights & thresholds from config/*.json
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │  services/intervention    │   HRI ≥ HIGH → suppress answer,
          │                           │   return a clarification request
          └─────────────┬─────────────┘
                        │
                   chat response  (answer or clarification, + full metrics)
```

**The HRI signals are computed _after_ generation finishes, from cached logits and hidden
states** — not streamed live during decoding. Two feature sets feed the same weighted-sigmoid
formula:

| Mode | Signals feeding the HRI score |
|---|---|
| **API** (3) | log-probability variance · confidence decay · mean log-probability |
| **Local** (5) | the 3 above **+** semantic energy (logit flatness) **+** inverted representation-collapse (RCD) |

- **Log-probability variance / mean / confidence decay** — token-level certainty and how it
  trends across the answer; available from any provider that returns logprobs.
- **Semantic energy** — a bounded ([0, 1]), shift-invariant measure of how *flat* the
  next-token distribution is (flatter = more uncertain). Local mode only, since it needs raw
  logits.
- **Representation Collapse Detection (RCD)** — `1 − cos(hₜ, hₜ₋₁)` between consecutive hidden
  states; flags degenerate/repetitive generation. Local mode only.

Risk bands and the intervention trigger share a **single source of truth**
(`config/thresholds.json`), so "classified HIGH" and "intervention fired" are by construction
the same set of responses:

| Band | HRI range | Behaviour |
|---|---|---|
| LOW | `< 0.15` | pass through |
| MEDIUM | `0.15 – 0.30` | pass through, flagged |
| HIGH | `≥ 0.30` | **intervene** — answer suppressed, clarification returned |

All weights, sigmoid parameters, normalization constants, and thresholds live in
`config/*.json` — there are no hard-coded risk numbers in the engine. The current thresholds
were selected with a precision-floor sweep on a HaluEval-QA calibration set; the full
methodology, operating-point rationale, and per-mode results live in
[`eval/REPORT.md`](eval/REPORT.md).

> Two further internal signals — **Logit Lens Divergence** and **Confidence-Stability Gap** —
> are implemented in `metrics/internal_metrics.py` and surfaced in the API response, but are
> **not currently part of the HRI score**; they are exploratory and reported for transparency.

---

## Architecture

Three layers plus a benchmark harness:

- **Frontend (React 18 + Vite 5)** — two pages: `ChatPage` (chat UI with an API/Local mode
  picker, per-bubble risk badge, and an intervention banner) and `MetricsPage` (per-message
  dashboard: metric grid, per-token HRI line chart with threshold lines, a token risk heatmap,
  and — local mode only — Semantic Energy / RCD trajectory charts).
- **API (FastAPI)** — routers for `POST /chat` + `POST /chat/reset` (chat orchestration with
  per-mode singleton risk services that hold conversation history), `GET /metrics/{message_id}`
  (dashboard lookup), `POST /analyze` (legacy single-shot), and `GET /health`. A background
  thread pre-warms the local model at startup so the first chat doesn't block.
- **Reliability engine** — `metrics/` computes the per-token signals, `risk/hri_engine.py`
  aggregates them into the HRI and classifies the band, and `services/intervention_service.py`
  decides when to suppress an answer and inject a clarification. A SQLite store
  (`backend/app/storage.py`) persists analysed messages so `/metrics/{id}` survives restarts.

---

## Quickstart

**Prerequisites:** Python 3.10+, Node 18+. An OpenAI API key is needed for API mode; local
mode runs without one (it downloads a small HuggingFace model on first run).

### 1. Backend

```bash
git clone https://github.com/shiv2077/HalluScope.git
cd HalluScope

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

cp .env.example .env               # then edit .env and add your OPENAI_API_KEY
uvicorn backend.app.main:app --reload
# → http://127.0.0.1:8000   (interactive docs at /docs)
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

The frontend talks to the backend on `127.0.0.1:8000`; CORS is pre-configured for the Vite dev
server on `5173`.

### 3. Try it

Open `http://localhost:5173`, pick **API** or **Local** mode, and ask a question. Confident,
answerable questions score LOW; questions the model can't actually know (e.g. future events)
score HIGH and trigger an intervention.

---

## API reference

| Method | Endpoint | Purpose | Key fields |
|---|---|---|---|
| `GET`  | `/health` | Liveness + local-model warmup status | → `status`, `local_model_loaded`, `local_model_name` |
| `POST` | `/chat` | Run one chat turn through the risk pipeline | body: `message`, `evaluation_mode` (`api`\|`local`), optional `temperature`, `seed`, `apply_intervention` → `answer`, `hri`, `risk_level`, `metrics`, `token_details`, `intervention`, `message_id` |
| `POST` | `/chat/reset` | Clear conversation history for both modes | → `status` |
| `GET`  | `/metrics/{message_id}` | Full uncertainty analysis for a past message | → `metrics`, `token_details`, `per_token_hri`, `internal_per_token`, `risk_level` |
| `POST` | `/analyze` | Legacy single-shot analyze (API mode only) | body: `prompt`, `max_tokens`, `temperature` → `answer`, `hri`, `risk_level`, `metrics` |

Example:

```bash
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Who won the 2027 Nobel Peace Prize?","evaluation_mode":"api"}'
```

---

## Evaluation

HalluScope ships with a reproducible benchmark harness in [`eval/`](eval/) built on
**HaluEval QA** ([`pminervini/HaluEval`](https://huggingface.co/datasets/pminervini/HaluEval)).
Rather than judging a supplied candidate answer, the harness sends only the question, lets the
model produce its own free-form answer, captures the HRI on that answer, and grades correctness
by SBERT cosine similarity against the dataset reference. Classical uncertainty baselines
(P(True), Semantic Entropy, Verbalized Confidence, and a text-similarity confound control) are
implemented in `eval/baselines/` for comparison.

The full methodology, threshold-calibration sweep, operating-point selection, and per-mode
results are written up in **[`eval/REPORT.md`](eval/REPORT.md)**. To reproduce:

```bash
python3 eval/build_subset.py                                  # build the QA subset (seed=42)
python3 eval/run_benchmark.py --mode api --eval-deterministic # or --mode local
python3 eval/match.py --csv eval/results_api.csv --threshold 0.35
python3 eval/analyze.py                                       # metrics + ROC plots
```

---

## Tech stack

- **Backend** — FastAPI, Python 3.10+, PyTorch 2.2+, HuggingFace Transformers 4.40+, OpenAI
  Python SDK 1.14+, Pydantic 2, SQLite (stdlib).
- **Frontend** — React 18, Vite 5, React Router 6, Recharts 2.
- **Models** — `gpt-4o-mini` (API mode); `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (local default,
  configurable to any `AutoModelForCausalLM` repo via the `LOCAL_MODEL` env var,
  e.g. `Qwen/Qwen2.5-3B-Instruct`).
- **Evaluation** — scikit-learn, sentence-transformers, matplotlib, HuggingFace `datasets`.

---

## Project structure

```
HalluScope/
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py            FastAPI app, startup hooks, /health
│       ├── config.py          env vars + model defaults
│       ├── config_store.py    loads config/*.json (single source of truth)
│       ├── storage.py         SQLite message store
│       ├── api/               chat.py, metrics.py, legacy /analyze
│       ├── adapters/          base · openai · local  (uniform StandardizedResult)
│       ├── metrics/           logprob, confidence-decay, internal-signal metrics
│       ├── risk/hri_engine.py HRI formula + LOW/MED/HIGH classification
│       ├── services/          llm · risk · intervention orchestration
│       └── models/            Pydantic request/response schemas
├── frontend/
│   └── src/
│       ├── api.js             single client for all backend calls
│       ├── pages/             ChatPage, MetricsPage
│       └── components/        ChatWindow, MessageBubble, MetricsPanel,
│                              RiskBadge, TokenHeatmap
├── config/                    weights.json, thresholds.json, norm.json, ...
├── eval/                      benchmark harness, baselines, REPORT.md
├── tests/                     pytest suite (run: python3 -m pytest)
└── scripts/                   verification / dev scripts
```

---

## Limitations & scope

- **Calibrated on a small set.** Thresholds were chosen on an n=100 HaluEval-QA slice; in local
  mode TinyLlama produced only 3 "correct" answers, so the local-mode class balance is severe.
  Treat the local numbers as directional and the thresholds as a starting point, not a
  production calibration — a larger held-out set is the obvious next step.
- **Single dataset.** Evaluated on HaluEval QA only. The `eval/build_subset.py` scaffold can
  pull a TruthfulQA subset, but it is not part of the reported results.
- **Post-hoc, not streaming.** All signals are computed after the answer is generated, from
  cached logits/hidden states — not live during decoding.
- **API mode is OpenAI-only.** The adapter layer is provider-agnostic by design, but only the
  OpenAI and local-HuggingFace adapters are implemented.
- **Single-user demo.** Per-mode singleton risk services, an in-process model-warmup thread, and
  a dev CORS allow-list are configured for a single machine. Multi-tenant session isolation,
  auth, and rate limiting are out of scope for this build.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: format Python with `ruff`/`black`, keep all
risk constants in `config/*.json` (never hard-code them in the engine), and run
`python3 -m pytest` before opening a PR.

## License

[MIT](LICENSE) © 2026 V Dinessh and Shivnarayan S.

## Authors

- **V Dinessh** — RA2211003011234
- **Shivnarayan S** — RA2211003011216
- **Guide:** Dr. R.I. Minu, Department of Computing Technologies, SRM Institute of Science and Technology
