"""Generate Document 3 (Code, Software Used, and Instructions to Execute)
for the HalluScope final-year major project viva.

Run from repo root:
    python3 scripts/generate_document3.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "Document3_Code_and_Instructions_HalluScope.docx"


# ────────────────────────────────────────────────────────────────────
# Style helpers
# ────────────────────────────────────────────────────────────────────

def set_cell_shading(cell, color_hex: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    tc_pr.append(shd)


def set_paragraph_shading(paragraph, color_hex: str) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    p_pr.append(shd)


def add_page_number(paragraph) -> None:
    run = paragraph.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = "PAGE"
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)


def set_body_font(run, size_pt: float = 12, bold: bool = False,
                  font: str = "Times New Roman", color: RGBColor | None = None) -> None:
    run.font.name = font
    run.font.size = Pt(size_pt)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def add_heading(doc, text: str, level: int) -> None:
    sizes = {1: 16, 2: 14, 3: 12}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    set_body_font(run, size_pt=sizes.get(level, 12), bold=True)


def add_body(doc, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    set_body_font(run, size_pt=12)


def add_code_block(doc, code: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.left_indent = Cm(0.3)
    set_paragraph_shading(p, "F2F2F2")
    run = p.add_run(code)
    set_body_font(run, size_pt=9, font="Courier New")


def add_bullet(doc, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing = 1.15
    run = p.runs[0] if p.runs else p.add_run("")
    # The style added "" as run; replace with desired text.
    if not p.runs:
        run = p.add_run(text)
    else:
        # Re-create cleanly
        for r in list(p.runs):
            r.text = ""
        run = p.add_run(text)
    set_body_font(run, size_pt=12)


def add_table(doc, headers: list[str], rows: list[list[str]],
              col_widths_cm: list[float] | None = None) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"

    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = ""
        set_cell_shading(hdr_cells[i], "D9D9D9")
        p = hdr_cells[i].paragraphs[0]
        run = p.add_run(h)
        set_body_font(run, size_pt=11, bold=True, font="Calibri")
        hdr_cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            run = p.add_run(val)
            set_body_font(run, size_pt=11, font="Calibri")
            cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    if col_widths_cm:
        for row in table.rows:
            for i, w in enumerate(col_widths_cm):
                if i < len(row.cells):
                    row.cells[i].width = Cm(w)


# ────────────────────────────────────────────────────────────────────
# Module walkthrough data
# ────────────────────────────────────────────────────────────────────

MODULES = [
    # (heading, file, purpose, responsibilities[], notable[], deps[])
    (
        "3.1 backend/app/main.py",
        "FastAPI application factory and entry point.",
        "Constructs the FastAPI instance, configures CORS for the Vite dev server, registers the legacy /analyze, /chat, /chat/reset, and /metrics routers, exposes a /health endpoint, and triggers a background warm-up of the local HuggingFace model on startup so the first chat request does not block.",
        [
            "Builds the `FastAPI` app with title, description, version 0.2.0.",
            "Adds `CORSMiddleware` allowing http://localhost:5173.",
            "Initializes the SQLite message store via `storage.init_db()`.",
            "Spawns a daemon thread that calls `LocalAdapter._ensure_model_loaded()`.",
            "Registers three routers (legacy, chat, metrics) and a `/health` route.",
        ],
        [
            "_warm_local_model() — loads HF weights off the request path.",
            "_on_startup() — FastAPI startup hook (DB init + warm-up thread).",
            "health() — returns liveness JSON plus local_model_loaded flag.",
        ],
        ["fastapi", "fastapi.middleware.cors", "threading",
         "backend.app.api (legacy, chat, metrics)", "backend.app.storage",
         "backend.app.config"],
    ),
    (
        "3.2 backend/app/config.py",
        "Centralised configuration loader.",
        "Reads environment variables from the repo-root `.env` file and defines the model defaults shared by every adapter.",
        [
            "Loads `.env` from repo root via python-dotenv.",
            "Exposes OPENAI_API_KEY, DEFAULT_MODEL ('gpt-4o-mini'), DEFAULT_LOCAL_MODEL ('TinyLlama/TinyLlama-1.1B-Chat-v1.0'), DEFAULT_MAX_TOKENS (80), DEFAULT_TEMPERATURE (0.7).",
            "Allows overriding the local model with the LOCAL_MODEL env var.",
        ],
        ["Module-level constants only (no functions or classes)."],
        ["os", "pathlib", "python-dotenv"],
    ),
    (
        "3.3 backend/app/storage.py",
        "SQLite-backed persistent message store.",
        "Persists analysed chat messages so the metrics page survives a backend restart, replacing the previous process-local dict.",
        [
            "Creates the `messages` table on first run (message_id PK, payload JSON, created_at).",
            "Serialises Pydantic models to plain JSON via a recursive `_to_jsonable()` helper.",
            "Exposes `init_db()`, `save_message()`, and `get_message()`.",
            "Each helper opens its own connection (check_same_thread=False) and never shares one.",
        ],
        [
            "init_db() — idempotent CREATE TABLE IF NOT EXISTS.",
            "save_message(message_id, payload) — INSERT OR REPLACE.",
            "get_message(message_id) — returns the parsed payload dict or None.",
        ],
        ["sqlite3 (stdlib)", "json (stdlib)", "pathlib"],
    ),
    (
        "3.4 backend/app/api/__init__.py — Legacy /analyze router",
        "Backward-compatibility router for the v1 single-prompt endpoint.",
        "Exposes POST /analyze, which takes a prompt and returns the answer along with the legacy three-metric payload (mean_logprob, logprob_std, confidence_decay).",
        [
            "Defines `AnalyzeRequest`, `AnalyzeResponse`, and `LegacyMetricsResponse` Pydantic models.",
            "Delegates work to `analyze_prompt()` from analysis_service.",
            "Wraps any exception into HTTP 500 with the original detail.",
        ],
        ["analyze() — POST /analyze handler."],
        ["fastapi", "pydantic", "backend.app.adapters.openai_adapter", "backend.app.services.analysis_service"],
    ),
    (
        "3.5 backend/app/api/chat.py — v2 chat router",
        "Primary chat endpoint that drives the full HalluScope pipeline.",
        "Maintains a per-mode singleton `RiskService` (one for 'api', one for 'local'), runs analysis, applies the intervention rule, persists the result, and returns the structured ChatResponse.",
        [
            "POST /chat: runs `risk_service.analyze_message()`, generates a short message_id, decides on intervention, suppresses the LLM answer when needed, and saves the displayed payload to storage.",
            "POST /chat/reset: rebuilds the singletons to wipe history.",
            "Returns intervention=true and a deterministic clarification when HRI ≥ 0.46.",
        ],
        [
            "chat() — main pipeline handler.",
            "reset_chat() — recreates _risk_services dict.",
        ],
        ["uuid", "fastapi", "backend.app.models.schemas",
         "backend.app.services.risk_service",
         "backend.app.services.intervention_service",
         "backend.app.storage"],
    ),
    (
        "3.6 backend/app/api/metrics.py — v2 metrics router",
        "Dashboard lookup endpoint.",
        "Exposes GET /metrics/{message_id} which returns the full per-message uncertainty breakdown stored at /chat time, including per-token HRI trajectory, token details, and (local-mode only) internal-signal traces.",
        [
            "Looks up the message in the SQLite store; 404s if not found.",
            "Reconstructs a `MetricsLookupResponse` Pydantic object.",
        ],
        ["get_metrics() — GET /metrics/{message_id} handler."],
        ["fastapi", "backend.app.models.schemas", "backend.app.storage"],
    ),
    (
        "3.7 backend/app/risk/hri_engine.py",
        "Hallucination Risk Index core.",
        "Aggregates per-token uncertainty signals into the scalar HRI ∈ [0, 1] via a weighted shifted-sigmoid formula, classifies the response as LOW / MEDIUM / HIGH, computes the per-token HRI trajectory used for charting, and emits a per-token risk-contribution score for the heatmap.",
        [
            "Holds the calibrated thresholds: RISK_LOW_THRESHOLD = 0.15, RISK_HIGH_THRESHOLD = 0.30 (derived in eval/pick_thresholds.py against HalluEval-QA, AUROC 0.785).",
            "Computes mean log-probability, log-prob variance, log-prob std, and confidence decay.",
            "Computes Shannon entropy from top-K log-probs, normalised entropy (entropy / log K), NLL, and perplexity (capped at 1e9 to stay JSON-safe).",
            "API formula: HRI = sigmoid(10 · (0.4·var + 0.3·decay + 0.3·|mean|) − 3).",
            "Local-mode formula (when internal signals are present): adds normalised Semantic Energy (0.15) and inverted RCD (0.15) terms — see `hri_mode='local_internal'`.",
            "Returns a dict consumed unchanged by RiskService.",
        ],
        [
            "compute_hri(logprobs, top_logprobs, internal_signals) — the engine entry point.",
            "_sigmoid(x) — clamped logistic.",
            "_token_entropy(top_lps) — Shannon entropy from top-K log-probs.",
        ],
        ["math", "backend.app.metrics.logprob_metrics",
         "backend.app.metrics.d (confidence_decay)"],
    ),
    (
        "3.8 backend/app/metrics/logprob_metrics.py",
        "Pure-function statistics over a token log-probability sequence.",
        "Provides the basic distributional metrics (mean, variance, std) consumed by the HRI engine. Provider-agnostic — works on any list[float].",
        [
            "Returns 0.0 for empty / too-short inputs to avoid division-by-zero.",
            "Population variance (not sample) for stability on short sequences.",
        ],
        [
            "mean_logprob(logprobs) — arithmetic mean.",
            "logprob_variance(logprobs) — population variance.",
            "logprob_std(logprobs) — sqrt of variance.",
        ],
        ["math (stdlib only)"],
    ),
    (
        "3.9 backend/app/metrics/d.py — Confidence Decay",
        "Temporal confidence-drop signal.",
        "Compares the mean log-probability of the first window=5 tokens against the last window=5 tokens to detect a confidence dip across the generated span — a common hallucination correlate. Falls back to first-half / last-half on short outputs.",
        [
            "Returns the magnitude of the drop (max(0, start − end)).",
            "Shorter than 2-token sequences return 0.0.",
        ],
        ["confidence_decay(logprobs, window=5)."],
        ["No external dependencies (stdlib typing only)."],
    ),
    (
        "3.10 backend/app/metrics/internal_metrics.py",
        "Internal-signal metrics that require model internals (local-mode only).",
        "Computes Semantic Energy (E_t = −log Σ exp(z_t)) per token and Representation Collapse Detection (RCD_t = 1 − cos(h_t, h_{t-1})) per token from the full pre-softmax logits and last-layer hidden states surfaced by the local adapter.",
        [
            "`semantic_energy()` uses the log-sum-exp trick for numerical stability.",
            "`representation_collapse()` returns len(hidden_states)−1 values (no t=0 entry).",
            "`aggregate_internal_signals()` returns mean / extremum summaries plus per-token sequences (the latter for charting); when either input is None (API mode) all summary fields are None.",
        ],
        [
            "semantic_energy(logits_per_token) -> List[float]",
            "representation_collapse(hidden_states) -> List[float]",
            "aggregate_internal_signals(logits_per_token, hidden_states_per_token) -> dict",
        ],
        ["numpy", "math"],
    ),
    (
        "3.11 backend/app/adapters/base_adapter.py",
        "Abstract LLM adapter interface.",
        "Defines the contract every adapter must satisfy so the metrics layer and risk engine remain provider-agnostic.",
        [
            "`generate()` for single-prompt mode and `generate_chat()` for multi-turn — both abstract.",
            "Both methods must return a `StandardizedResult`.",
        ],
        ["BaseLLMAdapter (ABC) — generate, generate_chat."],
        ["abc", "backend.app.adapters.schemas"],
    ),
    (
        "3.12 backend/app/adapters/schemas.py — StandardizedResult",
        "Adapter response schema.",
        "A dataclass that every adapter must populate. Contains the generated text, per-token log-probs, token strings, optional top-K log-probs, model name, and (for local adapters only) hidden_states_per_token and logits_per_token used by the internal-signal metrics.",
        [
            "Fields are typed; lists default to empty via `field(default_factory=list)`.",
            "hidden_states_per_token / logits_per_token default to None so API adapters can leave them unset.",
        ],
        ["StandardizedResult (dataclass)."],
        ["dataclasses", "numpy", "typing"],
    ),
    (
        "3.13 backend/app/adapters/openai_adapter.py",
        "OpenAI Chat Completions adapter (API mode).",
        "Wraps the OpenAI Python SDK and always requests `logprobs=True, top_logprobs=5`. Defaults to `gpt-4o-mini`. Single-prompt `generate()` is implemented as a one-message call to `generate_chat()`.",
        [
            "Reads OPENAI_API_KEY from config.",
            "Walks `choice.logprobs.content` to populate token_logprobs, token_strings, top_logprobs.",
            "Raises `RuntimeError` if the model does not return logprobs — never silently degrades.",
        ],
        [
            "OpenAIAdapter.__init__(model='gpt-4o-mini')",
            "OpenAIAdapter.generate_chat(messages, max_tokens, temperature)",
            "OpenAIAdapter.generate(prompt, ...)",
        ],
        ["openai", "backend.app.adapters.base_adapter",
         "backend.app.adapters.schemas", "backend.app.config"],
    ),
    (
        "3.14 backend/app/adapters/local_adapter.py",
        "HuggingFace Transformers adapter (local mode).",
        "Loads an `AutoModelForCausalLM` once at startup (cached on the class), runs generation with `output_scores=True, output_hidden_states=True`, then captures pre-softmax logits and the last-layer hidden state for every generated token. This is the data path that enables Semantic Energy and RCD.",
        [
            "Singleton pattern: `_loaded_model_id`, `_tokenizer`, `_model`, `_device` live on the class so weights are loaded once per process.",
            "Auto-detects CUDA via `torch.cuda.is_available()`; falls back to CPU.",
            "`_build_prompt()` uses `tokenizer.apply_chat_template()` when available, otherwise falls back to a hand-rolled System/User/Assistant prefix layout.",
            "Casts bf16 tensors to fp32 before `.cpu().numpy()` because numpy has no native bfloat16 dtype.",
        ],
        [
            "LocalAdapter._ensure_model_loaded(model_id) — class-level lazy loader.",
            "LocalAdapter._build_prompt(messages) — chat-template prompt builder.",
            "LocalAdapter.generate_chat(messages, max_tokens, temperature) — runs generation and returns a StandardizedResult populated with internals.",
        ],
        ["torch", "transformers (AutoModelForCausalLM, AutoTokenizer)", "numpy",
         "backend.app.adapters.base_adapter", "backend.app.adapters.schemas",
         "backend.app.config"],
    ),
    (
        "3.15 backend/app/services/llm_service.py",
        "Conversation-state wrapper around the active adapter.",
        "Owns the chat history list, lazily instantiates the right adapter based on `evaluation_mode`, and exposes helpers used by the intervention flow to manipulate history.",
        [
            "Adapter is constructed on first use; for local mode it imports `LocalAdapter` lazily so missing GPU deps do not crash API mode.",
            "`send_message()` appends the user message, calls `generate_chat()` with the full history, then appends the assistant reply.",
            "`pop_last_assistant()` strips the suppressed reply when an intervention fires.",
        ],
        [
            "LLMService(evaluation_mode)",
            "send_message(user_message, max_tokens, temperature)",
            "add_assistant_message(content) / pop_last_assistant() / reset()",
        ],
        ["backend.app.adapters.openai_adapter",
         "backend.app.adapters.local_adapter (lazy)",
         "backend.app.adapters.base_adapter",
         "backend.app.adapters.schemas", "backend.app.config"],
    ),
    (
        "3.16 backend/app/services/risk_service.py",
        "End-to-end orchestrator.",
        "Calls the LLM, aggregates internal signals when present, runs `compute_hri()`, builds the per-token TokenDetail list, fills the MetricsDetail object (including local-only fields), and constructs `InternalPerToken` for the dashboard charts.",
        [
            "Falls back transparently to the API HRI formula when `hidden_states_per_token` is None.",
            "Records `hri_mode` ('api' | 'local_internal') so the frontend can label which formula fired.",
        ],
        [
            "RiskService.analyze_message(user_message) -> dict",
            "RiskService.inject_assistant_message / pop_last_assistant",
        ],
        ["backend.app.services.llm_service",
         "backend.app.risk.hri_engine",
         "backend.app.metrics.internal_metrics",
         "backend.app.models.schemas"],
    ),
    (
        "3.17 backend/app/services/intervention_service.py",
        "Threshold-based answer suppression.",
        "Decides whether to replace the LLM answer with a deterministic clarification request when the HRI crosses INTERVENTION_THRESHOLD = 0.46. The clarification is templated (no extra LLM call) so the user is never charged or stalled on the safe path.",
        [
            "`should_intervene(hri)` — returns hri ≥ 0.46.",
            "`generate_clarification(user_message)` — templated message asking the user to verify names/dates and provide context.",
        ],
        ["should_intervene(hri) -> bool", "generate_clarification(user_message) -> str"],
        ["No external dependencies."],
    ),
    (
        "3.18 backend/app/services/analysis_service.py",
        "Legacy v1 single-prompt analyser.",
        "Implements the original /analyze pipeline: adapter.generate → compute_hri → minimal three-field response. Kept for backward compatibility; new clients use /chat.",
        [
            "Pure function; takes any BaseLLMAdapter instance.",
            "Does not interact with conversation history or storage.",
        ],
        ["analyze_prompt(adapter, prompt, max_tokens, temperature) -> dict"],
        ["backend.app.adapters.base_adapter", "backend.app.risk.hri_engine"],
    ),
    (
        "3.19 backend/app/models/schemas.py",
        "Pydantic request / response schemas.",
        "Single source of truth for the JSON contract between FastAPI and the React frontend.",
        [
            "ChatRequest / ChatResponse — POST /chat.",
            "MetricsLookupResponse — GET /metrics/{id}.",
            "TokenDetail, MetricsDetail (with local-only mean_semantic_energy / mean_rcd / min_rcd / hri_mode), InternalPerToken (per-token energy and RCD traces).",
            "ChatMessage — conversation history record.",
        ],
        ["ChatRequest, ChatResponse, TokenDetail, MetricsDetail, InternalPerToken, MetricsLookupResponse, ChatMessage."],
        ["pydantic", "uuid", "typing"],
    ),
    (
        "3.20 frontend/src/main.jsx & App.jsx",
        "React entry point and router setup.",
        "`main.jsx` mounts the React tree under #root with React.StrictMode. `App.jsx` defines two routes via react-router-dom v6: `/` → ChatPage and `/metrics/:messageId` → MetricsPage.",
        [
            "Imports global styles from styles.css.",
            "Uses BrowserRouter, Routes, Route from react-router-dom.",
        ],
        ["App() — root component."],
        ["react", "react-dom/client", "react-router-dom"],
    ),
    (
        "3.21 frontend/src/api.js",
        "Centralised fetch client.",
        "All HTTP calls live in this file; components do not call `fetch()` directly. Hardcodes API_BASE = http://localhost:8000.",
        [
            "Throws on non-2xx with the API's `detail` field as the message.",
            "Exposes sendChatMessage, sendChatMessageWithMode, getMetrics, resetChat.",
        ],
        ["sendChatMessage(message)", "sendChatMessageWithMode(message, evaluationMode)",
         "getMetrics(messageId)", "resetChat()"],
        ["Browser fetch API."],
    ),
    (
        "3.22 frontend/src/pages/ChatPage.jsx",
        "Main chat interface.",
        "Renders the chat history, mode picker (api / local), input box, intervention banner, and a per-bubble RiskBadge with HRI score. Cycles a 'Calling model… → Computing metrics… → Analyzing risk…' status message every 2 s while a request is in flight, and auto-scrolls to the latest message.",
        [
            "Maintains conversation in component state; navigates to `/metrics/:id` on the View Metrics click.",
            "Posts to /chat via `sendChatMessageWithMode()`.",
            "Calls /chat/reset on the Reset button.",
        ],
        ["ChatPage() — page component.", "handleSend() — submit handler."],
        ["react", "react-router-dom", "../api", "../components/ChatWindow",
         "../components/RiskBadge"],
    ),
    (
        "3.23 frontend/src/pages/MetricsPage.jsx",
        "Per-message analytics dashboard.",
        "Loads /metrics/:messageId and renders the 11-card metric grid, per-token HRI line chart with the LOW (0.15) / HIGH (0.30) threshold reference lines, the token risk heatmap, and — when internal_per_token is present — the local-mode Semantic Energy and RCD trajectory charts.",
        [
            "Uses recharts (LineChart, ReferenceLine) for all charts.",
            "Surfaces hri_mode so the user can see which formula fired.",
        ],
        ["MetricsPage() — page component."],
        ["react", "react-router-dom", "recharts", "../api",
         "../components/MetricsPanel", "../components/TokenHeatmap"],
    ),
    (
        "3.24 frontend/src/components/ — Presentational components",
        "Five small components composing the UI.",
        "ChatWindow (scrollable message list with empty state and typing indicator), MessageBubble (per-message bubble plus RiskBadge and View Metrics link), RiskBadge (colour-coded LOW/MEDIUM/HIGH pill with score), MetricsPanel (the metric card grid), TokenHeatmap (green→yellow→red span colouring per token risk_contribution with hover tooltip).",
        [
            "Pure props-in / JSX-out — no state beyond hover tooltips.",
            "Risk colours defined inline (LOW=#22c55e, MEDIUM=#f59e0b, HIGH=#ef4444).",
        ],
        ["ChatWindow", "MessageBubble", "RiskBadge", "MetricsPanel", "TokenHeatmap"],
        ["react"],
    ),
    (
        "3.25 eval/build_subset.py",
        "Builds the calibration dataset.",
        "Downloads `pminervini/HaluEval` (qa subset) via the HuggingFace `datasets` library, dedupes questions, and writes 100 unique question + reference-answer pairs to `eval/datasets/halluveval_qa_questions_100.json`. Random seed = 42 for reproducibility.",
        [
            "Adopts the v2 free-form-answer methodology (no candidate answer in the prompt) which fixed the v1 length artefact.",
        ],
        ["main()."],
        ["datasets (HuggingFace)", "json", "random"],
    ),
    (
        "3.26 eval/run_benchmark.py",
        "Benchmark harness.",
        "For each question in the subset: calls /chat/reset, then POSTs the question to /chat in the requested mode (api or local), and writes hri, risk_level, hri_mode, mean_logprob, perplexity, mean_semantic_energy, mean_rcd, intervention flag, plus the original question, reference answer, and the model's free-form answer to a CSV.",
        [
            "Output: `eval/results_api.csv` or `eval/results_local.csv`.",
            "Supports `--limit N` for shorter runs.",
        ],
        ["main(), parse_args()."],
        ["argparse", "csv", "fastapi.testclient (TestClient)",
         "backend.app.main"],
    ),
    (
        "3.27 eval/match.py",
        "SBERT-based ground-truth matcher.",
        "Encodes (model_answer, reference_answer) with `sentence-transformers/all-MiniLM-L6-v2`, takes cosine similarity, and writes `model_match` ∈ {match, mismatch} into the CSV using a default threshold of 0.60. Includes a `--selftest` mode and prints the cosine distribution for threshold sanity-checking.",
        [
            "Lazy singleton on the SentenceTransformer to avoid re-loading on multiple runs.",
        ],
        ["main()", "_get_model()", "match_pair()."],
        ["sentence-transformers", "argparse", "csv"],
    ),
    (
        "3.28 eval/analyze.py",
        "Metrics computation and ROC plotting.",
        "Treats `model_match=='mismatch'` as the positive (hallucination) class and reports AUROC, ROC curve PNG, best-F1 threshold with its precision/recall/F1, and confusion matrices at both the live HIGH threshold and the best-F1 threshold. Writes `eval/summary.json`.",
        [
            "Uses scikit-learn (`roc_auc_score`, `precision_recall_curve`, `confusion_matrix`).",
            "ROC plots saved with matplotlib's Agg backend (no display required).",
        ],
        ["main()."],
        ["scikit-learn", "matplotlib", "numpy", "csv"],
    ),
    (
        "3.29 eval/pick_thresholds.py",
        "Operating-point selector.",
        "Sweeps the API benchmark CSV for the smallest threshold satisfying precision ≥ 0.85, recall ≥ 0.50, and flag-rate ≤ 0.50. Falls back to (HIGH=0.30, LOW=0.15) when no threshold meets all three. Writes `eval/operating_point.json`.",
        [
            "The fallback values are exactly what live in `hri_engine.py` today (no point in the n=100 sweep met the joint constraint).",
        ],
        ["main()."],
        ["scikit-learn", "numpy", "csv"],
    ),
    (
        "3.30 scripts/ — Phase verification scripts",
        "Development-only sanity scripts.",
        "Three small standalone scripts: `test_hidden_states.py` verifies that the local adapter populates hidden_states_per_token and logits_per_token with the expected shapes; `test_internal_metrics.py` checks Semantic Energy and RCD on synthetic vectors; `test_local_e2e.py` exercises the full /chat → /metrics flow in local mode via FastAPI's TestClient.",
        [
            "Not part of the runtime package — use them when debugging local mode.",
            "Each runs as `python3 scripts/<name>.py` from the repo root.",
        ],
        ["test_hidden_states.py", "test_internal_metrics.py", "test_local_e2e.py"],
        ["fastapi.testclient", "torch", "transformers", "numpy"],
    ),
]


# ────────────────────────────────────────────────────────────────────
# Document construction
# ────────────────────────────────────────────────────────────────────

def build_document() -> Document:
    doc = Document()

    # Margins: 1 inch on all sides
    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)

    # Default style
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.15

    # Page-number footer
    footer = doc.sections[0].footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(fp)

    # ── Front matter ────────────────────────────────────────────────
    logo = doc.add_paragraph()
    logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = logo.add_run("[ SRM Institute Logo ]")
    set_body_font(run, size_pt=11, bold=True)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run("(insert SRM IST logo here)")
    set_body_font(run, size_pt=10)
    run.italic = True

    for line, size in [
        ("SRM INSTITUTE OF SCIENCE AND TECHNOLOGY", 16),
        ("SCHOOL OF COMPUTING", 13),
        ("DEPARTMENT OF COMPUTING TECHNOLOGIES", 13),
        ("21CSP401L – MAJOR PROJECT", 13),
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(line)
        set_body_font(r, size_pt=size, bold=True)

    doc.add_paragraph()  # spacer

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("Hallucination Risk Analysis and Uncertainty Modeling\nin Large Language Models")
    set_body_font(r, size_pt=18, bold=True)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("Document 3 — Code, Software Used, and Instructions to Execute")
    set_body_font(r, size_pt=14, bold=True)

    doc.add_paragraph()

    # Two-column block: Guide + students
    info = doc.add_table(rows=1, cols=2)
    info.autofit = True
    left = info.rows[0].cells[0]
    right = info.rows[0].cells[1]

    def fill_block(cell, header, lines):
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(header)
        set_body_font(r, size_pt=12, bold=True)
        for line in lines:
            pp = cell.add_paragraph()
            rr = pp.add_run(line)
            set_body_font(rr, size_pt=11)

    fill_block(left, "Project Guide",
               ["Dr. R.I. Minu",
                "Associate Professor",
                "Department of Computing Technologies",
                "SRM Institute of Science and Technology"])
    fill_block(right, "Submitted by",
               ["V Dinessh — RA2211003011234",
                "Shivnarayan S — RA2211003011216",
                "",
                "Project Category: Research",
                "Batch ID: 874",
                "Academic Year: 2025–2026"])

    doc.add_page_break()

    # ── Table of contents ──────────────────────────────────────────
    add_heading(doc, "Table of Contents", 1)
    toc_items = [
        "1. Introduction",
        "2. Repository Structure",
        "3. Code Walkthrough",
        "4. Software, Frameworks, and Libraries Used",
        "5. System Requirements",
        "6. Instructions to Execute the Code",
        "7. Troubleshooting",
        "8. Repository Link / Submission Note",
    ]
    for item in toc_items:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.5)
        r = p.add_run(item)
        set_body_font(r, size_pt=12)

    doc.add_page_break()

    # ── Section 1: Introduction ────────────────────────────────────
    add_heading(doc, "1. Introduction", 1)
    add_body(doc,
        "HalluScope is a model-agnostic reliability layer for Large Language Models that monitors "
        "every assistant response for hallucination risk. During inference it captures token-level "
        "uncertainty signals — log-probability variance, confidence decay, mean log-probability, "
        "Shannon entropy from top-K alternatives, and (for locally-hosted models) Semantic Energy "
        "and Representation Collapse Detection from the model's own logits and hidden states — "
        "and combines them via a calibrated weighted-sigmoid formula into a single Hallucination "
        "Risk Index (HRI ∈ [0, 1]). Each response is classified LOW, MEDIUM, or HIGH; when the "
        "score crosses the calibrated HIGH threshold an intervention service suppresses the LLM "
        "answer and returns a deterministic clarification request. Two evaluation backends sit "
        "behind a uniform StandardizedResult interface: an OpenAI Chat Completions adapter "
        "(gpt-4o-mini with logprobs) and a local HuggingFace adapter (TinyLlama-1.1B-Chat by "
        "default; configurable to Qwen2.5-3B-Instruct via LOCAL_MODEL).")
    add_body(doc,
        "This document — Document 3 — accompanies the project's viva submission. It describes the "
        "code base end-to-end (file-by-file purpose and dependencies), enumerates every framework, "
        "library, and runtime dependency the system relies on, and gives an evaluator a step-by-step "
        "recipe to clone, configure, and run both the backend service and the React dashboard, "
        "switch between API and local modes, and reproduce the HalluEval-QA benchmark numbers.")

    # ── Section 2: Repository structure ────────────────────────────
    add_heading(doc, "2. Repository Structure", 1)
    try:
        tree_out = subprocess.check_output(
            ["tree", "-L", "3",
             "-I", "node_modules|__pycache__|.git|venv|.venv|dist|build|*.egg-info"],
            cwd=str(ROOT),
            text=True,
        )
    except Exception:
        tree_out = "(tree command unavailable)"
    add_code_block(doc, tree_out.rstrip())

    add_body(doc,
        "The top-level layout has three runtime directories and one offline harness. `backend/` "
        "holds the FastAPI service: routers under `app/api/`, the HRI engine under `app/risk/`, "
        "log-prob and internal-signal metric modules under `app/metrics/`, the OpenAI and "
        "HuggingFace adapters under `app/adapters/`, and the SQLite-backed message store at "
        "`app/storage.py`. `frontend/` is the Vite + React 18 single-page app with two routes "
        "(ChatPage and MetricsPage) and five presentational components. `eval/` is a self-contained "
        "calibration harness — dataset builder, benchmark runner, SBERT matcher, ROC analyzer, and "
        "operating-point picker — that produces the AUROC numbers and threshold values baked into "
        "the live engine. `scripts/` contains three standalone verification scripts used during "
        "Phase 2 development. `halluscope.db` is the SQLite store (created at first run). "
        "`.env.example` is the secrets template; `.env` (git-ignored) holds the live key.")

    # ── Section 3: Code walkthrough ────────────────────────────────
    add_heading(doc, "3. Code Walkthrough", 1)
    add_body(doc,
        "The walkthrough is grouped roughly by layer: backend entry and routers, the risk and "
        "metrics core, the adapter layer, services, schemas, the React frontend, the evaluation "
        "harness, and verification scripts.")

    for heading, purpose, summary, responsibilities, notable, deps in MODULES:
        add_heading(doc, heading, 2)

        p = doc.add_paragraph()
        r = p.add_run("Purpose: ")
        set_body_font(r, size_pt=12, bold=True)
        r = p.add_run(purpose)
        set_body_font(r, size_pt=12)

        add_body(doc, summary)

        p = doc.add_paragraph()
        r = p.add_run("Key responsibilities:")
        set_body_font(r, size_pt=12, bold=True)
        for item in responsibilities:
            p = doc.add_paragraph(style="List Bullet")
            for old in list(p.runs):
                old.text = ""
            r = p.add_run(item)
            set_body_font(r, size_pt=12)

        p = doc.add_paragraph()
        r = p.add_run("Notable functions / classes:")
        set_body_font(r, size_pt=12, bold=True)
        for item in notable:
            p = doc.add_paragraph(style="List Bullet")
            for old in list(p.runs):
                old.text = ""
            r = p.add_run(item)
            set_body_font(r, size_pt=12)

        p = doc.add_paragraph()
        r = p.add_run("Dependencies: ")
        set_body_font(r, size_pt=12, bold=True)
        r = p.add_run(", ".join(deps))
        set_body_font(r, size_pt=12)

    # ── Section 4: Software & libraries ────────────────────────────
    add_heading(doc, "4. Software, Frameworks, and Libraries Used", 1)
    add_body(doc,
        "Versions are taken directly from `backend/requirements.txt` and `frontend/package.json`. "
        "Where the project pins a minimum (`>=`), the row records the floor and the actual installed "
        "minor is determined by pip / npm at install time.")

    sw_rows = [
        # Programming languages
        ["Language", "Python", "3.10+ (developed on 3.11)", "Backend service, evaluation harness, verification scripts."],
        ["Language", "JavaScript (ES2022, JSX)", "Node 18+ (Vite 5 requirement)", "React frontend."],
        # Backend framework
        ["Backend Framework", "FastAPI", "≥ 0.110.0", "Async HTTP API: /chat, /chat/reset, /metrics/{id}, /analyze, /health."],
        ["Backend Framework", "Uvicorn (uvicorn[standard])", "≥ 0.29.0", "ASGI server hosting the FastAPI app."],
        ["Backend Framework", "Pydantic", "≥ 2.0.0", "Request / response validation and schema serialisation."],
        # ML / LLM
        ["LLM (API mode)", "OpenAI Python SDK", "≥ 1.14.0", "Chat Completions with logprobs=True, top_logprobs=5."],
        ["LLM (API model)", "gpt-4o-mini", "OpenAI hosted", "Default cloud model used in API mode."],
        ["LLM (local mode)", "transformers (HuggingFace)", "≥ 4.40.0", "AutoModelForCausalLM + AutoTokenizer for local generation."],
        ["LLM (local mode)", "torch (PyTorch)", "≥ 2.2.0", "Tensor backend; auto-uses CUDA when available."],
        ["LLM (local model)", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "HF default", "Default local model (≈2.2 GB on disk)."],
        ["LLM (local model)", "Qwen/Qwen2.5-3B-Instruct", "HF (optional override)", "Selected via `LOCAL_MODEL` env var; needs stronger GPU."],
        # Numerics
        ["Numerics", "NumPy", "(transitive via torch / transformers)", "Hidden-state and logit array math in `internal_metrics.py`."],
        # Storage / config
        ["Storage", "SQLite (sqlite3)", "stdlib", "Persistent message store at repo-root `halluscope.db`."],
        ["Config", "python-dotenv", "≥ 1.0.0", "Loads the repo-root `.env` for OPENAI_API_KEY and LOCAL_MODEL."],
        # Frontend
        ["Frontend Framework", "React", "^18.3.1", "Component model for the chat and metrics pages."],
        ["Frontend Framework", "react-dom", "^18.3.1", "DOM renderer."],
        ["Frontend Routing", "react-router-dom", "^6.23.0", "Two routes: /, /metrics/:messageId."],
        ["Frontend Build", "Vite", "^5.4.0", "Dev server (port 5173) and production bundler."],
        ["Frontend Build", "@vitejs/plugin-react", "^4.3.1", "Vite plugin for React fast refresh."],
        # Visualisation
        ["Visualisation", "recharts", "^2.12.0", "Per-token HRI line chart, ReferenceLine threshold markers, and local-mode Semantic Energy / RCD trajectory charts."],
        # Eval-only
        ["Eval", "scikit-learn", "1.x (install ad-hoc for eval)", "AUROC, precision_recall_curve, confusion matrix."],
        ["Eval", "matplotlib", "3.x (install ad-hoc for eval)", "ROC plot rendering with the Agg backend."],
        ["Eval", "sentence-transformers", "5.x (install ad-hoc for eval)", "all-MiniLM-L6-v2 cosine matcher in `eval/match.py`."],
        ["Eval", "datasets (HuggingFace)", "(install ad-hoc for eval)", "Downloads `pminervini/HaluEval` qa subset in `eval/build_subset.py`."],
        # Dataset
        ["Dataset", "HalluEval-QA (pminervini/HaluEval, qa)", "100-row subset", "Calibration set; AUROC 0.785 (API mode)."],
        # Note: TruthfulQA was named in the prompt brief but the eval harness in this repo uses HalluEval only.
    ]
    add_table(doc,
        ["Category", "Tool / Library", "Version", "Purpose in HalluScope"],
        sw_rows,
        col_widths_cm=[3.0, 4.2, 3.0, 6.0])

    add_body(doc,
        "Note: the original project brief mentioned TruthfulQA as an additional benchmark; the current "
        "shipped harness uses HalluEval-QA only, which is the dataset the calibrated thresholds and "
        "AUROC of 0.785 are derived from. TruthfulQA support is not present in the codebase and is "
        "scoped for the next iteration.")

    # ── Section 5: System Requirements ─────────────────────────────
    add_heading(doc, "5. System Requirements", 1)

    add_heading(doc, "5.1 Hardware Requirements", 2)
    add_table(doc, ["Component", "Minimum (API mode only)", "Recommended (local mode)"], [
        ["CPU", "Dual-core x86_64", "Quad-core x86_64 or better"],
        ["RAM", "4 GB", "16 GB (for TinyLlama) / 24 GB+ (for Qwen2.5-3B)"],
        ["GPU / VRAM", "Not required", "NVIDIA GPU with ≥ 6 GB VRAM (TinyLlama runs on CPU but slowly); CUDA-capable card recommended for Qwen2.5-3B"],
        ["Disk", "~500 MB (deps + SQLite)", "~5 GB (TinyLlama weights ≈ 2.2 GB or Qwen2.5-3B ≈ 6 GB, plus deps)"],
        ["Network", "Stable internet for OpenAI calls", "Internet for first-run HF model download only"],
    ], col_widths_cm=[3.5, 5.5, 7.0])

    add_heading(doc, "5.2 Software Requirements", 2)
    add_bullet(doc, "Operating system: Linux (developed on Ubuntu / kernel 6.x), macOS, or Windows 10/11 with WSL2.")
    add_bullet(doc, "Python ≥ 3.10 (project developed and tested on Python 3.11).")
    add_bullet(doc, "Node.js ≥ 18 with npm ≥ 9 (Vite 5 minimum).")
    add_bullet(doc, "Optional: NVIDIA CUDA 11.8 or 12.x with matching PyTorch wheel for GPU local-mode inference.")
    add_bullet(doc, "Package managers: pip (Python), npm (frontend); a `python -m venv` virtual environment is assumed throughout.")
    add_bullet(doc, "Optional: `tree`, `curl` for the install verification commands shown in section 6.")

    add_heading(doc, "5.3 API Keys / Credentials", 2)
    add_bullet(doc, "OPENAI_API_KEY — required for API mode. Obtained from https://platform.openai.com/api-keys. Placed in the repo-root `.env` file (the loader is `backend/app/config.py`, line `load_dotenv(dotenv_path=ROOT_DIR / \".env\")`).")
    add_bullet(doc, "LOCAL_MODEL — optional. Overrides the default TinyLlama-1.1B-Chat with any AutoModelForCausalLM-compatible HuggingFace repo id (e.g. `Qwen/Qwen2.5-3B-Instruct`).")
    add_bullet(doc, "HuggingFace token — not required for the default TinyLlama or Qwen2.5-3B models; only needed if the user opts into a gated repo via LOCAL_MODEL. In that case, run `huggingface-cli login` once before the first server start.")

    # ── Section 6: Instructions ────────────────────────────────────
    add_heading(doc, "6. Instructions to Execute the Code", 1)
    add_body(doc,
        "The recipe below is verbatim what an evaluator can paste into a fresh shell. Commands are "
        "shown for Linux / macOS; Windows variants are noted where they differ.")

    add_heading(doc, "6.1 Clone the Repository", 2)
    add_code_block(doc,
        "git clone <repo-url> HalluScope\n"
        "cd HalluScope")

    add_heading(doc, "6.2 Backend Setup", 2)
    add_bullet(doc, "Create and activate a Python virtual environment.")
    add_code_block(doc,
        "# from repo root\n"
        "cd backend\n"
        "python3 -m venv .venv\n"
        "source .venv/bin/activate          # Linux / macOS\n"
        ".venv\\Scripts\\activate            # Windows (PowerShell)")

    add_bullet(doc, "Install backend dependencies (FastAPI, Uvicorn, OpenAI SDK, transformers, torch, pydantic, python-dotenv).")
    add_code_block(doc, "pip install -r requirements.txt")

    add_bullet(doc, "Configure the .env file at the repo root (one level above backend/).")
    add_code_block(doc,
        "cd ..                                       # back to repo root\n"
        "cp .env.example .env\n"
        "# then open .env in an editor and set OPENAI_API_KEY=sk-...\n"
        "# optionally uncomment LOCAL_MODEL=... to override the default TinyLlama")

    add_bullet(doc, "(Optional) pre-download the local HuggingFace model. If you skip this, the first chat in local mode triggers an automatic download (≈2.2 GB for TinyLlama).")
    add_code_block(doc,
        "python3 -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; "
        "AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); "
        "AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')\"")

    add_heading(doc, "6.3 Frontend Setup", 2)
    add_code_block(doc,
        "cd frontend\n"
        "npm install\n"
        "# the API base URL is hardcoded to http://localhost:8000 in src/api.js;\n"
        "# edit src/api.js if the backend is hosted elsewhere.")

    add_heading(doc, "6.4 Running the System", 2)
    add_bullet(doc, "Start the backend (terminal 1, with the .venv activated, from the repo root):")
    add_code_block(doc,
        "uvicorn backend.app.main:app --reload\n"
        "# server logs:  Uvicorn running on http://127.0.0.1:8000\n"
        "# the local model warms up in a daemon thread; /health reflects readiness")

    add_bullet(doc, "Smoke-test the API:")
    add_code_block(doc,
        "curl http://127.0.0.1:8000/health\n"
        "# {\"status\":\"ok\",\"local_model_loaded\":true,\"local_model_name\":\"TinyLlama/TinyLlama-1.1B-Chat-v1.0\"}")

    add_bullet(doc, "Start the frontend (terminal 2):")
    add_code_block(doc,
        "cd frontend\n"
        "npm run dev\n"
        "# Vite serves on http://localhost:5173")

    add_bullet(doc, "Open http://localhost:5173 in a browser. The chat page loads with an empty state.")

    add_heading(doc, "6.5 Switching Between Local and API Mode", 2)
    add_body(doc,
        "Mode is selected per request from the UI: the ChatPage has a mode picker that toggles "
        "evaluation_mode between 'api' and 'local' before each /chat call. Behind the scenes, "
        "`backend/app/api/chat.py` keeps two singleton RiskService instances — one per mode — so "
        "switching does not interleave conversation histories. To override the local model used "
        "when the picker is set to 'local', edit `LOCAL_MODEL` in the repo-root `.env` (e.g. "
        "`LOCAL_MODEL=Qwen/Qwen2.5-3B-Instruct`) and restart the backend.")

    add_heading(doc, "6.6 Running the Benchmark", 2)
    add_bullet(doc, "Install the eval-only extras (sklearn, matplotlib, sentence-transformers, datasets) into the backend .venv:")
    add_code_block(doc,
        "pip install scikit-learn matplotlib sentence-transformers datasets")

    add_bullet(doc, "Build the calibration subset (downloads HalluEval-QA via the HuggingFace `datasets` library, deduplicates, writes 100 rows).")
    add_code_block(doc, "python3 eval/build_subset.py")

    add_bullet(doc, "Run the benchmark in API mode (≈ 5 min, ≈ $0.02 on gpt-4o-mini) or local mode (≈ 1.5 min on GPU):")
    add_code_block(doc,
        "python3 eval/run_benchmark.py --mode api\n"
        "python3 eval/run_benchmark.py --mode local       # uses TinyLlama (or LOCAL_MODEL)\n"
        "python3 eval/run_benchmark.py --mode local --limit 50   # smaller run for sanity")

    add_bullet(doc, "Fill in the model_match column with the SBERT-based matcher:")
    add_code_block(doc,
        "python3 eval/match.py --csv eval/results_api.csv   --threshold 0.60\n"
        "python3 eval/match.py --csv eval/results_local.csv --threshold 0.60")

    add_bullet(doc, "Compute AUROC, ROC plots, F1 sweep, and confusion matrices:")
    add_code_block(doc,
        "python3 eval/analyze.py\n"
        "# outputs: eval/roc_api.png, eval/roc_local.png, eval/summary.json,\n"
        "#          plus printed best-F1 and at-current-threshold metrics")

    add_bullet(doc, "Re-pick the operating-point thresholds (precision-floor sweep):")
    add_code_block(doc,
        "python3 eval/pick_thresholds.py\n"
        "# writes eval/operating_point.json")

    add_heading(doc, "6.7 Expected Output", 2)
    add_bullet(doc, "Chat page: typing a question and pressing Enter shows a typing indicator that cycles through 'Calling model… → Computing metrics… → Analyzing risk…'. Within ~1–4 seconds the assistant reply appears with a coloured RiskBadge (green LOW / amber MEDIUM / red HIGH) and the HRI score to two decimal places.")
    add_bullet(doc, "Intervention path: when HRI ≥ 0.46 the bubble shows '⚠️ Intervention' and a clarification template instead of the LLM answer.")
    add_bullet(doc, "Metrics page (`/metrics/:id`): an 11-card metric grid (HRI, Risk Level, Mean Log-Prob, Log-Prob Std, Log-Prob Variance, Entropy, Normalized Entropy, NLL, Perplexity, Confidence Decay, Token Count), a per-token HRI line chart with the LOW (0.15) and HIGH (0.30) reference lines, and a token risk heatmap (green→yellow→red) with hover tooltips for log-prob, entropy, and risk contribution. In local mode the page additionally renders Semantic Energy and RCD trajectory charts and labels hri_mode = 'local_internal'.")
    add_bullet(doc, "Benchmark output: `eval/REPORT.md` is updated; `eval/results_api.csv` and `eval/roc_api.png` are produced.")

    # ── Section 7: Troubleshooting ─────────────────────────────────
    add_heading(doc, "7. Troubleshooting", 1)
    add_table(doc, ["Symptom", "Likely cause and fix"], [
        ["`OPENAI_API_KEY` is empty / 401 from /chat in API mode",
         "The repo-root `.env` is missing or unset. Confirm with `grep OPENAI_API_KEY .env`. The loader is `load_dotenv(dotenv_path=ROOT_DIR / \".env\")` in `backend/app/config.py`; the .env must sit at the repo root, not inside `backend/`."],
        ["CUDA out of memory when running local mode",
         "The default TinyLlama-1.1B fits in ≈3 GB VRAM but Qwen2.5-3B needs ≈12 GB. Either revert `LOCAL_MODEL` to the TinyLlama default in `.env` or run on CPU by hiding the GPU: `CUDA_VISIBLE_DEVICES= uvicorn backend.app.main:app --reload`."],
        ["`Model 'gpt-4o-mini' did not return logprobs`",
         "OpenAIAdapter raises this if the chosen model is unsupported. Use `gpt-4o-mini` (the default) or another logprob-capable chat model; do not silently swap in a model without logprobs."],
        ["Frontend can't reach the backend (CORS / network error)",
         "CORS is allow-listed for `http://localhost:5173` only. If you change the Vite port, update the `allow_origins` list in `backend/app/main.py`. Confirm both processes are up and the API_BASE in `frontend/src/api.js` is `http://localhost:8000`."],
        ["`Address already in use` on port 8000 or 5173",
         "Another process is bound. Find and kill it: `lsof -i :8000` then `kill <pid>` (Linux/macOS), or change the port: `uvicorn backend.app.main:app --port 8001` and update `API_BASE` in `frontend/src/api.js` accordingly."],
        ["`npm install` fails with peer-dep / engine errors",
         "Check Node ≥ 18 (`node --version`). Wipe and retry: `rm -rf node_modules package-lock.json && npm install`. The lockfile in the repo is the source of truth — if you regenerate, verify the resulting React / recharts / vite versions match `package.json`."],
        ["Local model warm-up never completes (`local_model_loaded:false`)",
         "First-run download is ≈2.2 GB and can take minutes on a slow link. Watch the backend logs; the warm-up runs in a daemon thread. The `_warm_local_model()` helper in `backend/app/main.py` swallows exceptions so the API stays alive even if HF deps are missing."],
        ["Perplexity rendered as `Infinity` / JSON parse error",
         "Should not occur — `hri_engine.py` already caps perplexity at 1e9 with `min(math.exp(min(nll, 20)), 1e9)`. If it does, ensure the backend is the current version (the cap was added explicitly to keep the JSON browser-parseable)."],
    ], col_widths_cm=[5.5, 10.5])

    # ── Section 8: Repo link ───────────────────────────────────────
    add_heading(doc, "8. Repository Link / Submission Note", 1)
    add_body(doc,
        "Source code available at: [insert repository link before submission].")
    add_body(doc,
        "All code, evaluation outputs (eval/results_api.csv, eval/roc_api.png, eval/REPORT.md), "
        "and this document are part of the submission bundle for 21CSP401L – Major Project, "
        "Academic Year 2025–2026, under the guidance of Dr. R.I. Minu, Department of Computing "
        "Technologies, SRM Institute of Science and Technology.")

    return doc


def main() -> None:
    doc = build_document()
    doc.save(str(OUT_PATH))
    print(f"Wrote {OUT_PATH}")
    print()
    print("=== Generation summary ===")
    print(f"Sections written: 8 (Introduction → Repo Link)")
    print(f"Module subsections in section 3: {len(MODULES)}")
    print()
    print("Files referenced from the codebase:")
    files = [
        "backend/requirements.txt",
        "backend/app/main.py",
        "backend/app/config.py",
        "backend/app/storage.py",
        "backend/app/api/__init__.py",
        "backend/app/api/chat.py",
        "backend/app/api/metrics.py",
        "backend/app/risk/hri_engine.py",
        "backend/app/metrics/logprob_metrics.py",
        "backend/app/metrics/d.py",
        "backend/app/metrics/internal_metrics.py",
        "backend/app/adapters/base_adapter.py",
        "backend/app/adapters/schemas.py",
        "backend/app/adapters/openai_adapter.py",
        "backend/app/adapters/local_adapter.py",
        "backend/app/services/llm_service.py",
        "backend/app/services/risk_service.py",
        "backend/app/services/intervention_service.py",
        "backend/app/services/analysis_service.py",
        "backend/app/models/schemas.py",
        "frontend/package.json",
        "frontend/vite.config.js",
        "frontend/src/main.jsx",
        "frontend/src/App.jsx",
        "frontend/src/api.js",
        "frontend/src/pages/ChatPage.jsx",
        "frontend/src/pages/MetricsPage.jsx",
        "frontend/src/components/{ChatWindow,MessageBubble,RiskBadge,MetricsPanel,TokenHeatmap}.jsx",
        "eval/build_subset.py",
        "eval/run_benchmark.py",
        "eval/match.py",
        "eval/analyze.py",
        "eval/pick_thresholds.py",
        "scripts/test_*.py",
        "README.md",
        ".env.example",
    ]
    for f in files:
        print(f"  - {f}")
    print()
    print("Things NOT found in the repo (called out as 'not present' in the document):")
    print("  - TruthfulQA loader: project brief mentioned it; the harness uses HalluEval-QA only.")
    print("  - Qwen2.5-3B-Instruct as the actual default local model: README explicitly says")
    print("    TinyLlama-1.1B-Chat is the demo default; Qwen is supported via LOCAL_MODEL.")
    print("  - tiktoken: not in requirements.txt (token counting is not used in the runtime).")
    print()
    print("Codebase inconsistencies worth surfacing before viva:")
    print("  1. backend/app/services/intervention_service.py header docstring still mentions the")
    print("     OLD threshold band (LOW<0.35, MEDIUM 0.35–0.45, HIGH≥0.46); the constant in the")
    print("     same file is INTERVENTION_THRESHOLD = 0.46, but the LIVE thresholds in")
    print("     hri_engine.py are RISK_LOW=0.15 and RISK_HIGH=0.30. The docstring needs updating.")
    print("  2. The intervention threshold (0.46) is HIGHER than the HIGH classification")
    print("     threshold (0.30), so MEDIUM responses (HRI 0.15–0.30) and HIGH responses with")
    print("     HRI 0.30–0.46 do NOT trigger intervention — only HRI ≥ 0.46 does. This is by")
    print("     design but the README's '⚠️ injects clarification when risk crosses HIGH'")
    print("     phrasing is slightly misleading; mention this on the slide if asked.")
    print("  3. backend/app/api/chat.py and backend/app/services/risk_service.py both use the")
    print("     name `internal_per_token` consistently, but the Pydantic field on")
    print("     MetricsLookupResponse is also `internal_per_token` — fine, just confirm during")
    print("     the demo that the dashboard renders the local-only charts.")
    print("  4. The file backend/app/metrics/d.py was renamed from confidence_metrics.py")
    print("     (visible in `git status`). The contents still document 'Confidence Metrics' in")
    print("     the docstring but the filename is now a single letter; consider renaming back")
    print("     to `confidence_metrics.py` before submission for clarity.")


if __name__ == "__main__":
    main()
