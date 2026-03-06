# HalluScope: GitHub Copilot Context & Project Guide

## Project Overview

**HalluScope** is a real-time hallucination detection and uncertainty quantification framework for Large Language Models. It implements a novel three-stage Hallucination Risk Index (HRI) pipeline that combines pre-generation prompt analysis, in-generation token-level monitoring, and post-generation semantic verification.

---

## Branch Architecture

### `beta` Branch — Local LLM Implementation (Full Pipeline)

The **beta branch** contains the **complete local LLM implementation** running **Qwen2.5-3B-Instruct** with 4-bit quantization (~6GB VRAM). This implementation provides:

**Key Features:**
- **Full model access** — Direct access to hidden states, attention patterns, and intermediate representations
- **Deep uncertainty signals** — Computes advanced metrics impossible via API:
  - **CSG (Confidence-Stability Gap)** — Detects sudden confidence drops
  - **LLD (Logit Lens Divergence)** — Probability instability across layers
  - **RCD (Representation Collapse Detection)** — Hidden state degeneration
  - **Semantic Energy** — Attention pattern sharpness via logsumexp
- **Three-stage HRI pipeline**:
  1. **Pre-Hoc (Stage 1)**: Prompt risk analysis before generation
  2. **In-Generation (Stage 2)**: Token-by-token real-time monitoring  
  3. **Post-Hoc (Stage 3)**: Semantic verification after completion

**Technical Stack:**
```
src/halluscope/
├── core/llm_backend_v4.py    # Local model inference + signal extraction
├── metrics/
│   ├── pre_hoc.py            # Stage 1: Prompt risk signals
│   └── post_hoc.py           # Stage 3: Semantic verification
├── risk/hri.py               # Risk classification
├── app/
│   ├── api.py                # FastAPI endpoints
│   ├── service_v4.py         # Analysis orchestration
│   └── ui.py                 # Streamlit visualization
```

**HRI Computation (V5.1):**
```python
HRI_final = max(
    BLEND_GEN × stage2_hri + BLEND_POST × verification_risk,
    verification_risk × POST_HOC_FLOOR,
    prompt_risk × PREHOC_FLOOR_MULT
)
```

**Use Case:** Research environments, local deployment, maximum accuracy, full uncertainty quantification.

---

### `feature/API` Branch — API-Based Implementation (Cloud LLM)

The **feature/API branch** contains the **API-based implementation** using **OpenAI GPT-4o-mini** (or compatible providers). This implementation provides:

**Key Features:**
- **No GPU required** — Cloud-based inference via API
- **Lightweight HRI** — Computes risk from API-available signals:
  - Log-probability variance
  - Confidence decay over sequence
  - Per-token entropy from top-K logprobs
  - Mean log-probability
- **Adapter pattern** — Pluggable backends (OpenAI, Anthropic, etc.)
- **Simplified pipeline** — Single-stage API-constrained analysis

**Technical Stack:**
```
backend/app/
├── adapters/
│   ├── base_adapter.py       # Abstract LLM interface
│   ├── openai_adapter.py     # OpenAI implementation
│   └── schemas.py            # Standardized result format
├── metrics/
│   ├── logprob_metrics.py    # Variance, NLL, perplexity
│   └── confidence_metrics.py # Confidence decay
├── risk/hri_engine.py        # HRI computation
├── services/
│   ├── risk_service.py       # Analysis orchestration
│   └── intervention_service.py
├── api/chat.py               # Chat endpoints
```

**HRI Computation (API):**
```python
HRI = sigmoid(
    0.4 * logprob_variance +
    0.3 * confidence_decay +
    0.3 * |mean_logprob|
)
```

**Use Case:** Production deployment, cost-effective scaling, API-first architectures, no GPU infrastructure.

---

## Research Paper Concept: HalluScope Framework

### Core Innovation

HalluScope addresses a critical gap in LLM reliability: **confident hallucinations** — fabricated content generated with high probability and low entropy that evade traditional uncertainty detection.

### Key Contributions

1. **Three-Stage HRI Pipeline**
   - Pre-hoc prompt risk estimation (before generation)
   - In-generation token-level uncertainty tracking
   - Post-hoc semantic verification (after completion)

2. **Novel Uncertainty Signals**
   - Confidence-Stability Gap (CSG): Detects overconfident instability
   - Logit Lens Divergence (LLD): Multi-layer probability consistency
   - Representation Collapse Detection (RCD): Hidden state degeneration
   - Semantic Energy: Attention distribution sharpness

3. **Confidence Paradox Detection**
   - Catches fabrications with HIGH confidence × HIGH novelty
   - Pattern matching for narrative fabrication templates
   - Claim density vs. evidence ratio analysis

4. **Dual Implementation Strategy**
   - Local LLM (beta): Full signal access, maximum accuracy
   - API-based (feature/API): Scalable deployment, reduced signals

### Mathematical Foundation

**Token-Level HRI (Stage 2):**
$$HRI_t = \sigma\left(\sum_i w_i \cdot s_i \cdot 10 + \beta\right)$$

Where signals $s_i$ include:
- $H(p)$ — Shannon entropy of token distribution  
- $E = -\log\sum_j \exp(z_j)$ — Semantic energy
- $\text{CSG} = \max(0, c - \bar{s})$ — Confidence-stability gap
- $\text{LLD}$ — Layer-wise logit divergence
- $\text{RCD}$ — Representation collapse score

**Final HRI Aggregation:**
$$HRI_{\text{final}} = \max\left(\alpha \cdot HRI_{\text{gen}} + \beta \cdot V_{\text{risk}}, \gamma \cdot V_{\text{risk}}, \delta \cdot P_{\text{risk}}\right)$$

### Risk Classification
| Level | Threshold | Interpretation |
|-------|-----------|----------------|
| LOW | HRI < 0.40 | Confident, grounded generation |
| MEDIUM | 0.40 ≤ HRI < 0.70 | Moderate uncertainty, review recommended |
| HIGH | HRI ≥ 0.70 | High hallucination risk, intervention triggered |

---

## Development Commands

```bash
# Beta branch (Local LLM)
git checkout beta
./scripts/run_api.sh     # Start API server (loads model)
./scripts/run_ui.sh      # Start Streamlit UI

# Feature/API branch (Cloud API)
git checkout feature/API
cd backend && uvicorn app.main:app --reload
```

---

## When to Use Each Branch

| Criterion | beta (Local) | feature/API (Cloud) |
|-----------|--------------|---------------------|
| GPU Available | ✅ Required (~6GB) | ❌ Not needed |
| Full Uncertainty Signals | ✅ CSG, LLD, RCD, etc. | ❌ Logprob-only |
| Latency | Moderate (local inference) | Variable (API calls) |
| Cost | Fixed (hardware) | Variable (API tokens) |
| Research Use | ✅ Recommended | Suitable |
| Production Scale | Moderate | ✅ High |

---

## File Structure Quick Reference

**Beta Branch Key Files:**
- [llm_backend_v4.py](src/halluscope/core/llm_backend_v4.py) — Core inference engine
- [pre_hoc.py](src/halluscope/metrics/pre_hoc.py) — Prompt risk analysis
- [post_hoc.py](src/halluscope/metrics/post_hoc.py) — Semantic verification
- [service_v4.py](src/halluscope/app/service_v4.py) — Analysis orchestration

**Feature/API Branch Key Files:**
- `backend/app/adapters/openai_adapter.py` — OpenAI API integration
- `backend/app/risk/hri_engine.py` — Simplified HRI computation
- `backend/app/api/chat.py` — Chat endpoints with intervention

---

This context document provides GitHub Copilot with comprehensive understanding of the HalluScope project architecture, enabling accurate code suggestions across both implementation branches.
