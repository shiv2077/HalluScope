# HalluScope V5.1

Real-time hallucination detection and visualization for Large Language Models.

## Overview

HalluScope monitors LLM generation in real-time using a **three-stage pipeline** to detect potential hallucinations:

```
┌─────────────┐    ┌─────────────────┐    ┌─────────────┐
│  Stage 1    │ → │     Stage 2     │ → │   Stage 3   │
│  Pre-Hoc    │    │  In-Generation  │    │  Post-Hoc   │
│   Prompt    │    │    Token-by-    │    │  Semantic   │
│  Analysis   │    │     Token       │    │ Verification│
└─────────────┘    └─────────────────┘    └─────────────┘
```

**V5.1 Visualization Mode**: Instead of blocking generation, HalluScope now completes full outputs and provides detailed per-token risk visualization with highlighted hallucination spans.

## Features

- **Three-stage HRI pipeline** — Pre-hoc, In-Generation, Post-Hoc analysis
- **Per-token diagnostics** — Entropy, energy, CSG, LLD, RCD signals for each token
- **Hallucination segments** — Groups consecutive high-risk tokens with signal breakdown
- **Highlighted output** — HTML and text markers showing LOW/MEDIUM/HIGH risk tokens
- **Real-time trajectory** — HRI, entropy, and energy charts during generation
- **4-bit quantization** — Runs Qwen2.5-3B-Instruct on consumer GPUs (~6GB VRAM)

## Architecture

### Stage 1: Pre-Hoc Prompt Analysis

Analyzes the prompt before generation for hallucination-prone patterns:

| Signal | Description |
|--------|-------------|
| Citation Risk | Detects fake paper citations, invented DOIs |
| Entity Rarity | Flags unusual entity combinations |
| Temporal Risk | Knowledge cutoff mismatches |
| Compositional Novelty | Novel concept combinations |
| Prompt Perplexity | Model confusion on prompt |

### Stage 2: In-Generation Monitoring

Computes uncertainty signals for each generated token:

| Signal | Description |
|--------|-------------|
| Entropy (H) | Token prediction uncertainty |
| Semantic Energy | Attention pattern sharpness |
| CSG | Commitment Spike Gate — sudden confidence drops |
| LLD | Local Logprob Drift — probability instability |
| RCD | Rolling Confidence Divergence — sustained uncertainty |
| Logprob Variance | Token probability consistency |

### Stage 3: Post-Hoc Verification

Analyzes the completed response for semantic issues:

| Signal | Description |
|--------|-------------|
| Novelty Risk | Novel entities not grounded in prompt |
| Claim Density | Unverifiable factual assertions |
| Paradox Detection | Self-contradictory statements |
| Template Risk | Generic hedging patterns |
| Explosion Risk | Unbounded lists or tangents |

### HRI Computation

Final HRI blends all three stages:

```
HRI_final = max(
    BLEND_GEN × stage2_hri + BLEND_POST × verification_risk,
    verification_risk × POST_HOC_FLOOR,
    prompt_risk × PREHOC_FLOOR_MULT
)
```

Risk levels:
- **LOW**: HRI < 0.40
- **MEDIUM**: 0.40 ≤ HRI < 0.70
- **HIGH**: HRI ≥ 0.70

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA GPU with ≥6GB VRAM
- Linux/macOS

### Installation

```bash
git clone <repo>
cd HalluScope
pip install -e ".[dev]"
```

### Run Services

```bash
# Terminal 1: API server (loads model)
./scripts/run_api.sh    # http://localhost:8000

# Terminal 2: Streamlit UI
./scripts/run_ui.sh     # http://localhost:8501
```

### API Usage

```python
import requests

resp = requests.post("http://localhost:8000/analyze", json={
    "prompt": "Summarize the 1923 paper 'Gravitational Harmonics'",
    "max_new_tokens": 100,
    "early_stop_enabled": False,  # V5.1: Complete full generation
})

data = resp.json()
```

### Response Structure (V5.1)

```json
{
  "answer": "Generated text...",
  "hri": {
    "final": 0.72,
    "mean": 0.65,
    "max": 0.78,
    "label": "HIGH"
  },
  "early_stopped": false,
  
  "hri_stats": {
    "mean": 0.65,
    "peak": 0.78,
    "final": 0.72
  },
  
  "token_diagnostics": [
    {
      "index": 0,
      "token": "I",
      "logprob": -0.12,
      "entropy": 1.45,
      "energy": 0.23,
      "csg": 0.0,
      "lld": 0.02,
      "rcd": 0.01,
      "hri": 0.35,
      "risk_level": "LOW"
    }
  ],
  
  "hallucination_segments": [
    {
      "start_idx": 15,
      "end_idx": 28,
      "text": "published in Nature 1923...",
      "avg_hri": 0.74,
      "max_hri": 0.78,
      "signal_breakdown": {
        "entropy": 0.32,
        "energy": 0.18,
        "csg": 0.25,
        "lld": 0.15,
        "rcd": 0.12
      }
    }
  ],
  
  "highlighted_html": "<span style='color:green'>I apologize</span>...",
  "highlighted_text": "I apologize [but the paper] «you mentioned does not exist»...",
  
  "per_token_entropy": [1.2, 1.5, 2.1],
  "per_token_energy": [0.1, 0.2, 0.3],
  "per_token_logprob": [-0.5, -0.8, -1.2]
}
```

### Highlighting Legend

**HTML output**: Color-coded spans
- 🟢 Green = LOW risk
- 🟠 Orange = MEDIUM risk  
- 🔴 Red = HIGH risk

**Text markers**:
- No markers = LOW
- `[brackets]` = MEDIUM
- `«guillemets»` = HIGH

## Project Structure

```
HalluScope/
├── configs/
│   ├── app.yaml          # Server configuration
│   ├── model.yaml        # Model settings
│   └── risk.yaml         # Stage weights and thresholds
├── scripts/
│   ├── run_api.sh        # Start FastAPI server
│   └── run_ui.sh         # Start Streamlit UI
├── src/halluscope/
│   ├── app/
│   │   ├── api.py        # FastAPI endpoints
│   │   ├── service.py    # Analysis orchestration
│   │   ├── service_v4.py # V5.1 visualization functions
│   │   └── ui.py         # Streamlit dashboard
│   ├── core/
│   │   ├── llm_backend.py    # HuggingFace model wrapper
│   │   └── llm_backend_v4.py # Three-stage generation
│   ├── metrics/
│   │   ├── entropy.py    # Token entropy computation
│   │   ├── kl_divergence.py
│   │   └── logprob.py    # Logprob utilities
│   └── risk/
│       ├── hri.py        # Risk classification
│       ├── prompt_analysis.py
│       ├── generation_monitor.py
│       └── post_verification.py
└── tests/
    └── test_*.py         # Unit tests (40 passing)
```

## Configuration

### Risk Weights (`configs/risk.yaml`)

```yaml
stage2_weights:
  FACTUAL:
    w_entropy: 0.20
    w_energy: 0.05
    w_variance: 0.15
    w_csg: 0.20
    w_lld: 0.20
    w_rcd: 0.20
    bias: -2.0

thresholds:
  low: 0.45
  high: 0.70
```

### Model (`configs/model.yaml`)

Default: `Qwen/Qwen2.5-3B-Instruct` with 4-bit NF4 quantization.

## Development

```bash
# Run tests
PYTHONPATH=src pytest tests/ -v

# Type checking
mypy src/halluscope

# Format
black src/ tests/
isort src/ tests/
```

## Model Requirements

| Model | VRAM | Notes |
|-------|------|-------|
| Qwen2.5-3B-Instruct | ~6GB | Default, 4-bit quantized |
| Qwen2.5-7B-Instruct | ~10GB | Higher quality |
| Llama-3-8B-Instruct | ~12GB | Alternative |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Full hallucination analysis |
| `/health` | GET | Service health check |
| `/model/info` | GET | Loaded model details |

## License

See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome. Run tests before submitting PRs:

```bash
PYTHONPATH=src pytest tests/ -v
```
