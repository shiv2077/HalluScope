# HalluScope – Proactive Hallucination Risk Estimation

A local LLM application with uncertainty-based hallucination detection. HalluScope estimates hallucination risk **at inference time** by analyzing model-internal signals (token probabilities) during generation, triggering clarification questions when risk is high.

## 🎯 Key Concept

Unlike retrieval-based fact-checking, HalluScope monitors the model's own uncertainty signals:
- **Token Entropy**: How uncertain is the model about each word choice?
- **Negative Log-Likelihood (NLL)**: How confident was the model in its outputs?
- **Hallucination Risk Index (HRI)**: Combined score in [0,1] triggering interventions

When HRI exceeds the threshold, the system asks for clarification rather than refusing.

## 📊 Sprint-1 Features (Current)

- ✅ Token-level entropy from per-step logits
- ✅ Mean NLL aggregation
- ✅ HRI-v1: `sigmoid(0.5*entropy + 0.5*nll - 2.0)`
- ✅ Clarification questions when HRI ≥ 0.65
- ✅ FastAPI backend + Streamlit UI

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to project
cd HalluScope

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the API (Terminal 1)

```bash
PYTHONPATH=src uvicorn halluscope.app.api:app --reload --host 0.0.0.0 --port 8000
```

The model (Qwen2.5-3B-Instruct) will auto-download on first request via HuggingFace.

### 3. Run the UI (Terminal 2)

```bash
PYTHONPATH=src streamlit run src/halluscope/app/ui.py
```

Open http://localhost:8501 in your browser.

### Alternative: Use Scripts

```bash
bash scripts/run_api.sh   # Terminal 1
bash scripts/run_ui.sh    # Terminal 2
```

## 🧪 API Usage

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What year did Einstein discover gravity?"}'
```

Response:
```json
{
  "answer": "...",
  "hri": 0.72,
  "risk_level": "high",
  "metrics": {"entropy_mean": 4.2, "nll": 3.1},
  "notes": ["High token entropy: ..."],
  "clarification_questions": ["Can you specify..."]
}
```

## 📁 Project Structure

```
HalluScope/
├── src/halluscope/
│   ├── core/           # LLM backend (generation with logits)
│   ├── metrics/        # Entropy, logprob, disagreement (Sprint-2)
│   ├── risk/           # HRI computation and thresholds
│   ├── app/            # FastAPI + Streamlit
│   ├── eval/           # Evaluation pipeline (TruthfulQA, HaluEval)
│   └── datasets/       # Dataset loaders
├── configs/            # YAML configuration
├── notebooks/          # Jupyter notebooks for exploration
├── scripts/            # Shell scripts
└── tests/              # Unit tests
```

## 🔮 Roadmap

| Sprint | Feature | Status |
|--------|---------|--------|
| 1 | Entropy + NLL → HRI-v1 | ✅ Done |
| 2 | Multi-sample disagreement | 🔜 Planned |
| 3 | EDFL proxy (evidence sensitivity) | 🔜 Planned |
| 4 | TruthfulQA/HaluEval calibration | 🔜 Planned |

## 🛠️ Development

```bash
# Run tests
PYTHONPATH=src pytest tests/ -v

# Format code
black src/ tests/
isort src/ tests/
```

## 📚 References

- TruthfulQA: Measuring How Models Mimic Human Falsehoods
- HaluEval: A Large-Scale Hallucination Evaluation Benchmark
- Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation

## 📄 License

See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read the documentation in `docs/` before submitting PRs.
