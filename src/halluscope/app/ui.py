"""
Streamlit UI for HalluScope.

TokenBurn-inspired hallucination risk visualization.

Run:
    PYTHONPATH=src streamlit run src/halluscope/app/ui.py
"""

import streamlit as st
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HalluScope",
    page_icon="🔬",
    layout="wide",
)

API_URL = "http://localhost:8000"


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def risk_color(label: str) -> str:
    """Return emoji based on risk label."""
    if label == "HIGH":
        return "🔴"
    if label == "MEDIUM":
        return "🟡"
    return "🟢"


def risk_style(label: str) -> str:
    """Return CSS color for risk label."""
    colors = {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#dc3545"}
    return colors.get(label, "#6c757d")


# ─────────────────────────────────────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔬 HalluScope")
st.markdown("**TokenBurn-inspired Hallucination Risk Estimation** — v1.1")
st.markdown("---")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Generation Settings")
    max_new = st.slider("Max new tokens", 32, 512, 256, step=32)
    temp = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
    compute_kl = st.checkbox("Compute KL divergence", value=True)
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown("""
    **HalluScope** estimates hallucination risk by analyzing 
    the model's internal uncertainty signals:
    
    📊 **Entropy**: Uncertainty in token choices  
    📉 **NLL / PPL**: Confidence in selected tokens  
    🌊 **Instability**: Variance across tokens  
    🎯 **KL**: Distance from uniform distribution  
    
    High risk → Clarification questions appear.
    """)
    
    st.markdown("---")
    st.caption("v1.1 (TokenBurn) | Qwen2.5-3B-Instruct")

# Main content area
st.subheader("📝 Enter Your Prompt")
prompt = st.text_area(
    "Prompt",
    height=120,
    placeholder="Ask a question or provide a statement for the model to respond to...",
    label_visibility="collapsed",
)

analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Analysis Results
# ─────────────────────────────────────────────────────────────────────────────

if analyze_btn and prompt.strip():
    with st.spinner("⏳ Analyzing... (First run loads the model, ~20s)"):
        try:
            payload = {
                "prompt": prompt,
                "max_new_tokens": max_new,
                "temperature": temp,
                "top_p": top_p,
                "compute_kl": compute_kl,
            }
            resp = requests.post(f"{API_URL}/analyze", json=payload, timeout=600)
            resp.raise_for_status()
            r = resp.json()
        except requests.exceptions.ReadTimeout:
            st.error("❌ Request timed out. The model may still be loading.")
            st.info("💡 First run loads the model into GPU memory. Please wait and try again.")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the backend is running.")
            st.code("PYTHONPATH=src uvicorn halluscope.app.api:app --port 8000", language="bash")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # HRI Score + Label
    # ─────────────────────────────────────────────────────────────────────────
    
    hri = r["hri"]
    score = hri["score"]
    label = hri["label"]
    
    col_hri1, col_hri2 = st.columns([1, 2])
    
    with col_hri1:
        st.subheader(f"{risk_color(label)} Risk: **{label}**")
        st.metric(
            label="HRI Score",
            value=f"{score:.3f}",
            delta=None,
        )
    
    with col_hri2:
        st.subheader("Risk Gauge")
        st.progress(min(score, 1.0))
        st.caption(f"0.0 = Safe | 0.45 = Medium | 0.70 = High | 1.0 = Critical")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # Generated Answer
    # ─────────────────────────────────────────────────────────────────────────
    
    st.subheader("💬 Generated Answer")
    st.markdown(f"> {r['answer']}")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # Uncertainty Signals (TokenBurn-style)
    # ─────────────────────────────────────────────────────────────────────────
    
    st.subheader("📈 Uncertainty Signals")
    
    metrics = r["metrics"]
    
    # Row 1: Primary signals
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Entropy (mean)", f"{metrics['entropy_mean']:.3f}")
        st.caption("Token uncertainty")
    
    with col2:
        st.metric("NLL", f"{metrics['nll']:.3f}")
        st.caption("Negative log-likelihood")
    
    with col3:
        ppl = metrics['perplexity']
        ppl_display = f"{ppl:.1f}" if ppl < 1000 else f"{ppl:.1e}"
        st.metric("Perplexity", ppl_display)
        st.caption("exp(NLL)")
    
    with col4:
        st.metric("Logprob Std", f"{metrics['logprob_std']:.3f}")
        st.caption("Instability signal")

    # Row 2: Secondary signals
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Logprob Var", f"{metrics['logprob_variance']:.3f}")
        st.caption("Variance")
    
    with col6:
        st.metric("Entropy (norm)", f"{metrics['entropy_normalized']:.3f}")
        st.caption("H / log(vocab)")
    
    with col7:
        kl = metrics.get('kl_uniform_mean')
        st.metric("KL Uniform", f"{kl:.3f}" if kl is not None else "—")
        st.caption("Distance from uniform")
    
    with col8:
        st.metric("Tokens", f"{metrics['num_tokens']}")
        st.caption("Generated")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # Explanation Notes
    # ─────────────────────────────────────────────────────────────────────────
    
    st.subheader("📝 Analysis Notes")
    for note in r["notes"]:
        st.info(note)

    # ─────────────────────────────────────────────────────────────────────────
    # Clarification Questions
    # ─────────────────────────────────────────────────────────────────────────
    
    if r.get("clarification_questions"):
        st.markdown("---")
        
        if label == "HIGH":
            st.subheader("⚠️ High Risk — Clarification Needed")
            st.error(
                "The model showed significant uncertainty. Before trusting this answer, "
                "please consider providing more specific context:"
            )
        else:  # MEDIUM
            st.subheader("💡 Moderate Risk — Consider Clarifying")
            st.warning(
                "The model showed some uncertainty. You may want to provide additional context:"
            )
        
        for i, q in enumerate(r["clarification_questions"], 1):
            st.markdown(f"**{i}.** {q}")

elif analyze_btn:
    st.warning("Please enter a prompt before analyzing.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "HalluScope v1.1 • TokenBurn-inspired uncertainty estimation • "
    "HRI = sigmoid(wH×entropy + wN×nll + wV×std + b)"
)
