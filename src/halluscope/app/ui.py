"""
Streamlit UI for HalluScope.

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

def hri_color(hri: float) -> str:
    """Return color based on HRI value."""
    if hri >= 0.65:
        return "🔴"
    if hri >= 0.40:
        return "🟡"
    return "🟢"


def risk_badge(level: str) -> str:
    """Return styled risk badge."""
    colors = {"low": "green", "medium": "orange", "high": "red"}
    return f":{colors.get(level, 'gray')}[**{level.upper()}**]"


# ─────────────────────────────────────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔬 HalluScope")
st.markdown("**Proactive Hallucination Risk Estimation** — Sprint-1 (Entropy + NLL)")
st.markdown("---")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Generation Settings")
    max_new = st.slider("Max new tokens", 32, 512, 256, step=32)
    temp = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown("""
    **HalluScope** estimates hallucination risk
    by analyzing model uncertainty signals:
    
    - **Entropy**: How uncertain is the model about each token?
    - **NLL**: How confident was the model in the tokens it chose?
    - **HRI**: Combined risk score in [0,1]
    
    High HRI → Clarification questions appear.
    """)
    
    st.markdown("---")
    st.caption("Sprint-1 | Model: Qwen2.5-3B-Instruct")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Your Prompt")
    prompt = st.text_area(
        "Prompt",
        height=150,
        placeholder="Ask a question or provide a statement for the model to respond to...",
        label_visibility="collapsed",
    )

analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Analysis Results
# ─────────────────────────────────────────────────────────────────────────────

if analyze_btn and prompt.strip():
    with st.spinner("⏳ Generating response... (First run downloads the ~6GB model, please wait 5-10 min)"):
        try:
            payload = {
                "prompt": prompt,
                "max_new_tokens": max_new,
                "temperature": temp,
                "top_p": top_p,
            }
            resp = requests.post(f"{API_URL}/analyze", json=payload, timeout=600)
            resp.raise_for_status()
            r = resp.json()
        except requests.exceptions.ReadTimeout:
            st.error("❌ Request timed out. The model may still be downloading/loading. Check terminal for progress.")
            st.info("💡 First run downloads Qwen2.5-3B-Instruct (~6GB). Please wait and try again.")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the backend is running on localhost:8000")
            st.code("PYTHONPATH=src uvicorn halluscope.app.api:app --reload", language="bash")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    st.markdown("---")

    # Results layout
    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        st.subheader("💬 Generated Answer")
        st.markdown(f"> {r['answer']}")

    with res_col2:
        st.subheader("📊 Risk Assessment")
        
        hri = r["hri"]
        level = r["risk_level"]
        
        st.metric(
            label=f"{hri_color(hri)} Hallucination Risk Index",
            value=f"{hri:.3f}",
            delta=f"{level.upper()} risk",
            delta_color="inverse" if level == "low" else "off",
        )

    # Metrics breakdown
    st.markdown("---")
    st.subheader("📈 Uncertainty Signals")
    
    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
    
    metrics = r["metrics"]
    with met_col1:
        st.metric("Entropy (mean)", f"{metrics['entropy_mean']:.4f}")
    with met_col2:
        st.metric("NLL (mean)", f"{metrics['nll']:.4f}")
    with met_col3:
        st.metric("Disagreement", metrics.get("disagreement") or "—")
        st.caption("Sprint-2")
    with met_col4:
        st.metric("EDFL Proxy", metrics.get("edfl_proxy") or "—")
        st.caption("Sprint-3")

    # Explanation notes
    st.markdown("---")
    st.subheader("📝 Explanation Notes")
    for note in r["notes"]:
        st.info(note)

    # Clarification questions (only if high risk)
    if r.get("clarification_questions"):
        st.markdown("---")
        st.subheader("⚠️ Clarification Needed (High Risk)")
        st.warning(
            "The model showed high uncertainty. Before trusting this answer, "
            "please consider providing more specific context:"
        )
        for i, q in enumerate(r["clarification_questions"], 1):
            st.markdown(f"**{i}.** {q}")

elif analyze_btn:
    st.warning("Please enter a prompt before analyzing.")
