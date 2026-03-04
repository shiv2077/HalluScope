"""
Streamlit UI for HalluScope.

TokenBurn-inspired hallucination risk visualization with:
- Token-level risk highlighting (green/orange/red)
- Real-time HRI trajectory chart
- Early stopping indicators

Run:
    PYTHONPATH=src streamlit run src/halluscope/app/ui.py
"""

import streamlit as st
import requests
import pandas as pd

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


def colorize_tokens(tokens: list, risks: list) -> str:
    """
    Generate HTML with color-coded tokens based on risk scores.
    
    Color rules:
    - risk < 0.4 → green (safe)
    - 0.4 <= risk < 0.7 → orange (moderate risk)
    - risk >= 0.7 → red (high risk)
    
    Args:
        tokens: List of token strings
        risks: List of risk scores [0, 1]
        
    Returns:
        HTML string with colored spans
    """
    html_parts = []
    for token, risk in zip(tokens, risks):
        # Determine color based on risk
        if risk >= 0.7:
            color = "#dc3545"  # Red
            bg = "rgba(220, 53, 69, 0.15)"
        elif risk >= 0.4:
            color = "#fd7e14"  # Orange
            bg = "rgba(253, 126, 20, 0.15)"
        else:
            color = "#28a745"  # Green
            bg = "rgba(40, 167, 69, 0.1)"
        
        # Escape HTML characters in token
        safe_token = (
            token.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        
        # Create styled span with tooltip showing risk score
        html_parts.append(
            f'<span style="color:{color}; background:{bg}; padding:1px 2px; '
            f'border-radius:3px; cursor:help;" title="Risk: {risk:.2f}">{safe_token}</span>'
        )
    
    return "".join(html_parts)


# ─────────────────────────────────────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔬 HalluScope")
st.markdown("**Pre-Hoc Hallucination Risk Detection** — v3.0 (Multi-signal detection)")
st.markdown("---")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Generation Settings")
    max_new = st.slider("Max new tokens", 16, 128, 60, step=8)
    temp = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
    
    st.markdown("---")
    st.header("🛡️ Pre-Hoc Detection")
    hri_threshold = st.slider("HRI Threshold", 0.5, 0.95, 0.70, step=0.05)
    early_stop = st.checkbox("Enable Early Stopping", value=True)
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown("""
    **HalluScope v3.0** detects hallucinations 
    using **multiple signals**:
    
    🎯 **Multi-Signal Detection**:
    - 📊 Uncertainty (entropy, NLL)
    - 📋 Prompt risk patterns
    - 🏷️ Entity novelty tracking
    - 🔍 Retrieval verification
    
    🛑 **Early Stopping**: Halts risky generation  
    🎨 **Token Highlighting**: Visual risk map  
    📈 **HRI Trajectory**: Risk over time  
    
    **Color Legend:**
    - 🟢 Green: Low risk (<0.4)
    - 🟠 Orange: Moderate (0.4-0.7)
    - 🔴 Red: High risk (≥0.7)
    """)
    
    st.markdown("---")
    st.caption("v3.0 (Multi-Signal) | Qwen2.5-3B-Instruct")

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
                "hri_threshold": hri_threshold,
                "early_stop_enabled": early_stop,
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
    # Early Stopping Banner
    # ─────────────────────────────────────────────────────────────────────────
    
    if r.get("early_stopped", False):
        st.error(
            "⚠️ **EARLY STOPPING TRIGGERED** — High hallucination risk detected. "
            "Generation halted to prevent potentially unreliable output."
        )
        st.markdown("")

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
            label="Final HRI Score",
            value=f"{score:.3f}",
            delta=None,
        )
        if r.get("early_stopped"):
            st.caption("⛔ Generation stopped early")
    
    with col_hri2:
        st.subheader("Risk Gauge")
        st.progress(min(score, 1.0))
        st.caption(f"0.0 = Safe | 0.45 = Medium | 0.70 = High | 1.0 = Critical")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # HRI Trajectory Chart
    # ─────────────────────────────────────────────────────────────────────────
    
    st.subheader("📈 HRI Trajectory (Real-Time Monitoring)")
    
    hri_values = r.get("hri_values", [])
    if hri_values:
        # Create DataFrame for chart
        df = pd.DataFrame({
            "Token": range(1, len(hri_values) + 1),
            "HRI": hri_values,
        })
        
        # Add threshold line reference
        st.line_chart(df.set_index("Token"), use_container_width=True)
        
        col_traj1, col_traj2, col_traj3 = st.columns(3)
        with col_traj1:
            st.metric("Start HRI", f"{hri_values[0]:.3f}" if hri_values else "—")
        with col_traj2:
            max_hri = max(hri_values) if hri_values else 0
            st.metric("Peak HRI", f"{max_hri:.3f}")
        with col_traj3:
            st.metric("Final HRI", f"{hri_values[-1]:.3f}" if hri_values else "—")
        
        st.caption(f"🔸 Threshold: {hri_threshold} | 📊 Tokens generated: {len(hri_values)}")
    else:
        st.info("No HRI trajectory data available.")
    
    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # Generated Answer with Token Highlighting
    # ─────────────────────────────────────────────────────────────────────────
    
    st.subheader("💬 Generated Answer (Token Risk Highlighting)")
    
    tokens = r.get("tokens", [])
    token_risks = r.get("token_risks", [])
    
    if tokens and token_risks:
        # Render colorized tokens
        html_output = colorize_tokens(tokens, token_risks)
        st.markdown(
            f'<div style="font-size: 16px; line-height: 1.8; padding: 10px; '
            f'background: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">'
            f'{html_output}</div>',
            unsafe_allow_html=True
        )
        st.caption("💡 Hover over tokens to see individual risk scores")
        
        # Token risk distribution
        with st.expander("📊 Token Risk Distribution"):
            high_risk = sum(1 for r in token_risks if r >= 0.7)
            med_risk = sum(1 for r in token_risks if 0.4 <= r < 0.7)
            low_risk = sum(1 for r in token_risks if r < 0.4)
            
            col_dist1, col_dist2, col_dist3 = st.columns(3)
            with col_dist1:
                st.metric("🟢 Low Risk", f"{low_risk} tokens")
            with col_dist2:
                st.metric("🟠 Moderate Risk", f"{med_risk} tokens")
            with col_dist3:
                st.metric("🔴 High Risk", f"{high_risk} tokens")
    else:
        # Fallback to plain text
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
        early_str = "Yes ⛔" if r.get("early_stopped") else "No ✅"
        st.metric("Early Stopped", early_str)
        st.caption("Pre-hoc detection")
    
    with col8:
        st.metric("Tokens", f"{metrics['num_tokens']}")
        st.caption("Generated")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-SIGNAL HALLUCINATION DETECTION (NEW v3.0)
    # ─────────────────────────────────────────────────────────────────────────
    
    st.subheader("🎯 Multi-Signal Hallucination Detection")
    st.caption("Detects both uncertain AND confident hallucinations using multiple signals")
    
    # Get risk signals
    risk_signals = r.get("risk_signals", {})
    
    if risk_signals:
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            prompt_risk = risk_signals.get("prompt_risk", 0.0)
            risk_pct = int(prompt_risk * 400)  # Scale to 100%
            st.metric(
                "📋 Prompt Risk", 
                f"{prompt_risk:.3f}",
                delta=f"{risk_pct}%" if prompt_risk > 0.05 else None,
                delta_color="inverse"
            )
            st.caption("Citation/specificity patterns")
        
        with col_s2:
            entity_risk = risk_signals.get("entity_risk", 0.0)
            risk_pct = int(entity_risk * 400)
            st.metric(
                "🏷️ Entity Risk", 
                f"{entity_risk:.3f}",
                delta=f"{risk_pct}%" if entity_risk > 0.05 else None,
                delta_color="inverse"
            )
            st.caption("Novel entities fabricated")
        
        with col_s3:
            retrieval_risk = risk_signals.get("retrieval_risk", 0.0)
            risk_pct = int(retrieval_risk * 400)
            st.metric(
                "🔍 Retrieval Risk", 
                f"{retrieval_risk:.3f}",
                delta=f"{risk_pct}%" if retrieval_risk > 0.05 else None,
                delta_color="inverse"
            )
            st.caption("Unverifiable claims")
        
        # Risk slope and acceleration
        col_sl1, col_sl2 = st.columns(2)
        with col_sl1:
            hri_slope = risk_signals.get("hri_slope", 0.0)
            st.metric("📈 HRI Slope", f"{hri_slope:.4f}")
            st.caption("Rate of risk change")
        with col_sl2:
            hri_accel = risk_signals.get("hri_acceleration", 0.0)
            st.metric("⚡ HRI Acceleration", f"{hri_accel:.4f}")
            st.caption("Risk spike detection")
    
    # Novel Entities Section
    novel_entities = r.get("novel_entities", [])
    if novel_entities:
        with st.expander(f"🏷️ Novel Entities Detected ({len(novel_entities)})"):
            st.warning("These entities appear in the output but NOT in the prompt. They may be fabricated:")
            for entity in novel_entities[:10]:
                st.markdown(f"• **{entity}**")
    
    # Unverified Claims Section
    unverified_claims = r.get("unverified_claims", [])
    risk_signals_detail = r.get("risk_signals_detail", [])
    
    if unverified_claims or risk_signals_detail:
        with st.expander(f"⚠️ Unverified Claims ({len(unverified_claims)})"):
            if unverified_claims:
                st.error("These claims could not be verified:")
                for claim in unverified_claims[:5]:
                    st.markdown(f"• {claim}")
            
            if risk_signals_detail:
                st.warning("Risk signals detected:")
                for signal in risk_signals_detail[:5]:
                    st.markdown(f"• {signal}")
    
    # Risk Breakdown Chart
    risk_breakdown = r.get("risk_breakdown", {})
    if risk_breakdown:
        with st.expander("📊 Full Risk Breakdown"):
            # Create a simple bar chart using columns
            st.markdown("**Risk Component Contributions:**")
            for component, value in risk_breakdown.items():
                col_name, col_bar = st.columns([1, 3])
                with col_name:
                    st.markdown(f"**{component}**")
                with col_bar:
                    st.progress(min(value / 5.0, 1.0))  # Scale for visibility
                    st.caption(f"{value:.4f}")

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
        
        if label == "HIGH" or r.get("early_stopped"):
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
    "HalluScope v3.0 • Multi-signal hallucination detection • "
    "HRI = sigmoid(wH×entropy + wN×nll + wV×std + wP×prompt_risk + wE×entity_risk + wR×retrieval_risk + b)"
)
