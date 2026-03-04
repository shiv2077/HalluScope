"""
Streamlit UI for HalluScope V4.

Displays V4 mathematical signals:
- HRI trajectory & risk gauge
- Token-level risk highlighting
- V4 signals: CSG, LLD, RCD, Entropy, Energy
- Per-token signal charts
- Risk breakdown

Run:
    PYTHONPATH=src streamlit run src/halluscope/app/ui.py
"""

import streamlit as st
import requests
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="HalluScope", page_icon="🔬", layout="wide")

API_URL = "http://localhost:8000"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def risk_color(label: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(label, "⚪")


def risk_css(label: str) -> str:
    return {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#dc3545"}.get(label, "#6c757d")


def colorize_tokens(tokens: list, risks: list) -> str:
    parts = []
    for tok, risk in zip(tokens, risks):
        if risk >= 0.7:
            c, bg = "#dc3545", "rgba(220,53,69,0.15)"
        elif risk >= 0.4:
            c, bg = "#fd7e14", "rgba(253,126,20,0.15)"
        else:
            c, bg = "#28a745", "rgba(40,167,69,0.1)"
        safe = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        parts.append(
            f'<span style="color:{c};background:{bg};padding:1px 2px;'
            f'border-radius:3px;cursor:help" title="Risk: {risk:.2f}">{safe}</span>'
        )
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔬 HalluScope")
st.markdown("**Universal Hallucination Detection** — V4.1 (Pure Mathematical Signals)")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Generation Settings")
    max_new = st.slider("Max new tokens", 16, 128, 60, step=8)
    temp = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.1)
    top_p_val = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)

    st.markdown("---")
    st.header("🛡️ Detection Settings")
    hri_threshold = st.slider("HRI Threshold", 0.5, 0.95, 0.70, step=0.05)
    early_stop = st.checkbox("Enable Early Stopping", value=True)

    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown("""
    **HalluScope V4.1** detects hallucinations
    using **pure mathematical signals**:

    📊 **Entropy** — token uncertainty
    ⚡ **Semantic Energy** — confidence intensity
    🎯 **CSG** — confident hallucination detector
    🔬 **LLD** — layer reasoning stability
    🔄 **RCD** — repetition collapse

    🛑 **Early Stopping** — halts risky generation
    🎨 **Token Highlighting** — visual risk map
    📈 **HRI Trajectory** — risk over time

    **Legend:** 🟢 <0.4 | 🟠 0.4-0.7 | 🔴 ≥0.7
    """)
    st.markdown("---")
    st.caption("V4.1 | Qwen2.5-3B-Instruct | KV-Cached")

# Main
st.subheader("📝 Enter Your Prompt")
prompt = st.text_area(
    "Prompt", height=120,
    placeholder="Ask a question or provide a statement...",
    label_visibility="collapsed",
)
analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

if analyze_btn and prompt.strip():
    with st.spinner("⏳ Analyzing... (first run loads model ~30s)"):
        try:
            payload = {
                "prompt": prompt,
                "max_new_tokens": max_new,
                "temperature": temp,
                "top_p": top_p_val,
                "hri_threshold": hri_threshold,
                "early_stop_enabled": early_stop,
            }
            resp = requests.post(f"{API_URL}/analyze", json=payload, timeout=600)
            resp.raise_for_status()
            r = resp.json()
        except requests.exceptions.ReadTimeout:
            st.error("❌ Timeout. Model may still be loading — try again.")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Start the backend first:")
            st.code("PYTHONPATH=src uvicorn halluscope.app.api:app --port 8000", language="bash")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    st.markdown("---")

    # Early stopping banner
    if r.get("early_stopped"):
        st.error("⚠️ **EARLY STOPPING** — High hallucination risk. Generation halted.")

    # Intervention message
    if r.get("intervention_message"):
        st.warning(r["intervention_message"])

    # ── HRI Score ────────────────────────────────────────────────────────────

    hri = r["hri"]
    score, label = hri["score"], hri["label"]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader(f"{risk_color(label)} Risk: **{label}**")
        st.metric("Final HRI", f"{score:.3f}")
        if r.get("early_stopped"):
            st.caption("⛔ Generation stopped early")
    with c2:
        st.subheader("Risk Gauge")
        st.progress(min(score, 1.0))
        st.caption("0.0 = Safe | 0.45 = Medium | 0.70 = High | 1.0 = Critical")

    st.markdown("---")

    # ── HRI Trajectory ───────────────────────────────────────────────────────

    st.subheader("📈 HRI Trajectory")
    hri_values = r.get("hri_values", [])
    if hri_values:
        df = pd.DataFrame({"Token": range(1, len(hri_values) + 1), "HRI": hri_values})
        st.line_chart(df.set_index("Token"), use_container_width=True)
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.metric("Start", f"{hri_values[0]:.3f}")
        with tc2:
            st.metric("Peak", f"{max(hri_values):.3f}")
        with tc3:
            st.metric("Final", f"{hri_values[-1]:.3f}")
        st.caption(f"Threshold: {hri_threshold} | Tokens: {len(hri_values)}")

    st.markdown("---")

    # ── Token Highlighting ───────────────────────────────────────────────────

    st.subheader("💬 Generated Answer (Token Risk Highlighting)")
    tokens = r.get("tokens", [])
    token_risks = r.get("token_risks", [])
    if tokens and token_risks:
        html = colorize_tokens(tokens, token_risks)
        st.markdown(
            f'<div style="font-size:16px;line-height:1.8;padding:10px;'
            f'background:#f8f9fa;border-radius:8px;border:1px solid #dee2e6">'
            f'{html}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Hover tokens for risk scores")
        with st.expander("📊 Token Risk Distribution"):
            hi = sum(1 for x in token_risks if x >= 0.7)
            md = sum(1 for x in token_risks if 0.4 <= x < 0.7)
            lo = sum(1 for x in token_risks if x < 0.4)
            d1, d2, d3 = st.columns(3)
            with d1:
                st.metric("🟢 Low", f"{lo}")
            with d2:
                st.metric("🟠 Moderate", f"{md}")
            with d3:
                st.metric("🔴 High", f"{hi}")
    else:
        st.markdown(f"> {r['answer']}")

    st.markdown("---")

    # ── V4 Mathematical Signals ──────────────────────────────────────────────

    st.subheader("🎯 V4 Hallucination Signals")
    v4 = r.get("v4_signals", {})

    if v4:
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("🎯 CSG", f"{v4.get('confidence_stability_gap', 0):.3f}")
            st.caption("Confident hallucination")
        with s2:
            st.metric("🔬 LLD", f"{v4.get('logit_lens_divergence', 0):.3f}")
            st.caption("Layer instability")
        with s3:
            st.metric("🔄 RCD", f"{v4.get('representation_collapse', 0):.3f}")
            st.caption("Collapse risk")
        with s4:
            st.metric("📊 Entropy", f"{v4.get('entropy', 0):.3f}")
            st.caption("Uncertainty")

        s5, s6, s7, s8 = st.columns(4)
        with s5:
            st.metric("⚡ Energy", f"{v4.get('semantic_energy', 0):.3f}")
            st.caption("Confidence intensity")
        with s6:
            st.metric("📉 LP Var", f"{v4.get('logprob_variance', 0):.3f}")
            st.caption("Instability")
        with s7:
            st.metric("🎯 Confidence", f"{v4.get('confidence', 0):.3f}")
            st.caption("Model confidence")
        with s8:
            st.metric("🔗 Stability", f"{v4.get('semantic_stability', 0):.3f}")
            st.caption("Prompt alignment")

    st.markdown("---")

    # ── Per-Token Signal Charts ──────────────────────────────────────────────

    pts = r.get("per_token_signals", {})
    if pts:
        with st.expander("📈 Per-Token Signal Trajectories"):
            chart_data = {}
            for sig in ["hri", "csg", "lld", "rcd"]:
                vals = pts.get(sig, [])
                if vals:
                    chart_data[sig.upper()] = vals
            if chart_data:
                max_len = max(len(v) for v in chart_data.values())
                for k in chart_data:
                    while len(chart_data[k]) < max_len:
                        chart_data[k].append(chart_data[k][-1] if chart_data[k] else 0)
                df_signals = pd.DataFrame(chart_data)
                df_signals.index = range(1, len(df_signals) + 1)
                df_signals.index.name = "Token"
                st.line_chart(df_signals, use_container_width=True)

    # ── HRI Dynamics ─────────────────────────────────────────────────────────

    dynamics = r.get("hri_dynamics", {})
    if dynamics:
        with st.expander("⚡ HRI Dynamics"):
            dy1, dy2 = st.columns(2)
            with dy1:
                st.metric("📈 Slope", f"{dynamics.get('slope', 0):.4f}")
                st.caption("Rate of risk change")
            with dy2:
                st.metric("⚡ Acceleration", f"{dynamics.get('acceleration', 0):.4f}")
                st.caption("Risk spike detection")

    # ── Uncertainty Metrics ──────────────────────────────────────────────────

    metrics = r.get("metrics", {})
    if metrics:
        with st.expander("📈 Uncertainty Metrics"):
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Entropy (mean)", f"{metrics.get('entropy_mean', 0):.3f}")
            with m2:
                st.metric("NLL", f"{metrics.get('nll', 0):.3f}")
            with m3:
                ppl = metrics.get('perplexity', 1)
                st.metric("Perplexity", f"{ppl:.1f}" if ppl < 1000 else f"{ppl:.1e}")
            with m4:
                st.metric("LP Std", f"{metrics.get('logprob_std', 0):.3f}")

    # ── Risk Breakdown ───────────────────────────────────────────────────────

    rb = r.get("risk_breakdown", {})
    if rb:
        with st.expander("📊 Risk Component Breakdown"):
            for comp, val in rb.items():
                cn, cb = st.columns([1, 3])
                with cn:
                    st.markdown(f"**{comp}**")
                with cb:
                    st.progress(min(float(val) / 1.0, 1.0))
                    st.caption(f"{val:.4f}")

    st.markdown("---")

    # ── Notes ────────────────────────────────────────────────────────────────

    st.subheader("📝 Analysis Notes")
    for note in r.get("notes", []):
        st.info(note)

    # ── Clarification Questions ──────────────────────────────────────────────

    if r.get("clarification_questions"):
        st.markdown("---")
        if label == "HIGH" or r.get("early_stopped"):
            st.subheader("⚠️ High Risk — Clarification Needed")
            st.error("The model showed significant uncertainty. Verify output carefully:")
        else:
            st.subheader("💡 Moderate Risk")
            st.warning("Consider providing additional context:")
        for i, q in enumerate(r["clarification_questions"], 1):
            st.markdown(f"**{i}.** {q}")

elif analyze_btn:
    st.warning("Please enter a prompt.")

# Footer
st.markdown("---")
st.caption(
    "HalluScope V4.1 • Pure mathematical hallucination detection • "
    "KV-cached inference • Qwen2.5-3B-Instruct"
)
