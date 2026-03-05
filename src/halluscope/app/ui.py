"""
Streamlit UI for HalluScope V5.

Three-stage hallucination detection display:
- Stage 1: Pre-hoc prompt risk
- Stage 2: In-generation token signals
- Stage 3: Post-generation semantic verification

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


def stage_indicator(score: float) -> str:
    if score >= 0.70:
        return "🔴"
    elif score >= 0.45:
        return "🟡"
    return "🟢"


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔬 HalluScope")
st.markdown("**Three-Stage Universal Hallucination Detection** — V5")
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
    st.header("ℹ️ Architecture")
    st.markdown("""
    **Stage 1** — Pre-Hoc Prompt Risk
    🔍 Citation patterns, entity rarity, prompt perplexity

    **Stage 2** — In-Generation Monitoring
    📊 Entropy, CSG, LLD, RCD, Energy

    **Stage 3** — Post-Generation Verification
    🧪 Novelty, Claim Density, Confidence Paradox

    **Final HRI** = blend of all 3 stages

    **Legend:** 🟢 <0.45 | 🟡 0.45-0.70 | 🔴 ≥0.70
    """)
    st.markdown("---")
    st.caption("V5 | Qwen2.5-3B-Instruct | 3-Stage")

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

    if r.get("intervention_message"):
        st.warning(r["intervention_message"])

    # ── Final HRI Score ──────────────────────────────────────────────────────

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

    # ══════════════════════════════════════════════════════════════════════════
    # THREE-STAGE BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════════

    st.subheader("🏗️ Three-Stage Detection Pipeline")

    # Stage scores bar
    s1_score = hri.get("prompt_risk", 0.0)
    s2_score = hri.get("stage2_hri", 0.0)
    s3_score = hri.get("verification_risk", 0.0)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric(f"{stage_indicator(s1_score)} Stage 1", f"{s1_score:.3f}")
        st.caption("Prompt Risk")
    with sc2:
        st.metric(f"{stage_indicator(s2_score)} Stage 2", f"{s2_score:.3f}")
        st.caption("Generation HRI")
    with sc3:
        st.metric(f"{stage_indicator(s3_score)} Stage 3", f"{s3_score:.3f}")
        st.caption("Verification Risk")
    with sc4:
        st.metric(f"{stage_indicator(score)} Final", f"{score:.3f}")
        st.caption("Blended HRI")

    # ── Stage 1 Detail ───────────────────────────────────────────────────────

    s1 = r.get("stage1_prehoc", {})
    if s1:
        with st.expander("🔍 Stage 1 — Pre-Hoc Prompt Risk Analysis", expanded=s1_score > 0.35):
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric("📄 Citation", f"{s1.get('citation_score', 0):.3f}")
            with p2:
                st.metric("📊 Perplexity", f"{s1.get('prompt_perplexity', 1):.1f}")
            with p3:
                st.metric("⚡ Entity Rarity", f"{s1.get('entity_rarity', 0):.3f}")
            with p4:
                st.metric("🕐 Temporal", f"{s1.get('temporal_risk', 0):.3f}")

            p5, p6, p7 = st.columns(3)
            with p5:
                st.metric("👤 Entity Density", f"{s1.get('entity_density', 0):.3f}")
            with p6:
                st.metric("📌 Specificity", f"{s1.get('specificity_score', 0):.3f}")
            with p7:
                st.metric("📑 Quoted Title", f"{s1.get('quoted_title_score', 0):.3f}")

            entities = s1.get("entities_found", [])
            if entities:
                st.caption(f"Entities detected: {', '.join(entities[:8])}")
            years = s1.get("years_found", [])
            if years:
                st.caption(f"Years found: {years}")

    # ── Stage 2 Detail ───────────────────────────────────────────────────────

    s2 = r.get("stage2_generation", {})
    if s2:
        with st.expander("📊 Stage 2 — In-Generation Token Signals", expanded=s2_score > 0.40):
            g1, g2, g3 = st.columns(3)
            with g1:
                st.metric("📊 Entropy", f"{s2.get('entropy', 0):.3f}")
                st.metric("⚡ Energy", f"{s2.get('semantic_energy', 0):.3f}")
            with g2:
                st.metric("📉 LP Variance", f"{s2.get('logprob_variance', 0):.3f}")
                st.metric("🎯 CSG", f"{s2.get('confidence_stability_gap', 0):.3f}")
            with g3:
                st.metric("🔬 LLD", f"{s2.get('logit_lens_divergence', 0):.3f}")
                st.metric("🔄 RCD", f"{s2.get('representation_collapse', 0):.3f}")

            g4, g5 = st.columns(2)
            with g4:
                st.metric("🎯 Confidence", f"{s2.get('confidence', 0):.3f}")
            with g5:
                st.metric("🔗 Stability", f"{s2.get('semantic_stability', 0):.3f}")

    # ── Stage 3 Detail ───────────────────────────────────────────────────────

    s3 = r.get("stage3_posthoc", {})
    if s3:
        with st.expander("🧪 Stage 3 — Post-Generation Verification", expanded=s3_score > 0.40):
            v1, v2, v3 = st.columns(3)
            with v1:
                paradox = s3.get('confidence_paradox', 0)
                st.metric("💥 Confidence Paradox", f"{paradox:.3f}")
                st.caption("confidence × novelty")
            with v2:
                st.metric("🧭 Semantic Novelty", f"{s3.get('semantic_novelty', 0):.3f}")
                st.caption("Embedding distance")
            with v3:
                st.metric("📝 Claim Density", f"{s3.get('claim_density', 0):.3f}")
                st.caption("Assertions / evidence")

            v4, v5, v6 = st.columns(3)
            with v4:
                st.metric("📜 Template Score", f"{s3.get('narrative_template_score', 0):.3f}")
                st.caption("Fabrication patterns")
            with v5:
                st.metric("💡 Concept Explosion", f"{s3.get('concept_explosion', 0):.3f}")
                st.caption("New concept ratio")
            with v6:
                st.metric("🔤 Entity Expansion", f"{s3.get('entity_expansion', 0):.3f}")
                st.caption("Novel word ratio")

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
    else:
        st.markdown(f"> {r['answer']}")

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

    # ── Notes ────────────────────────────────────────────────────────────────

    st.subheader("📝 Analysis Notes")
    for note in r.get("notes", []):
        if note.startswith("⚠️"):
            st.warning(note)
        elif note.startswith("✅"):
            st.success(note)
        else:
            st.info(note)

    # ── Clarification Questions ──────────────────────────────────────────────

    if r.get("clarification_questions"):
        st.markdown("---")
        if label == "HIGH" or r.get("early_stopped"):
            st.subheader("⚠️ High Risk — Clarification Needed")
            st.error("The model showed significant hallucination signals. Verify output carefully:")
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
    "HalluScope V5 • Three-Stage Hallucination Detection • "
    "KV-cached inference • Qwen2.5-3B-Instruct"
)
