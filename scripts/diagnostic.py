#!/usr/bin/env python3
"""
HalluScope V5 — Deep Diagnostic Mode

Traces the entire three-stage hallucination detection pipeline step-by-step.
Produces a structured report for each prompt to identify:
- Whether hallucination signals are incorrectly computed
- Whether signal weights are miscalibrated
- Whether semantic embeddings are misleading
- Whether the model itself is producing unstable tokens
- Whether the HRI aggregation formula suppresses valid signals
- Stage 1 pre-hoc prompt risk analysis
- Stage 3 post-hoc verification signals

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src python scripts/diagnostic.py
"""

from __future__ import annotations

import io
import math
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ─── Global Debug Flag ───────────────────────────────────────────────────────
HALLUSCOPE_DEBUG = True

# ─── Import HalluScope internals ─────────────────────────────────────────────
from halluscope.core.llm_backend_v4 import (
    LLMBackend,
    Regime,
    WEIGHT_PROFILES,
    _detect_regime,
    _sigmoid,
    _safe_mean,
    _safe_variance,
    _cosine_similarity,
    _compute_hri_slope,
    CSGResult,
    LLDResult,
    ENTROPY_MIN_TOKENS,
    SEMANTIC_MIN_TOKENS,
    LLD_MIN_TOKENS,
    RCD_MIN_TOKENS,
    EARLY_STOP_MIN_TOKENS,
    CSG_MAX,
    LLD_MAX,
    RCD_MAX,
    NOVELTY_MAX,
    ENTITY_MAX,
    CLAIM_MAX,
    _compute_entity_expansion,
    _compute_claim_density,
    CSG_COMPUTE_INTERVAL,
    LLD_COMPUTE_INTERVAL,
    LLD_SAMPLE_LAYERS,
    LLD_START_LAYER_FRACTION,
    HRI_THRESHOLD,
    HRI_INTERVENTION_THRESHOLD,
    TOP_K_ENTROPY,
    SLIDING_WINDOW_SIZE,
    STOP_SEQUENCES,
)


# ─── Diagnostic Writer ───────────────────────────────────────────────────────

class DiagnosticWriter:
    """Tee — writes to both stdout and a buffer for file output."""

    def __init__(self):
        self.buffer = io.StringIO()

    def write(self, text: str):
        sys.stdout.write(text)
        sys.stdout.flush()
        self.buffer.write(text)

    def section(self, title: str, char: str = "=", width: int = 72):
        border = char * width
        self.write(f"\n{border}\n{title}\n{border}\n")

    def kv(self, key: str, value: Any, indent: int = 2):
        pad = " " * indent
        self.write(f"{pad}{key}: {value}\n")

    def blank(self):
        self.write("\n")

    def line(self, text: str = "", indent: int = 0):
        pad = " " * indent
        self.write(f"{pad}{text}\n")

    def contents(self) -> str:
        return self.buffer.getvalue()


# ─── Failure Mode Heuristics ─────────────────────────────────────────────────

def detect_failure_modes(
    tokens: List[str],
    entropy_vals: List[float],
    lld_vals: List[float],
    out: DiagnosticWriter,
):
    out.section("FAILURE MODE FLAGS")

    flags = []

    # 1. Repetition collapse
    if len(tokens) >= 6:
        bigrams = [tokens[i] + tokens[i + 1] for i in range(len(tokens) - 1)]
        counts = Counter(bigrams)
        most_common_count = counts.most_common(1)[0][1] if counts else 0
        rep_ratio = most_common_count / max(len(bigrams), 1)
        out.kv("bigram_repetition_ratio", f"{rep_ratio:.3f}")
        if rep_ratio > 0.3:
            flags.append("REPETITION_COLLAPSE")
            out.line("  ⚠️  WARNING: repetition collapse detected (bigram ratio > 0.3)")

    # 2. Symbolic / digit generation  
    digit_tokens = [t for t in tokens if any(c.isdigit() for c in t)]
    digit_ratio = len(digit_tokens) / max(len(tokens), 1)
    out.kv("digit_token_ratio", f"{digit_ratio:.3f}")
    if digit_ratio > 0.5:
        digit_ents = [entropy_vals[i] for i, t in enumerate(tokens) if any(c.isdigit() for c in t) and i < len(entropy_vals)]
        avg_dent = _safe_mean(digit_ents)
        out.kv("digit_sequence_entropy", f"{avg_dent:.4f}")
        if avg_dent < 0.05:
            flags.append("SYMBOLIC_GENERATION")
            out.line("  ⚠️  WARNING: symbolic generation (digit entropy < 0.05)")

    # 3. LLD spikes
    if lld_vals:
        lld_arr = np.array(lld_vals)
        lld_mean = np.mean(lld_arr)
        lld_std = np.std(lld_arr) if len(lld_arr) > 1 else 0.0
        spikes = int(np.sum(lld_arr > lld_mean + 2 * lld_std)) if lld_std > 0 else 0
        out.kv("lld_spike_count", spikes)
        if spikes >= 3:
            flags.append("INTERNAL_REASONING_INSTABILITY")
            out.line("  ⚠️  WARNING: internal reasoning instability (LLD spikes >= 3)")

    # 4. Entropy plateau (near-zero throughout = overconfident)
    if entropy_vals:
        low_ent_count = sum(1 for e in entropy_vals if e < 0.02)
        low_ent_ratio = low_ent_count / len(entropy_vals)
        out.kv("low_entropy_ratio (<0.02)", f"{low_ent_ratio:.3f}")
        if low_ent_ratio > 0.8:
            flags.append("OVERCONFIDENT_GENERATION")
            out.line("  ⚠️  WARNING: overconfident generation (>80% tokens near-zero entropy)")

    # 5. Non-ASCII / language drift
    non_ascii_tokens = [t for t in tokens if any(ord(c) > 127 for c in t)]
    non_ascii_ratio = len(non_ascii_tokens) / max(len(tokens), 1)
    out.kv("non_ascii_token_ratio", f"{non_ascii_ratio:.3f}")
    if non_ascii_ratio > 0.3:
        flags.append("LANGUAGE_DRIFT")
        out.line("  ⚠️  WARNING: language drift (>30% non-ASCII tokens)")

    if not flags:
        out.line("  ✅ No failure modes detected.")
    else:
        out.line(f"\n  Active flags: {flags}")


# ─── Main Diagnostic Runner ──────────────────────────────────────────────────

@torch.no_grad()
def run_diagnostic(
    backend: LLMBackend,
    prompt: str,
    prompt_idx: int,
    out: DiagnosticWriter,
    max_new_tokens: int = 60,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 50,
):
    """Run full diagnostic pipeline on a single prompt."""

    do_sample = temperature > 0.0

    out.section(f"DIAGNOSTIC RUN #{prompt_idx}", char="█", width=72)
    out.line(f'Prompt: "{prompt}"')
    out.blank()

    t0 = time.perf_counter()

    # ─────────────────────────────────────────────────────────────────────
    # Part 2: Prompt Analysis Report
    # ─────────────────────────────────────────────────────────────────────
    out.section("PROMPT ANALYSIS")

    prompt_tokens = backend.tokenizer.encode(prompt, add_special_tokens=False)
    prompt_token_count = len(prompt_tokens)
    regime = _detect_regime(prompt, backend.tokenizer)

    # Prompt embedding
    prompt_emb = backend.embedder.encode(prompt, convert_to_tensor=True, normalize_embeddings=False)
    prompt_emb_norm = torch.norm(prompt_emb.float()).item()
    prompt_emb_normalized = F.normalize(prompt_emb.unsqueeze(0), dim=-1).squeeze(0)

    # Complexity heuristic
    unique_tokens = len(set(prompt_tokens))
    complexity = unique_tokens / max(prompt_token_count, 1)

    # Type guess (extended from regime)
    has_digits = any(c.isdigit() for c in prompt)
    has_math = any(op in prompt for op in ["+", "-", "*", "/", "=", "^"])
    if has_math and has_digits:
        type_guess = "SYMBOLIC"
    elif prompt_token_count <= 6:
        type_guess = "SHORT"
    elif prompt_token_count <= 20:
        type_guess = "FACTUAL"
    else:
        type_guess = "GENERATIVE"

    out.kv("prompt_text", f'"{prompt}"')
    out.kv("prompt_length_tokens", prompt_token_count)
    out.kv("prompt_token_ids", prompt_tokens[:20])
    out.kv("prompt_embedding_norm", f"{prompt_emb_norm:.6f}")
    out.kv("prompt_complexity_score", f"{complexity:.4f} ({unique_tokens} unique / {prompt_token_count} total)")
    out.kv("prompt_type_guess", type_guess)
    out.kv("regime_detected", regime)
    out.kv("weights_profile", regime)

    # ─────────────────────────────────────────────────────────────────────
    # Part 3: Model Configuration Report
    # ─────────────────────────────────────────────────────────────────────
    out.section("MODEL CONFIGURATION")

    precision = "4-bit NF4 (compute fp16)" if backend.device == "cuda" else "fp32"
    out.kv("model_name", backend.model_id)
    out.kv("precision", precision)
    out.kv("device", str(backend.model.device))
    out.kv("num_hidden_layers", backend.model.config.num_hidden_layers)
    out.kv("hidden_size", backend.model.config.hidden_size)
    out.kv("vocab_size", backend.vocab_size)
    out.kv("temperature", temperature)
    out.kv("top_p", top_p)
    out.kv("top_k", top_k)
    out.kv("do_sample", do_sample)
    out.kv("max_new_tokens", max_new_tokens)
    out.kv("kv_cache_enabled", True)
    out.kv("lld_layers", backend._lld_layers)
    out.kv("lld_compute_interval", LLD_COMPUTE_INTERVAL)
    out.kv("csg_compute_interval", CSG_COMPUTE_INTERVAL)
    if hasattr(backend, '_unembed_weight') and backend._unembed_weight is not None:
        out.kv("unembed_shape", str(backend._unembed_weight.shape))
        out.kv("unembed_device", str(backend._unembed_weight.device))

    # ─────────────────────────────────────────────────────────────────────
    # Part 9: Signal Weighting Report
    # ─────────────────────────────────────────────────────────────────────
    weights = WEIGHT_PROFILES[regime]
    out.section("HRI WEIGHTS")
    out.kv("regime", regime)
    for k, v in weights.items():
        out.kv(k, v)

    # ─────────────────────────────────────────────────────────────────────
    # Generation with full instrumentation
    # ─────────────────────────────────────────────────────────────────────
    out.section("TOKEN-LEVEL GENERATION TRACE")

    # Reset state
    backend._prompt_embedding = None
    backend._prev_hidden = None

    # Tokenize + prefill
    inputs = backend.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(backend.model.device)

    prefill = backend.model(input_ids=input_ids, output_hidden_states=True, use_cache=True)
    past_kv = prefill.past_key_values

    # Store prefill hidden for RCD
    prefill_last_hidden = prefill.hidden_states[-1][:, -1, :].squeeze(0)
    backend._prev_hidden = prefill_last_hidden.clone().detach()

    prefill_hidden_norm = torch.norm(prefill_last_hidden.float()).item()
    out.line(f"  Prefill hidden state norm: {prefill_hidden_norm:.4f}")
    out.line(f"  Prefill hidden state shape: {prefill.hidden_states[-1].shape}")
    out.line(f"  Number of hidden layers returned: {len(prefill.hidden_states)}")

    # Storage
    tokens_str: List[str] = []
    tokens_id: List[int] = []
    entropy_vals: List[float] = []
    energy_vals: List[float] = []
    logprob_vals: List[float] = []
    risk_vals: List[float] = []
    hri_vals: List[float] = []
    pt_lld: List[float] = []
    pt_rcd: List[float] = []
    pt_csg: List[float] = []

    # Extended diagnostic storage
    hidden_norms: List[float] = []
    hidden_cosines: List[float] = []
    raw_energies: List[float] = []
    raw_entropies: List[float] = []
    top_logprobs_log: List[List[Tuple[str, float]]] = []
    lld_detail_log: List[Optional[Dict]] = []
    csg_detail_log: List[Optional[Dict]] = []

    cached_csg = 0.0
    cached_lld = 0.0
    cached_rcd = 0.0
    cached_novelty = 0.0
    cached_entity = 0.0
    cached_claim = 0.0
    early_stopped = False
    intervention_triggered = False
    intervention_msg = None

    logits = prefill.logits[:, -1, :]  # [1, vocab]
    prev_hidden = prefill_last_hidden.clone()

    for step in range(max_new_tokens):
        # Temperature scaling
        sc_logits = logits / temperature if temperature > 0 else logits
        probs = F.softmax(sc_logits[0], dim=-1)

        # ── Signal 1: Semantic Energy (raw) ──
        raw_energy = backend.compute_semantic_energy(logits[0])
        norm_energy = max(0.0, min(1.0, (raw_energy + 30.0) / 15.0))
        energy_vals.append(norm_energy)
        raw_energies.append(raw_energy)

        # ── Signal 2: Entropy (raw) ──
        raw_ent = backend.compute_token_entropy(probs)
        norm_ent = min(1.0, raw_ent / 6.0)
        entropy_vals.append(norm_ent)
        raw_entropies.append(raw_ent)

        # ── Top logprobs ──
        top_vals, top_ids = torch.topk(probs, min(5, probs.size(-1)))
        top_list = []
        for vi, ii in zip(top_vals.tolist(), top_ids.tolist()):
            tok_str = backend.tokenizer.decode([ii])
            lp_val = math.log(vi + 1e-12)
            top_list.append((tok_str, lp_val))
        top_logprobs_log.append(top_list)

        # ── Sample ──
        if do_sample:
            if top_k > 0:
                tk_p, tk_i = torch.topk(probs, top_k)
                tk_p = tk_p / tk_p.sum()
                sp, si = torch.sort(tk_p, descending=True)
                cm = torch.cumsum(sp, dim=-1)
                mask = cm <= top_p
                mask[0] = True
                fp = sp * mask
                fp = fp / fp.sum()
                sx = torch.multinomial(fp, 1)
                li = si[sx]
                next_tok = tk_i[li].item()
            else:
                next_tok = torch.multinomial(probs, 1).item()
        else:
            next_tok = torch.argmax(probs).item()

        lp = torch.log(probs[next_tok] + 1e-12).item()
        logprob_vals.append(lp)

        tokens_id.append(next_tok)
        ts = backend.tokenizer.decode([next_tok])
        tokens_str.append(ts)
        risk_vals.append(backend._token_risk(raw_ent, lp))

        # ── Decide if expensive signals needed ──
        need_hidden = (step >= LLD_MIN_TOKENS and step % LLD_COMPUTE_INTERVAL == 0)

        # ── KV-cached forward ──
        nt = torch.tensor([[next_tok]], device=backend.model.device)
        out_model = backend.model(
            input_ids=nt,
            past_key_values=past_kv,
            output_hidden_states=need_hidden,
            use_cache=True,
        )
        past_kv = out_model.past_key_values
        logits = out_model.logits[:, -1:, :].squeeze(1)

        # ── Hidden state diagnostics (Part 5) ──
        h_norm = 0.0
        h_cos = 0.0
        lld_detail = None
        csg_detail = None

        if need_hidden and out_model.hidden_states is not None:
            cur_hidden = out_model.hidden_states[-1][:, -1, :].squeeze(0)
            h_norm = torch.norm(cur_hidden.float()).item()
            h_cos = _cosine_similarity(cur_hidden, prev_hidden)
            prev_hidden = cur_hidden.clone()

            # ── LLD (Part 6 — Logit Lens) ──
            lld_result = backend.compute_lld(out_model.hidden_states)
            cached_lld = min(lld_result.lld, LLD_MAX)

            # Detailed logit lens report
            layer_top_tokens = {}
            if backend._unembed_weight is not None:
                for idx in backend._lld_layers:
                    if idx < len(out_model.hidden_states):
                        h_layer = out_model.hidden_states[idx][:, -1, :]
                        logits_layer = torch.matmul(h_layer.float(), backend._unembed_weight.float().T)
                        top_id = torch.argmax(logits_layer, dim=-1).item()
                        layer_top_tokens[idx] = backend.tokenizer.decode([top_id])

            lld_detail = {
                "lld_score": cached_lld,
                "layer_divergences": lld_result.layer_divergences,
                "num_layers_analyzed": lld_result.num_layers_analyzed,
                "layer_top_tokens": layer_top_tokens,
            }

            # ── RCD ──
            if step >= RCD_MIN_TOKENS:
                cached_rcd_val, _ = backend.compute_rcd(cur_hidden)
                cached_rcd = min(cached_rcd_val, RCD_MAX)
            else:
                backend._prev_hidden = cur_hidden.clone().detach()

        hidden_norms.append(h_norm)
        hidden_cosines.append(h_cos)
        pt_lld.append(cached_lld)
        pt_rcd.append(cached_rcd)
        lld_detail_log.append(lld_detail)

        # ── CSG (Part 7, 8) ──
        if step >= SEMANTIC_MIN_TOKENS and step % CSG_COMPUTE_INTERVAL == 0:
            gen_text_so_far = "".join(tokens_str)
            csg_result = backend.compute_csg(prompt, gen_text_so_far, _safe_mean(logprob_vals))
            cached_csg = min(csg_result.csg, CSG_MAX)

            # Compute window embedding diagnostics
            window_text = "".join(tokens_str[-SLIDING_WINDOW_SIZE:])
            window_emb = backend.embedder.encode(window_text, convert_to_tensor=True, normalize_embeddings=False)
            window_norm = torch.norm(window_emb.float()).item()

            prompt_emb_raw = backend.embedder.encode(prompt, convert_to_tensor=True, normalize_embeddings=False)
            prompt_norm_raw = torch.norm(prompt_emb_raw.float()).item()

            cos_sim = F.cosine_similarity(
                prompt_emb_raw.unsqueeze(0).float(), window_emb.unsqueeze(0).float(), dim=1
            ).item()

            csg_detail = {
                "confidence": csg_result.confidence,
                "semantic_stability": csg_result.semantic_stability,
                "csg": csg_result.csg,
                "prompt_embedding_norm": prompt_norm_raw,
                "window_embedding_norm": window_norm,
                "cosine_similarity": cos_sim,
            }
            # V4.2: Semantic fabrication signals
            cached_novelty = min(1.0 - csg_result.semantic_stability, NOVELTY_MAX)
            cached_entity = min(_compute_entity_expansion(prompt, gen_text_so_far), ENTITY_MAX)
            cached_claim = min(_compute_claim_density(gen_text_so_far), CLAIM_MAX)
        else:
            csg_detail = None

        pt_csg.append(cached_csg)
        csg_detail_log.append(csg_detail)

        # ── Running HRI (Part 10, 11) ──
        m_ent = _safe_mean(entropy_vals)
        m_eng = _safe_mean(energy_vals)
        n_var = min(1.0, _safe_variance(logprob_vals) / 4.0)
        m_lld = _safe_mean(pt_lld) if step >= LLD_MIN_TOKENS else 0.0
        m_rcd = _safe_mean(pt_rcd)

        # Raw logit computation (Part 11)
        raw_logit = (
            weights["w_entropy"] * m_ent * 10
            + weights["w_energy"] * m_eng * 10
            + weights["w_variance"] * n_var * 10
            + weights["w_csg"] * pt_csg[-1] * 10
            + weights["w_lld"] * m_lld * 10
            + weights["w_rcd"] * m_rcd * 10
            + weights.get("w_novelty", 0.0) * cached_novelty * 10
            + weights.get("w_entity", 0.0) * cached_entity * 10
            + weights.get("w_claim", 0.0) * cached_claim * 10
            + weights["bias"]
        )
        hri = _sigmoid(raw_logit)
        hri_vals.append(hri)

        # ── TOKEN TRACE (Part 4) ──
        out.line(f"\n  === TOKEN STEP {step} ===")
        out.kv("token", f'"{ts}"', indent=4)
        out.kv("token_id", next_tok, indent=4)
        out.kv("logprob", f"{lp:.4f}", indent=4)
        out.line(f"    top_logprobs:", indent=0)
        for tok_s, lp_v in top_list:
            out.line(f"        {tok_s!r:20s}: {lp_v:.4f}")

        out.kv("entropy (raw)", f"{raw_ent:.4f}", indent=4)
        out.kv("entropy (norm)", f"{norm_ent:.4f}", indent=4)
        out.kv("semantic_energy (raw)", f"{raw_energy:.4f}", indent=4)
        out.kv("semantic_energy (norm)", f"{norm_energy:.4f}", indent=4)
        out.kv("logprob_variance (running)", f"{n_var:.4f}", indent=4)
        out.kv("confidence (running)", f"{_sigmoid(-_safe_mean(logprob_vals)):.4f}", indent=4)
        out.kv("token_risk", f"{risk_vals[-1]:.4f}", indent=4)

        # Hidden state diagnostics
        if need_hidden:
            out.line(f"    --- Hidden State Diagnostics ---")
            out.kv("hidden_state_norm", f"{h_norm:.4f}", indent=6)
            out.kv("hidden_state_cosine_with_previous", f"{h_cos:.4f}", indent=6)
            collapse_score = max(0.0, (h_cos - 0.5) * 2.0) if h_cos > 0 else 0.0
            out.kv("representation_collapse_score", f"{collapse_score:.4f}", indent=6)

        # Logit lens report
        if lld_detail:
            out.line(f"    --- Logit Lens Report ---")
            for layer_idx, top_tok in lld_detail["layer_top_tokens"].items():
                out.kv(f"layer {layer_idx} top token", f'"{top_tok}"', indent=6)
            out.kv("KL divergences", [f"{d:.4f}" for d in lld_detail["layer_divergences"]], indent=6)
            out.kv("LLD score", f"{lld_detail['lld_score']:.4f}", indent=6)

        # Semantic stability report
        if csg_detail:
            out.line(f"    --- Semantic Stability ---")
            out.kv("prompt_embedding_norm", f"{csg_detail['prompt_embedding_norm']:.4f}", indent=6)
            out.kv("window_embedding_norm", f"{csg_detail['window_embedding_norm']:.4f}", indent=6)
            out.kv("cosine_similarity", f"{csg_detail['cosine_similarity']:.4f}", indent=6)
            out.kv("semantic_stability_score", f"{csg_detail['semantic_stability']:.4f}", indent=6)
            out.line(f"    --- CSG Breakdown ---")
            out.kv("confidence", f"{csg_detail['confidence']:.4f}", indent=6)
            out.kv("semantic_stability", f"{csg_detail['semantic_stability']:.4f}", indent=6)
            out.kv("CSG = confidence * (1 - stability)", f"{csg_detail['csg']:.4f}", indent=6)

        # Signal vector + HRI
        out.line(f"    --- Signal Vector ---")
        out.kv("entropy (mean)", f"{m_ent:.4f}", indent=6)
        out.kv("semantic_energy (mean)", f"{m_eng:.4f}", indent=6)
        out.kv("logprob_variance", f"{n_var:.4f}", indent=6)
        out.kv("CSG", f"{pt_csg[-1]:.4f}", indent=6)
        out.kv("LLD", f"{m_lld:.4f}", indent=6)
        out.kv("RCD", f"{m_rcd:.4f}", indent=6)
        out.kv("novelty", f"{cached_novelty:.4f}", indent=6)
        out.kv("entity_expansion", f"{cached_entity:.4f}", indent=6)
        out.kv("claim_density", f"{cached_claim:.4f}", indent=6)

        out.line(f"    --- HRI Calculation ---")
        out.kv("raw_logit", f"{raw_logit:.6f}", indent=6)
        out.line(f"      = {weights['w_entropy']}*{m_ent:.4f}*10"
                 f" + {weights['w_energy']}*{m_eng:.4f}*10"
                 f" + {weights['w_variance']}*{n_var:.4f}*10"
                 f" + {weights['w_csg']}*{pt_csg[-1]:.4f}*10"
                 f" + {weights['w_lld']}*{m_lld:.4f}*10"
                 f" + {weights['w_rcd']}*{m_rcd:.4f}*10"
                 f" + {weights.get('w_novelty', 0.0)}*{cached_novelty:.4f}*10"
                 f" + {weights.get('w_entity', 0.0)}*{cached_entity:.4f}*10"
                 f" + {weights.get('w_claim', 0.0)}*{cached_claim:.4f}*10"
                 f" + ({weights['bias']})")
        out.kv("sigmoid(raw_logit) = HRI", f"{hri:.6f}", indent=6)

        # ── Intervention ──
        if hri >= HRI_INTERVENTION_THRESHOLD and not intervention_triggered and step >= SEMANTIC_MIN_TOKENS:
            intervention_triggered = True
            intervention_msg = f"⚠️ HRI={hri:.2f} — possible hallucination detected."
            out.line(f"    🚨 INTERVENTION TRIGGERED: {intervention_msg}")

        # ── Early stop check (Part 13) ──
        if step >= EARLY_STOP_MIN_TOKENS and hri >= HRI_THRESHOLD:
            sl = _compute_hri_slope(hri_vals, 4)
            out.line(f"    --- Early Stop Check ---")
            out.kv("tokens_generated", step + 1, indent=6)
            out.kv("HRI", f"{hri:.6f}", indent=6)
            out.kv("threshold", HRI_THRESHOLD, indent=6)
            out.kv("HRI_slope", f"{sl:.6f}", indent=6)
            if sl > 0:
                out.kv("decision", "STOP (slope > 0)", indent=6)
                early_stopped = True
                break
            else:
                out.kv("decision", "CONTINUE (slope <= 0)", indent=6)

        # EOS
        if next_tok == backend.tokenizer.eos_token_id:
            out.line(f"    [EOS token reached]")
            break

        # Stop sequences
        cur = "".join(tokens_str)
        found = False
        for ss in STOP_SEQUENCES:
            if ss in cur:
                out.line(f"    [Stop sequence detected: {ss!r}]")
                found = True
                break
        if found:
            break

    gen_text = "".join(tokens_str)
    for ss in STOP_SEQUENCES:
        if ss in gen_text:
            gen_text = gen_text.split(ss)[0]
    gen_text = gen_text.rstrip()

    elapsed = time.perf_counter() - t0

    # ─────────────────────────────────────────────────────────────────────
    # Part 12: Trajectory Monitoring
    # ─────────────────────────────────────────────────────────────────────
    out.section("HRI TRAJECTORY")
    for i, h in enumerate(hri_vals):
        marker = ""
        if h >= HRI_THRESHOLD:
            marker = " ◀ ABOVE THRESHOLD"
        elif h >= HRI_INTERVENTION_THRESHOLD:
            marker = " ◀ intervention zone"
        out.line(f"  step {i:3d}: HRI={h:.6f}{marker}")

    if len(hri_vals) >= 3:
        slope = (hri_vals[-1] - hri_vals[0]) / max(len(hri_vals) - 1, 1)
        out.blank()
        out.kv("HRI slope (first→last)", f"{slope:.6f}")
        if len(hri_vals) >= 6:
            first_half = hri_vals[:len(hri_vals)//2]
            second_half = hri_vals[len(hri_vals)//2:]
            s1 = (first_half[-1] - first_half[0]) / max(len(first_half) - 1, 1)
            s2 = (second_half[-1] - second_half[0]) / max(len(second_half) - 1, 1)
            out.kv("HRI acceleration (slope2 - slope1)", f"{s2 - s1:.6f}")

    # ─────────────────────────────────────────────────────────────────────
    # Part 14: Final Generation Report
    # ─────────────────────────────────────────────────────────────────────
    out.section("FINAL GENERATION REPORT")

    out.kv("generated_text", f'"{gen_text}"')
    out.kv("token_count", len(tokens_str))
    out.kv("generation_time", f"{elapsed:.2f}s ({elapsed / max(len(tokens_str), 1) * 1000:.0f}ms/tok)")
    out.kv("early_stopped", early_stopped)
    out.kv("intervention_triggered", intervention_triggered)
    out.blank()

    ent_mean = _safe_mean(entropy_vals)
    ent_max = max(entropy_vals) if entropy_vals else 0.0
    eng_mean = _safe_mean(energy_vals)
    lp_var = _safe_variance(logprob_vals)
    lp_std = math.sqrt(lp_var) if lp_var > 0 else 0.0
    nll = -_safe_mean(logprob_vals) if logprob_vals else 0.0
    f_csg = pt_csg[-1] if pt_csg else 0.0
    f_lld = _safe_mean(pt_lld)
    f_rcd = _safe_mean(pt_rcd)
    final_hri = hri_vals[-1] if hri_vals else 0.0
    max_hri = max(hri_vals) if hri_vals else 0.0

    # Risk classification
    if final_hri < 0.45:
        risk_level = "LOW"
    elif final_hri < 0.70:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    out.kv("average_entropy", f"{ent_mean:.4f}")
    out.kv("max_entropy", f"{ent_max:.4f}")
    out.kv("average_semantic_energy", f"{eng_mean:.4f}")
    out.kv("average_logprob_variance", f"{lp_var:.4f}")
    out.kv("logprob_std", f"{lp_std:.4f}")
    out.kv("nll", f"{nll:.4f}")
    out.kv("perplexity", f"{math.exp(nll) if nll < 20 else float('inf'):.4f}")
    out.blank()
    out.kv("final_CSG", f"{f_csg:.4f}")
    out.kv("final_LLD", f"{f_lld:.4f}")
    out.kv("final_RCD", f"{f_rcd:.4f}")
    out.blank()
    out.kv("max_HRI", f"{max_hri:.6f}")
    out.kv("final_HRI", f"{final_hri:.6f}")
    out.kv("risk_level", risk_level)
    out.blank()

    # Final full signal vector
    out.section("FINAL SIGNAL VECTOR")
    out.kv("entropy", f"{ent_mean:.6f}")
    out.kv("semantic_energy", f"{eng_mean:.6f}")
    out.kv("logprob_variance", f"{min(1.0, lp_var / 4.0):.6f}")
    out.kv("CSG", f"{f_csg:.6f}")
    out.kv("LLD", f"{f_lld:.6f}")
    out.kv("RCD", f"{f_rcd:.6f}")

    final_csg_result = backend.compute_csg(prompt, gen_text, _safe_mean(logprob_vals)) if logprob_vals else CSGResult()
    out.kv("confidence", f"{final_csg_result.confidence:.6f}")
    out.kv("semantic_stability", f"{final_csg_result.semantic_stability:.6f}")

    # V4.2 Semantic fabrication signals (final)
    f_novelty = 1.0 - final_csg_result.semantic_stability
    f_entity = _compute_entity_expansion(prompt, gen_text)
    f_claim = _compute_claim_density(gen_text)
    out.kv("novelty", f"{f_novelty:.6f}")
    out.kv("entity_expansion", f"{f_entity:.6f}")
    out.kv("claim_density", f"{f_claim:.6f}")

    # Final HRI calculation breakdown
    out.section("FINAL HRI CALCULATION BREAKDOWN")
    n_var_final = min(1.0, lp_var / 4.0)
    raw_logit_final = (
        weights["w_entropy"] * ent_mean * 10
        + weights["w_energy"] * eng_mean * 10
        + weights["w_variance"] * n_var_final * 10
        + weights["w_csg"] * f_csg * 10
        + weights["w_lld"] * f_lld * 10
        + weights["w_rcd"] * f_rcd * 10
        + weights.get("w_novelty", 0.0) * f_novelty * 10
        + weights.get("w_entity", 0.0) * f_entity * 10
        + weights.get("w_claim", 0.0) * f_claim * 10
        + weights["bias"]
    )
    hri_final = _sigmoid(raw_logit_final)

    out.line(f"  raw_logit = ")
    out.line(f"    {weights['w_entropy']} * {ent_mean:.4f} * 10  (entropy)      = {weights['w_entropy'] * ent_mean * 10:.4f}")
    out.line(f"  + {weights['w_energy']} * {eng_mean:.4f} * 10  (energy)        = {weights['w_energy'] * eng_mean * 10:.4f}")
    out.line(f"  + {weights['w_variance']} * {n_var_final:.4f} * 10  (variance)  = {weights['w_variance'] * n_var_final * 10:.4f}")
    out.line(f"  + {weights['w_csg']} * {f_csg:.4f} * 10  (CSG)            = {weights['w_csg'] * f_csg * 10:.4f}")
    out.line(f"  + {weights['w_lld']} * {f_lld:.4f} * 10  (LLD)            = {weights['w_lld'] * f_lld * 10:.4f}")
    out.line(f"  + {weights['w_rcd']} * {f_rcd:.4f} * 10  (RCD)            = {weights['w_rcd'] * f_rcd * 10:.4f}")
    out.line(f"  + {weights.get('w_novelty', 0.0)} * {f_novelty:.4f} * 10  (novelty)    = {weights.get('w_novelty', 0.0) * f_novelty * 10:.4f}")
    out.line(f"  + {weights.get('w_entity', 0.0)} * {f_entity:.4f} * 10  (entity_exp)  = {weights.get('w_entity', 0.0) * f_entity * 10:.4f}")
    out.line(f"  + {weights.get('w_claim', 0.0)} * {f_claim:.4f} * 10  (claim)       = {weights.get('w_claim', 0.0) * f_claim * 10:.4f}")
    out.line(f"  + {weights['bias']}  (bias)")
    out.line(f"  = {raw_logit_final:.6f}")
    out.blank()
    out.line(f"  sigmoid({raw_logit_final:.6f}) = {hri_final:.6f}")
    out.kv("FINAL HRI", f"{hri_final:.6f}")
    out.kv("RISK", risk_level)

    # ─────────────────────────────────────────────────────────────────────
    # Part 15: Failure Mode Flags
    # ─────────────────────────────────────────────────────────────────────
    detect_failure_modes(tokens_str, entropy_vals, pt_lld, out)

    out.line(f"\n{'─'*72}")
    out.line(f"End of diagnostic for prompt #{prompt_idx}")
    out.line(f"{'─'*72}\n\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

TEST_PROMPTS = [
    'Summarize the 1923 paper "Gravitational Harmonics in Fluid Space" by Henri Vallon',
    "What is 2 + 2",
    "Who discovered penicillin",
    "Explain quantum entanglement",
    "What theorem did John Smith prove in 1993",
]


def main():
    out = DiagnosticWriter()

    out.section("HALLUSCOPE V4.1 — DEEP DIAGNOSTIC REPORT", char="█", width=72)
    out.line(f"HALLUSCOPE_DEBUG = {HALLUSCOPE_DEBUG}")
    out.line(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out.line(f"Torch: {torch.__version__}")
    out.line(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        out.line(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        out.line(f"VRAM: {mem:.1f} GiB")
    out.blank()

    # Load model
    out.line("Loading HalluScope backend...")
    t_load = time.perf_counter()
    backend = LLMBackend(model_id="Qwen/Qwen2.5-3B-Instruct", device="auto")
    out.line(f"Backend loaded in {time.perf_counter() - t_load:.1f}s")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        out.line(f"GPU memory: {alloc:.2f} GiB allocated, {reserved:.2f} GiB reserved")

    out.blank()
    out.line(f"Running {len(TEST_PROMPTS)} diagnostic prompts...")
    out.line(f"Temperature: 0.0 (greedy) for deterministic reproduction")

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        run_diagnostic(
            backend=backend,
            prompt=prompt,
            prompt_idx=i,
            out=out,
            max_new_tokens=60,
            temperature=0.0,  # greedy for reproducibility
        )

    # Write report
    out.section("END OF DIAGNOSTIC REPORT", char="█", width=72)

    report_path = "halluscope_debug_report.txt"
    with open(report_path, "w") as f:
        f.write(out.contents())

    print(f"\n✅ Report saved to: {report_path}")
    print(f"   Size: {len(out.contents()):,} bytes")


if __name__ == "__main__":
    main()
