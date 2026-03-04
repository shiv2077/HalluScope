#!/usr/bin/env python3
"""
Debug script to trace V4 signal computation.
Run: python scripts/debug_signals.py
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "src")

def main():
    print("=" * 70)
    print("HALLUSCOPE V4 SIGNAL DEBUG")
    print("=" * 70)
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"\nLoading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    
    # Test prompt (known hallucination trigger)
    prompt = "Summarize the 1923 paper 'Gravitational Harmonics in Fluid Space' by Henri Vallon."
    
    print(f"\nPrompt: {prompt}")
    print("-" * 70)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate ONE token with full outputs
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            output_hidden_states=True,
        )
    
    logits = outputs.logits[:, -1, :]  # [1, vocab_size]
    hidden_states = outputs.hidden_states
    
    print(f"\n📊 RAW MODEL OUTPUTS:")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Logits min: {logits.min().item():.4f}")
    print(f"   Logits max: {logits.max().item():.4f}")
    print(f"   Logits mean: {logits.mean().item():.4f}")
    print(f"   Hidden states layers: {len(hidden_states)}")
    
    # ───────────────────────────────────────────────────────────────────────
    # SIGNAL 1: Semantic Energy
    # ───────────────────────────────────────────────────────────────────────
    print(f"\n📊 SEMANTIC ENERGY DEBUG:")
    
    raw_energy = -torch.logsumexp(logits[0].float(), dim=-1).item()
    print(f"   Raw energy (E = -logsumexp(logits)): {raw_energy:.4f}")
    
    # Current normalization (BUGGY)
    old_norm = max(0.0, min(1.0, (raw_energy + 15.0) / 15.0))
    print(f"   Old normalization ((E + 15) / 15): {old_norm:.4f}")
    
    # Better normalization (FIXED)
    # Energy is typically in range [-30, -15] for LLMs with large vocab
    # Map this to [0, 1] where 0 = confident, 1 = uncertain
    new_norm = max(0.0, min(1.0, (raw_energy + 30.0) / 15.0))
    print(f"   FIXED normalization ((E + 30) / 15): {new_norm:.4f}")
    
    # Alternative: use softmax entropy directly
    probs = F.softmax(logits[0], dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
    print(f"   Direct entropy: {entropy:.4f}")
    print(f"   Normalized entropy (/ 6.0): {min(1.0, entropy / 6.0):.4f}")
    
    # ───────────────────────────────────────────────────────────────────────
    # SIGNAL 4: Logit Lens Divergence (LLD)
    # ───────────────────────────────────────────────────────────────────────
    print(f"\n📊 LOGIT LENS DIVERGENCE (LLD) DEBUG:")
    
    print(f"   Number of layers: {len(hidden_states)}")
    
    # Check for lm_head
    if hasattr(model, 'lm_head'):
        print(f"   ✅ Found lm_head")
        unembed_weight = model.lm_head.weight
        print(f"   Unembed weight shape: {unembed_weight.shape}")
    else:
        print(f"   ❌ No lm_head found!")
        unembed_weight = None
    
    if unembed_weight is not None:
        # Analyze layer-by-layer
        num_layers = len(hidden_states)
        start_layer = int(num_layers * 0.5)  # Last 50% of layers
        
        print(f"   Analyzing layers {start_layer} to {num_layers-1}")
        
        divergences = []
        prev_probs = None
        
        for i in range(start_layer, num_layers):
            h_i = hidden_states[i][:, -1, :]  # [1, hidden_dim]
            
            # Project to vocabulary
            logits_i = torch.matmul(h_i.float(), unembed_weight.T.float())
            probs_i = F.softmax(logits_i, dim=-1)
            
            if prev_probs is not None:
                kl = F.kl_div(
                    (probs_i + 1e-9).log(),
                    (prev_probs + 1e-9),
                    reduction='batchmean',
                )
                divergences.append(kl.item())
                print(f"      Layer {i-1}→{i}: KL = {kl.item():.6f}")
            
            prev_probs = probs_i
        
        if divergences:
            mean_lld = np.mean(divergences)
            print(f"   Raw mean LLD: {mean_lld:.6f}")
            print(f"   Normalized LLD (/ 5.0): {min(1.0, mean_lld / 5.0):.6f}")
        else:
            print(f"   ❌ No divergences computed!")
    
    # ───────────────────────────────────────────────────────────────────────
    # SIGNAL 5: RCD (just check hidden state)
    # ───────────────────────────────────────────────────────────────────────
    print(f"\n📊 REPRESENTATION COLLAPSE (RCD) DEBUG:")
    
    last_hidden = hidden_states[-1][:, -1, :]
    print(f"   Last hidden state shape: {last_hidden.shape}")
    print(f"   Last hidden norm: {torch.norm(last_hidden).item():.4f}")
    
    # ───────────────────────────────────────────────────────────────────────
    # Suggested Fixes
    # ───────────────────────────────────────────────────────────────────────
    print(f"\n" + "=" * 70)
    print("📝 DIAGNOSIS:")
    print("=" * 70)
    
    if raw_energy < -15:
        print(f"\n⚠️  SEMANTIC ENERGY BUG CONFIRMED:")
        print(f"   Raw energy ({raw_energy:.2f}) is more negative than -15")
        print(f"   The normalization (E + 15) / 15 produces {old_norm:.4f}")
        print(f"   FIX: Change normalization to (E + 20) / 10 → {new_norm:.4f}")
    
    if unembed_weight is not None and divergences and mean_lld < 0.001:
        print(f"\n⚠️  LLD MAY BE TOO SMALL:")
        print(f"   Mean LLD ({mean_lld:.6f}) is very small")
        print(f"   This might indicate stable predictions (good) or a bug")
    
    print(f"\n✅ Debug complete. Use these values to fix llm_backend_v4.py")


if __name__ == "__main__":
    main()
