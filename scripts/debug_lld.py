#!/usr/bin/env python3
"""Debug LLD computation specifically."""

import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
import numpy as np

from halluscope.core.llm_backend_v4 import LLMBackend

def main():
    print("=" * 70)
    print("LLD SPECIFIC DEBUG")
    print("=" * 70)
    
    # Load backend
    print("\nLoading LLMBackend V4...")
    llm = LLMBackend(model_id="Qwen/Qwen2.5-3B-Instruct")
    
    # Check if model has output_hidden_states enabled
    print(f"\nModel config output_hidden_states: {llm.model.config.output_hidden_states}")
    
    # Test single forward pass
    prompt = "Hello, how are you?"
    inputs = llm.tokenizer(prompt, return_tensors="pt").to(llm.model.device)
    
    with torch.no_grad():
        outputs = llm.model(
            input_ids=inputs["input_ids"],
            output_hidden_states=True,
        )
    
    print(f"\nOutputs has hidden_states: {hasattr(outputs, 'hidden_states')}")
    
    if hasattr(outputs, 'hidden_states'):
        print(f"Hidden states is None: {outputs.hidden_states is None}")
        if outputs.hidden_states is not None:
            print(f"Number of layers: {len(outputs.hidden_states)}")
            print(f"First layer shape: {outputs.hidden_states[0].shape}")
            print(f"Last layer shape: {outputs.hidden_states[-1].shape}")
            
            # Test LLD computation directly
            print("\nComputing LLD...")
            lld_result = llm.compute_logit_lens_divergence(outputs.hidden_states)
            print(f"LLD result: {lld_result}")
            print(f"  lld: {lld_result.lld}")
            print(f"  num_layers_analyzed: {lld_result.num_layers_analyzed}")
            print(f"  divergences: {lld_result.layer_divergences[:5]}..." if lld_result.layer_divergences else "  divergences: []")
    
    # Check if lm_head exists
    print(f"\nModel has lm_head: {hasattr(llm.model, 'lm_head')}")
    if hasattr(llm.model, 'lm_head'):
        print(f"lm_head weight shape: {llm.model.lm_head.weight.shape}")
    
    # Now test generation with LLD tracking
    print("\n" + "=" * 70)
    print("TESTING GENERATION WITH LLD")
    print("=" * 70)
    
    result = llm.generate(
        prompt="What is 2+2?",
        max_new_tokens=10,
        temperature=0.7,
        debug=True,
    )
    
    print(f"\nGeneration result:")
    print(f"  Text: {result.text}")
    print(f"  LLD (aggregated): {result.logit_lens_divergence}")
    print(f"  per_token_lld: {result.per_token_lld}")
    print(f"  num tokens: {len(result.tokens)}")


if __name__ == "__main__":
    main()
