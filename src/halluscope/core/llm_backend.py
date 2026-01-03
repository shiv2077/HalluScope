"""
LLM backend for HalluScope.

Loads an open-source causal LLM via HuggingFace Transformers and provides
generation with per-step logits (output_scores) so downstream metrics can
compute entropy and token log-probabilities.

TokenBurn-style: Extracts logits for uncertainty estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationResult:
    """Result of LLM generation with uncertainty signals."""
    
    text: str
    prompt: str
    input_ids: torch.Tensor
    generated_ids: torch.Tensor
    new_token_ids: torch.Tensor
    scores: List[torch.Tensor]  # list of logits per step [1, vocab]
    vocab_size: int = 0
    # Optional top-k probabilities for efficient KL computation
    topk_token_ids: Optional[List[torch.Tensor]] = None  # each [k]
    topk_probs: Optional[List[torch.Tensor]] = None  # each [k]


class LLMBackend:
    """
    LLM backend with logits extraction for uncertainty estimation.
    
    Uses HuggingFace Transformers with output_scores=True to extract
    per-step logits for entropy, NLL, and variance computation.
    """
    
    def __init__(self, model_id: str, device: str = "auto"):
        self.model_id = model_id

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.vocab_size = self.tokenizer.vocab_size

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        ).eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: Optional[bool] = None,
        top_k_probs: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate text with per-step logits for uncertainty analysis.
        
        Args:
            prompt: Input prompt string.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.
            do_sample: Whether to sample (default: True if temperature > 0).
            top_k_probs: If set, compute top-k probabilities per step for
                         efficient KL divergence computation. Set to 50 for
                         a good balance of accuracy and memory.
        
        Returns:
            GenerationResult with text, token IDs, logits, and optional top-k probs.
        """
        if do_sample is None:
            do_sample = temperature > 0.0

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            return_dict_in_generate=True,
            output_scores=True,
        )

        seq = out.sequences[0]
        prompt_len = input_ids.shape[1]
        new_token_ids = seq[prompt_len:]

        text = self.tokenizer.decode(seq, skip_special_tokens=True)
        
        # Move scores to CPU
        scores = [s.detach().cpu() for s in out.scores]
        
        # Optionally compute top-k probabilities for KL divergence
        topk_token_ids: Optional[List[torch.Tensor]] = None
        topk_probs: Optional[List[torch.Tensor]] = None
        
        if top_k_probs is not None and top_k_probs > 0:
            topk_token_ids = []
            topk_probs = []
            k = min(top_k_probs, self.tokenizer.vocab_size)
            
            for logits in scores:
                # logits: [1, vocab_size]
                probs = torch.softmax(logits[0], dim=-1)
                top_probs, top_ids = torch.topk(probs, k)
                topk_token_ids.append(top_ids)
                topk_probs.append(top_probs)

        return GenerationResult(
            text=text,
            prompt=prompt,
            input_ids=input_ids[0].detach().cpu(),
            generated_ids=seq.detach().cpu(),
            new_token_ids=new_token_ids.detach().cpu(),
            scores=scores,
            vocab_size=self.tokenizer.vocab_size,
            topk_token_ids=topk_token_ids,
            topk_probs=topk_probs,
        )
