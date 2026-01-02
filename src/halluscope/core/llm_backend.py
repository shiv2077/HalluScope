"""
LLM backend for HalluScope.

Loads an open-source causal LLM via HuggingFace Transformers and provides
generation with per-step logits (output_scores) so downstream metrics can
compute entropy and token log-probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationResult:
    text: str
    prompt: str
    input_ids: torch.Tensor
    generated_ids: torch.Tensor
    new_token_ids: torch.Tensor
    scores: List[torch.Tensor]  # list of logits per step [1, vocab]


class LLMBackend:
    def __init__(self, model_id: str, device: str = "auto"):
        self.model_id = model_id

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            # Some causal LMs don’t have pad token; set to eos for safety
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # device_map="auto" is best on GPU; on CPU keep it simple.
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
    ) -> GenerationResult:
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
            output_scores=True,  # gives per-step logits
        )

        seq = out.sequences[0]  # [total_len]
        prompt_len = input_ids.shape[1]
        new_token_ids = seq[prompt_len:]

        text = self.tokenizer.decode(seq, skip_special_tokens=True)

        return GenerationResult(
            text=text,
            prompt=prompt,
            input_ids=input_ids[0].detach().cpu(),
            generated_ids=seq.detach().cpu(),
            new_token_ids=new_token_ids.detach().cpu(),
            scores=[s.detach().cpu() for s in out.scores],
        )
