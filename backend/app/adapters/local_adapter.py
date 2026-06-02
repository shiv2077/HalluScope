"""HalluScope - Local HuggingFace adapter.

Provides a local-model implementation of BaseLLMAdapter so the same
risk pipeline can run without external API calls.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.app.adapters.base_adapter import BaseLLMAdapter
from backend.app.adapters.schemas import StandardizedResult
from backend.app.config import DEFAULT_LOCAL_MODEL


class LocalAdapter(BaseLLMAdapter):
    """Adapter that runs generation on a local HuggingFace model."""

    _loaded_model_id: str | None = None
    _tokenizer = None
    _model = None
    _device = "cpu"

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL):
        self.model = model
        self._ensure_model_loaded(model)

    @classmethod
    def _ensure_model_loaded(cls, model_id: str | None = None) -> None:
        target = model_id or DEFAULT_LOCAL_MODEL
        if cls._model is not None and cls._loaded_model_id == target:
            return

        cls._tokenizer = AutoTokenizer.from_pretrained(target)
        if cls._tokenizer.pad_token_id is None:
            cls._tokenizer.pad_token = cls._tokenizer.eos_token

        cls._model = AutoModelForCausalLM.from_pretrained(target)
        cls._model.eval()

        cls._device = "cuda" if torch.cuda.is_available() else "cpu"
        cls._model.to(cls._device)
        cls._loaded_model_id = target

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        tokenizer = self.__class__._tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        lines: List[str] = []
        for msg in messages:
            role = msg.get("role", "user").strip().lower()
            content = msg.get("content", "")
            if role == "system":
                lines.append(f"System: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
            else:
                lines.append(f"User: {content}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 80,
        temperature: float = 0.7,
    ) -> StandardizedResult:
        return self.generate_chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 80,
        temperature: float = 0.7,
    ) -> StandardizedResult:
        prompt = self._build_prompt(messages)
        tokenizer = self.__class__._tokenizer
        model = self.__class__._model
        device = self.__class__._device

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # temperature <= 0 => greedy decoding => deterministic given the input
        # (used by the benchmark's --eval-deterministic path). Sampling params
        # are only passed when actually sampling, to avoid transformers warnings.
        do_sample = temperature > 0
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            output_attentions=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with torch.inference_mode():
            generation = model.generate(**gen_kwargs)

        generated_ids = generation.sequences[0, input_ids.shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        token_logprobs: List[float] = []
        token_strings: List[str] = []
        top_logprobs: List[List[float]] = []
        hidden_states_per_token: List[np.ndarray] = []
        logits_per_token: List[np.ndarray] = []

        # generation.scores         : tuple len = n_generated_tokens
        # generation.hidden_states  : tuple len = n_generated_tokens
        #     each element is a tuple of (n_layers + 1) tensors (embed + per layer)
        #     each tensor has shape (batch, seq_len_at_step, hidden_dim);
        #     for step 0 seq_len = full prompt len, for later steps seq_len = 1
        #     the just-generated token always lives at position [-1]
        n_steps = min(
            len(generated_ids),
            len(generation.scores),
            len(generation.hidden_states) if generation.hidden_states is not None else 0,
        )

        for idx in range(n_steps):
            token_id = generated_ids[idx]

            # ── log-probs (unchanged behaviour) ──────────────────────
            logits = generation.scores[idx][0]                 # (vocab,)
            log_probs = torch.log_softmax(logits, dim=-1)

            selected = float(log_probs[token_id].item())
            token_logprobs.append(selected)
            token_strings.append(tokenizer.decode(token_id.unsqueeze(0), skip_special_tokens=False))

            top_vals, _ = torch.topk(log_probs, k=5)
            top_logprobs.append([float(v.item()) for v in top_vals])

            # ── full pre-softmax logits (for Semantic Energy) ────────
            # Cast to float32 *on the device* before numpy(): numpy has no
            # native bfloat16, so a bf16 tensor (TinyLlama default) crashes
            # with "Got unsupported ScalarType BFloat16" otherwise.
            logits_per_token.append(
                logits.detach().to(torch.float32).cpu().numpy()  # (vocab_size,)
            )

            # ── last-layer hidden state of the just-generated token ──
            step_hs = generation.hidden_states[idx]
            last_layer = step_hs[-1]                             # (1, seq_len_at_step, hidden_dim)
            last_token_vec = last_layer[0, -1, :]                # (hidden_dim,)
            hidden_states_per_token.append(
                last_token_vec.detach().to(torch.float32).cpu().numpy()
            )

        if not token_logprobs:
            raise RuntimeError("Local model did not return any token logprobs.")

        return StandardizedResult(
            text=text,
            token_logprobs=token_logprobs,
            token_strings=token_strings,
            top_logprobs=top_logprobs,
            model_name=self.model,
            hidden_states_per_token=hidden_states_per_token,
            logits_per_token=logits_per_token,
        )
