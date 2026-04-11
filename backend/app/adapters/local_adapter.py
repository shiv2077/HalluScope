"""HalluScope - Local HuggingFace adapter.

Provides a local-model implementation of BaseLLMAdapter so the same
risk pipeline can run without external API calls.
"""

from __future__ import annotations

from typing import Dict, List

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

        do_sample = temperature > 0
        generation = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_ids = generation.sequences[0, input_ids.shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        token_logprobs: List[float] = []
        token_strings: List[str] = []
        top_logprobs: List[List[float]] = []

        for idx, token_id in enumerate(generated_ids):
            if idx >= len(generation.scores):
                break
            logits = generation.scores[idx][0]
            log_probs = torch.log_softmax(logits, dim=-1)

            selected = float(log_probs[token_id].item())
            token_logprobs.append(selected)
            token_strings.append(tokenizer.decode(token_id.unsqueeze(0), skip_special_tokens=False))

            top_vals, _ = torch.topk(log_probs, k=5)
            top_logprobs.append([float(v.item()) for v in top_vals])

        if not token_logprobs:
            raise RuntimeError("Local model did not return any token logprobs.")

        return StandardizedResult(
            text=text,
            token_logprobs=token_logprobs,
            token_strings=token_strings,
            top_logprobs=top_logprobs,
            model_name=self.model,
        )
