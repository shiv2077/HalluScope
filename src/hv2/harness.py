"""The observability harness: one model load, per-prompt GenRecord capture.

We run a *manual* decode loop (not ``model.generate``) so we can read the RAW
pre-warp logits at every step and capture hidden states at exactly the positions
we want. This is the scientific instrument; control matters more than brevity.

Per decoding step the model does one forward pass whose final-position hidden
state is the representation FROM WHICH that step's token is emitted. Call it
``H[i]`` (per-layer, stacked -> [L, H]) for emitted token ``i`` in ``0..T-1``.
``H[0]`` is the last-prompt-token state (the prompt forward ends there).

Captured key positions (CLAUDE.md):
  * last_prompt  = H[0]
  * first_answer = H[1]      (state after consuming answer token 0; ==H[0] if T==1)
  * last_answer  = H[T-1]
  * exact_answer = H[i] for each i in the exact-answer token span (best-effort
                   token-subsequence match of a gold alias; empty if not found)

Levels b/c per step come from the RAW logits (float32):
  * top-100 logprobs + their token ids        (level b)
  * entropy = -sum p log p   over full vocab
  * energy  = -logsumexp(logits)
  * margin  = top1_logprob - top2_logprob     (level c)

NEVER stores full-vocab logits (CLAUDE.md). Hidden states are float16.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import HV2Config
from .grading import normalize
from .hdf5_store import GenRecord


def load_model(cfg: HV2Config):
    """Load model + tokenizer ONCE. eval mode, resolved device/dtype.

    4-bit NF4 is taken only on CUDA (cfg.resolve_quantization raises otherwise).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quant = cfg.resolve_quantization()
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    kwargs: dict = {}
    if quant == "nf4":
        # imported lazily; only reachable on CUDA
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = cfg.torch_dtype

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    if quant != "nf4":
        model = model.to(cfg.device)
    model.eval()
    return model, tok


def chat_prompt(tok, user: str, system: str | None = None) -> str:
    """Render a Qwen chat prompt ending at the assistant turn."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _find_answer_token_span(gen_ids: list[int], tok, aliases: list[str] | None) -> list[int]:
    """Best-effort: indices (into gen_ids) of the first gold-alias token subsequence."""
    if not aliases:
        return []
    for alias in aliases:
        for variant in (alias, " " + alias):
            needle = tok.encode(variant, add_special_tokens=False)
            n, m = len(gen_ids), len(needle)
            if m == 0:
                continue
            for i in range(n - m + 1):
                if gen_ids[i : i + m] == needle:
                    return list(range(i, i + m))
    return []


@torch.inference_mode()
def generate(
    model,
    tok,
    prompt_text: str,
    cfg: HV2Config,
    *,
    key: str,
    qid: str,
    kind: str,
    gold_aliases: list[str] | None = None,
    greedy: bool = True,
    temperature: float | None = None,
    seed: int | None = None,
) -> GenRecord:
    """Decode ``prompt_text`` and return a full GenRecord. Never raises on a
    normal generation; degenerate (immediate-EOS) prompts still yield T>=1."""
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
    temp = temperature if (temperature and not greedy) else 1.0

    enc = tok(prompt_text, return_tensors="pt").to(device)
    cur = enc.input_ids
    past = None

    K = cfg.topk_logprobs
    top_lp_rows: list[np.ndarray] = []
    top_id_rows: list[np.ndarray] = []
    func_rows: list[np.ndarray] = []
    per_step_hidden: list[np.ndarray] = []  # H[i] as [L, H] float16
    gen_ids: list[int] = []

    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(int(tok.eos_token_id))

    for step in range(cfg.max_new_tokens):
        out = model(input_ids=cur, past_key_values=past, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float().squeeze(0)  # [vocab], raw, float32

        # level c functionals (full vocab, online — logits discarded after)
        logprobs = torch.log_softmax(logits, dim=-1)
        probs = logprobs.exp()
        entropy = float(-(probs * logprobs).sum())
        energy = float(-torch.logsumexp(logits, dim=-1))
        top_lp, top_idx = torch.topk(logprobs, K)
        margin = float(top_lp[0] - top_lp[1])
        func_rows.append(np.array([entropy, energy, margin], dtype=np.float32))
        top_lp_rows.append(top_lp.cpu().numpy().astype(np.float32))
        top_id_rows.append(top_idx.cpu().numpy().astype(np.int64))

        # level d: producing-state H[step], all layers, last position, float16
        layers = out.hidden_states  # tuple len L, each [1, seq, Hdim]
        H = np.stack([layers[l][0, -1, :].float().cpu().numpy() for l in range(len(layers))]).astype(np.float16)
        per_step_hidden.append(H)

        # choose next token
        if greedy:
            nxt = int(torch.argmax(logits))
        else:
            p = torch.softmax(logits / temp, dim=-1)
            nxt = int(torch.multinomial(p, 1))
        gen_ids.append(nxt)
        if nxt in eos_ids:
            break
        cur = torch.tensor([[nxt]], device=device)

    T = len(gen_ids)
    # strip a single trailing EOS for the decoded text
    text_ids = gen_ids[:-1] if (gen_ids and gen_ids[-1] in eos_ids) else gen_ids
    text = tok.decode(text_ids, skip_special_tokens=True).strip()

    # exact-answer token span (best effort) within the *non-eos* answer tokens
    answer_idx = _find_answer_token_span(text_ids, tok, gold_aliases)

    # assemble captured hidden positions
    pos_kinds: list[str] = []
    pos_index: list[int] = []
    hidden_sel: list[np.ndarray] = []

    def _add(i: int, kind_label: str):
        if 0 <= i < T:
            hidden_sel.append(per_step_hidden[i])
            pos_index.append(i)
            pos_kinds.append(kind_label)

    _add(0, "last_prompt")
    _add(1 if T >= 2 else 0, "first_answer")
    _add(T - 1, "last_answer")
    for i in answer_idx:
        _add(i, "exact_answer")

    hidden_states = np.stack(hidden_sel).astype(np.float16)  # [P, L, H]

    rec = GenRecord(
        key=key,
        prompt=prompt_text,
        text=text,
        token_ids=np.array(gen_ids, dtype=np.int64),
        top_logprobs=np.stack(top_lp_rows).astype(np.float32),
        top_token_ids=np.stack(top_id_rows).astype(np.int64),
        functionals=np.stack(func_rows).astype(np.float32),
        hidden_states=hidden_states,
        hidden_pos_index=np.array(pos_index, dtype=np.int64),
        hidden_pos_kind=pos_kinds,
        answer_token_indices=np.array(answer_idx, dtype=np.int64),
        meta={
            "model": cfg.model_name,
            "qid": qid,
            "kind": kind,
            "greedy": greedy,
            "temperature": None if greedy else temp,
            "n_layers": int(hidden_states.shape[1]),
            "hidden_dim": int(hidden_states.shape[2]),
            "normalized_text": normalize(text),
        },
    )
    rec.validate()
    return rec
