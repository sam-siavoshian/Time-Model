"""Abstract base for tau_sessions adapters.

A TauAdapter owns one HuggingFace causal LM (frozen or LoRA-wrapped) and
knows how to inject tau (real elapsed seconds) into a generation call.
Subclasses customize three things:

  1. construct_prompt(prompt, tau_seconds) -> str
       How to turn the dataset prompt into the model's actual input. The
       prompt adapter prepends "[elapsed: ...]"; the CI and vanilla
       adapters pass the prompt through unchanged.

  2. forward_kwargs(tau_seconds) -> dict
       Extra kwargs to thread into the model call. The CI adapter passes
       tau_t=<seconds> here; everyone else returns {}.

  3. generate(prompt, tau_seconds) -> str
       The full text generation. Default impl uses greedy decoding with
       the model's tokenizer + apply_chat_template. Subclasses that need
       a non-standard call path (e.g. CIAdapter routing tau through
       _chi_stack via model.forward) override generate() directly.

License: MIT.
"""

from __future__ import annotations

import abc
from typing import Any

import torch


class TauAdapter(abc.ABC):
    """One frozen LLM + a strategy for injecting tau.

    Concrete subclasses MUST:
      - implement `load()` to populate `self.model` and `self.tokenizer`
      - implement `generate(prompt, tau_seconds)` to return decoded text

    Optional overrides:
      - construct_prompt(...)  for prompt-level tau injection
      - cleanup()              for any per-adapter teardown
    """

    name: str = "base"

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 48,
        **kwargs: Any,
    ) -> None:
        self.base_model = base_model
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype, self.device)
        self.max_new_tokens = max_new_tokens
        self.extra_kwargs = kwargs
        self.model = None
        self.tokenizer = None
        self._loaded = False

    # --- lifecycle -----------------------------------------------------

    @abc.abstractmethod
    def load(self) -> None:
        """Load weights, tokenizer, and any LoRA / chrono modules."""

    def cleanup(self) -> None:
        """Optional teardown hook. Default: drop model refs so the GC
        can reclaim GPU memory between adapters."""
        self.model = None
        self.tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:                                # noqa: BLE001
                pass

    # --- prompt strategy ----------------------------------------------

    def construct_prompt(self, prompt: str, tau_seconds: float) -> str:
        """How to render the dataset prompt for the model. Default: wrap
        in the Qwen chat template with no tau text added (the chrono
        channel will carry tau for CI; vanilla just sees the prompt)."""
        return self._chat_wrap(prompt)

    def _chat_wrap(self, user_msg: str) -> str:
        """Apply the tokenizer's chat template. Falls back to the raw
        Qwen IM-tags format if the tokenizer lacks a template (e.g. a
        slim base distribution)."""
        if self.tokenizer is None:
            raise RuntimeError("adapter not loaded")
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            "<|im_start|>user\n"
            f"{user_msg}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    # --- generation ----------------------------------------------------

    @abc.abstractmethod
    def generate(self, prompt: str, tau_seconds: float) -> str:
        """Return the model's generated text for (prompt, tau).

        Implementations should:
          - call construct_prompt() to render the input
          - decode greedily (temperature 0) for determinism
          - stop on <|im_end|> or EOS
          - return the decoded NEW tokens only (no echoed input)
        """


# -- helpers ----------------------------------------------------------------

def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        # bf16 on CUDA and MPS (Apple silicon supports bf16 since macOS 14);
        # fp32 on CPU to avoid silent slowdown.
        if device in ("cuda", "mps"):
            return torch.bfloat16
        return torch.float32
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[dtype]


@torch.no_grad()
def greedy_generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    tau_t: float | None = None,
) -> str:
    """Generic greedy decode that works for both vanilla HF causal LMs
    AND the QwenTime wrapper (which accepts tau_t as a forward kwarg).

    We hand-roll the loop instead of model.generate() because:
      1. QwenTime.forward returns {"logits": ...}, not GenerateOutput.
      2. We need bit-exact determinism (no sampling) and a tight stop
         on <|im_end|>.
      3. The tau_t plumbing for CI is a positional/kwarg the HF generate
         loop does not know about.
    """
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if ids.dim() == 2:
        ids = ids.squeeze(0)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>") if hasattr(tokenizer, "convert_tokens_to_ids") else None
    eos = tokenizer.eos_token_id
    generated: list[int] = []
    cur = ids
    for _ in range(max_new_tokens):
        if tau_t is None:
            out = model(cur.unsqueeze(0) if cur.dim() == 1 else cur)
            logits = out.logits if hasattr(out, "logits") else out["logits"]
            if logits.dim() == 3:
                logits = logits[0]                          # (T, V)
        else:
            out = model(cur, tau_t=tau_t)
            logits = out["logits"] if isinstance(out, dict) else out.logits
            if logits.dim() == 3:
                logits = logits[0]
        next_id = int(logits[-1].float().argmax().item())
        if im_end is not None and next_id == im_end:
            break
        if eos is not None and next_id == eos:
            break
        generated.append(next_id)
        cur = torch.cat([cur, torch.tensor([next_id], device=device)])
    return tokenizer.decode(generated, skip_special_tokens=True)
