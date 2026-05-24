"""Prompt-conditioning baseline: tau is injected as text in the prompt.

The obvious approach a user without a tau channel would try: tell the
model the elapsed time in natural language and let attention figure it
out. Strong baseline for duration_recall (answer is literally in the
input) and a fair baseline for staleness and adaptive because the model
still has to do arithmetic and length control.

Injected prefix uses an unambiguous format: "[elapsed: 3h 42m] ".
The format matches common LLM telemetry conventions so it stays in-
distribution for instruction-tuned models.

License: MIT.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import TauAdapter, greedy_generate


def _format_elapsed(tau_s: float) -> str:
    """Compact, unambiguous breakdown.

    < 60s         -> "12s"
    < 60m         -> "5m 12s"
    < 24h         -> "3h 42m"
    >= 24h        -> "5d 7h"
    """
    if tau_s < 60:
        return f"{int(round(tau_s))}s"
    if tau_s < 3600:
        m, s = divmod(int(round(tau_s)), 60)
        return f"{m}m {s}s"
    if tau_s < 86400:
        h, rem = divmod(int(round(tau_s)), 3600)
        m = rem // 60
        return f"{h}h {m}m"
    d, rem = divmod(int(round(tau_s)), 86400)
    h = rem // 3600
    return f"{d}d {h}h"


class PromptAdapter(TauAdapter):
    name = "prompt"

    def construct_prompt(self, prompt: str, tau_seconds: float) -> str:
        elapsed = _format_elapsed(tau_seconds)
        # Inject as a bracketed preamble inside the user turn so the
        # message reaches the model regardless of the chat-template
        # implementation.
        prefixed = f"[elapsed: {elapsed}] {prompt}"
        return self._chat_wrap(prefixed)

    def load(self) -> None:
        if self._loaded:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True
        )
        # transformers >=5 renamed torch_dtype -> dtype; older versions
        # use torch_dtype. We try the new kwarg first and fall back.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model, dtype=self.dtype, trust_remote_code=True,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model, torch_dtype=self.dtype, trust_remote_code=True,
            )
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._loaded = True

    @torch.no_grad()
    def generate(self, prompt: str, tau_seconds: float) -> str:
        if not self._loaded:
            raise RuntimeError("call load() before generate()")
        wrapped = self.construct_prompt(prompt, tau_seconds)
        return greedy_generate(
            self.model, self.tokenizer, wrapped,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            tau_t=None,
        )
