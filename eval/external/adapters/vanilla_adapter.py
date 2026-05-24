"""Vanilla baseline: the frozen base LLM with no tau information.

This adapter is the floor of the tau_sessions benchmark: it shows what
the base model knows about elapsed time from chat-template position
alone. On duration_recall and staleness it should perform near chance
(or default to "no time has passed" / coin-flip). On adaptive it should
produce length distributions that do NOT correlate with tau.

License: MIT.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import TauAdapter, greedy_generate


class VanillaAdapter(TauAdapter):
    name = "vanilla"

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
        del tau_seconds                                      # ignored by design
        if not self._loaded:
            raise RuntimeError("call load() before generate()")
        wrapped = self.construct_prompt(prompt, 0.0)
        return greedy_generate(
            self.model, self.tokenizer, wrapped,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            tau_t=None,
        )
