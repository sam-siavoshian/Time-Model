"""Chronometric Injection adapter for the tau_sessions benchmark.

Wraps the frozen base LLM with the v15 LoRA + per-layer FiLM chrono
module from model/qwen_time.py and feeds tau (real elapsed seconds) as
a float tensor through the chrono encoder. This is the system under
test for the CI paper -- everything else in this folder is a baseline.

Defaults are pinned to the v15 release config:
  base       Qwen/Qwen2.5-3B-Instruct
  timescales 2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800
  inject     every decoder layer except the last (handled inside QwenTime)
  lora       rank 8 on q/k/v/o + lm_head, every layer
  injection  film (DiT AdaLN-Zero)

The user MUST supply --checkpoint pointing at a v15 release file:
  qwen_time_v15s_20260523_141410_seed{0,1,2}.pt

Checkpoints are released on GitHub at the tag v15.0; SHA256s in README.

License: MIT.
"""

from __future__ import annotations

from typing import Optional

import torch

from model.qwen_time import QwenTime, QwenTimeConfig, build_qwen_time
from model.qwen_time_check import load_trainable

from .base import TauAdapter, greedy_generate


# v15 cross-seed training timescales (15 scales, see scripts/run_v15_seeds.sh).
V15_TIMESCALES: tuple[int, ...] = (
    2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 86400, 604800,
)


class CIAdapter(TauAdapter):
    name = "ci"

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 48,
        *,
        checkpoint: Optional[str] = None,
        timescales: tuple[int, ...] = V15_TIMESCALES,
        injection_type: str = "film",
        inject_layers: tuple[int, ...] = (),
        lora_rank: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(
            base_model=base_model,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        if checkpoint is None:
            raise ValueError(
                "CIAdapter requires --checkpoint pointing at a v15 release file. "
                "Download from https://github.com/sam-siavoshian/Time-Model/releases/tag/v15.0"
            )
        self.checkpoint = checkpoint
        self.timescales = tuple(int(x) for x in timescales)
        self.injection_type = injection_type
        self.inject_layers = tuple(int(x) for x in inject_layers)
        self.lora_rank = int(lora_rank)
        self._qwen_time: QwenTime | None = None

    def load(self) -> None:
        if self._loaded:
            return
        cfg = QwenTimeConfig(
            base_model_name=self.base_model,
            timescales=self.timescales,
            inject_layers=self.inject_layers,
            lora_rank=self.lora_rank,
            injection_type=self.injection_type,
        )
        qt = build_qwen_time(cfg)
        qt.to(self.device)
        load_trainable(qt, self.checkpoint)
        qt.train(False)
        self._qwen_time = qt
        # The base + tokenizer live INSIDE QwenTime. Surface them at the
        # adapter level so construct_prompt's chat-template fallback works.
        self.model = qt
        self.tokenizer = qt.tokenizer
        # Belt-and-suspenders: freeze every parameter. We only forward.
        for p in qt.parameters():
            p.requires_grad_(False)
        self._loaded = True

    @torch.no_grad()
    def generate(self, prompt: str, tau_seconds: float) -> str:
        if not self._loaded or self._qwen_time is None:
            raise RuntimeError("call load() before generate()")
        wrapped = self.construct_prompt(prompt, tau_seconds)
        return greedy_generate(
            self._qwen_time, self.tokenizer, wrapped,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            tau_t=float(tau_seconds),
        )

    def cleanup(self) -> None:
        self._qwen_time = None
        super().cleanup()
