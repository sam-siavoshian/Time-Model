"""LoRA adapters.

Low-rank perturbation on top of a frozen Linear weight:
    W_eff = W + (alpha / r) * B @ A
where A: (r, in), B: (out, r). Only A, B are trainable.

Used inside PFC and core layers 0-2 per ARCHITECTURE_LOCKED.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """nn.Linear with optional low-rank additive adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 0,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = (alpha / rank) if rank > 0 else 0.0

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

        if rank > 0:
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # B starts at zero so the adapter contributes nothing at init
            self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

        self._merged = False
        self._adapter_enabled = rank > 0

    def freeze_base(self):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def set_adapter_enabled(self, on: bool):
        self._adapter_enabled = on and (self.rank > 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight.t()
        if self.bias is not None:
            out = out + self.bias
        if self._adapter_enabled and self.lora_A is not None:
            lora = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
            out = out + self.scaling * lora
        return out

    def snapshot_adapter(self):
        """Return a deep copy of adapter weights for rollback."""
        if self.lora_A is None:
            return None
        return (self.lora_A.detach().clone(), self.lora_B.detach().clone())

    def restore_adapter(self, snapshot):
        if snapshot is None or self.lora_A is None:
            return
        a, b = snapshot
        with torch.no_grad():
            self.lora_A.copy_(a)
            self.lora_B.copy_(b)

    @property
    def adapter_parameters(self):
        if self.lora_A is None:
            return []
        return [self.lora_A, self.lora_B]
