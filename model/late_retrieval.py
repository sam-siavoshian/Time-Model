"""Late-retrieval head (A1 baseline).

Architecture difference from IPCN:
- NO pre-forward prefix (zero injection)
- Standard transformer forward
- THEN: query memory bank with hidden states
- Retrieved values mixed into final hidden state via gated residual
- LM head on mixed result

This is the strict "retrieval-augmented" baseline. Memory is consulted
AFTER computation has started, not BEFORE. Per SPEC.tex H3 ordering claim,
A1 should LOSE to A2+ on memory-biased ambiguity tasks because
late retrieval cannot bend layer-0 interpretation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import IPCNConfig
from model.memory import MemoryBank


class LateRetrievalHead(nn.Module):
    """Queries memory AFTER the core stack, then mixes back via gated residual."""

    def __init__(self, cfg: IPCNConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        d_memory = cfg.d_memory

        # Project hidden state -> memory query
        self.q_proj = nn.Linear(d_model, d_memory, bias=False)
        # Project retrieved value back up to model dim
        self.v_proj = nn.Linear(d_memory, d_model, bias=False)
        # Gate
        self.gate = nn.Linear(d_model + d_model, d_model, bias=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden: torch.Tensor,                # (S, d_model) post-core, pre-LM-head
        memory: MemoryBank,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mixed_hidden, retrieved_features)."""
        cfg = self.cfg
        d_memory = cfg.d_memory

        # Per-position query
        Q = self.q_proj(hidden)                              # (S, d_memory)
        # Attention scores over memory slots
        # K = memory.k  (N_m, d_memory)
        scores = Q @ memory.k.t() / math.sqrt(d_memory)      # (S, N_m)
        # Bias by confidence and inverse-conflict (looser than prefix-attention biases)
        scores = scores + cfg.beta_c * memory.conf
        scores = scores - cfg.beta_delta * memory.conflict
        alpha = F.softmax(scores, dim=-1)                     # (S, N_m)
        # Retrieve weighted value
        retrieved = alpha @ memory.v                          # (S, d_memory)
        retrieved_up = self.v_proj(retrieved)                # (S, d_model)

        # Gated residual mix
        gate_in = torch.cat([hidden, retrieved_up], dim=-1)
        g = torch.sigmoid(self.gate(gate_in))                # (S, d_model)
        mixed = self.ln(hidden + g * retrieved_up)
        return mixed, retrieved_up
