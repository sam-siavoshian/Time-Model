"""Prefix-Forming Controller (PFC).

Small Transformer that builds the involuntary prefix BEFORE the main core
runs. Receives a cheap input sketch + temporal state z + chronometric chi_t
+ persistent memory bank. Produces K_p prefix vectors that will be injected
into the main hidden state.

LoRA adapters live inside this controller (the consolidation target).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.adapters import LoRALinear
from model.config import IPCNConfig
from model.memory import MemoryBank


def shallow_sketch(
    e: torch.Tensor,                # (L, d_model) — token embeddings of current chunk
    z: torch.Tensor,                # (d_temporal_state,)
    W_s: nn.Linear,                 # projection of embeddings (d_model -> d_sketch_inner)
) -> torch.Tensor:
    """Build cheap input sketch s_t = [mean, max, e_first, e_last, z]."""
    proj = W_s(e)                   # (L, d_inner)
    mean = proj.mean(dim=0)         # (d_inner,)
    mx = proj.max(dim=0).values     # (d_inner,)
    first = e[0]                    # (d_model,)
    last = e[-1]                    # (d_model,)
    return torch.cat([mean, mx, first, last, z], dim=-1)


class PFCBlock(nn.Module):
    """One self-attention + FFN layer over prefix tokens."""

    def __init__(self, cfg: IPCNConfig):
        super().__init__()
        d = cfg.pfc_d_model
        self.n_heads = cfg.pfc_n_heads
        self.d_head = d // cfg.pfc_n_heads
        rank = cfg.lora_rank if cfg.consolidate_pfc else 0

        self.ln1 = nn.LayerNorm(d)
        self.q_proj = LoRALinear(d, d, rank=rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
        self.k_proj = LoRALinear(d, d, rank=rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
        self.v_proj = LoRALinear(d, d, rank=rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
        self.o_proj = LoRALinear(d, d, rank=rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = LoRALinear(d, cfg.pfc_d_ffn, rank=rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
        self.fc2 = LoRALinear(cfg.pfc_d_ffn, d, rank=rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (K_p, d_model). Bidirectional attention within prefix tokens.
        h = self.ln1(x)
        K_p, d = h.shape
        Q = self.q_proj(h).view(K_p, self.n_heads, self.d_head).transpose(0, 1)  # (H, K_p, d_head)
        K = self.k_proj(h).view(K_p, self.n_heads, self.d_head).transpose(0, 1)
        V = self.v_proj(h).view(K_p, self.n_heads, self.d_head).transpose(0, 1)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)                # (H, K_p, K_p)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V                                                            # (H, K_p, d_head)
        out = out.transpose(0, 1).contiguous().view(K_p, d)
        x = x + self.o_proj(out)

        h2 = self.ln2(x)
        ff = self.fc2(F.gelu(self.fc1(h2)))
        x = x + ff
        return x


class PrefixFormingController(nn.Module):
    """The whole PFC: input sketch -> prefix queries -> memory read -> refine."""

    def __init__(self, cfg: IPCNConfig):
        super().__init__()
        self.cfg = cfg
        d_inner = cfg.pfc_d_model // 2

        # Sketch projection
        self.W_s = nn.Linear(cfg.d_model, d_inner, bias=False)
        sketch_dim = 2 * d_inner + 2 * cfg.d_model + cfg.d_temporal_state

        # Project sketch+z+chi -> K_p query vectors of dim d_memory
        self.W_Q_P = nn.Linear(
            sketch_dim + cfg.d_temporal_state + cfg.d_chronometric,
            cfg.prefix_length * cfg.d_memory,
            bias=False,
        )

        # Project read values (d_memory) up to d_model for refinement
        self.value_to_model = nn.Linear(cfg.d_memory, cfg.pfc_d_model, bias=False)

        # Stack of PFC refinement blocks
        self.blocks = nn.ModuleList([PFCBlock(cfg) for _ in range(cfg.pfc_n_layers)])
        self.ln_out = nn.LayerNorm(cfg.pfc_d_model)

    def forward(
        self,
        e: torch.Tensor,                # (L, d_model) — token embeddings (before injection)
        z: torch.Tensor,                # (d_temporal_state,)
        chi_t: torch.Tensor,            # (d_chronometric,)
        memory: MemoryBank,
        tau_t: float,
    ):
        """Returns (prefix P_t: (K_p, d_model), alpha_p: (K_p, N_m))."""
        cfg = self.cfg

        s_t = shallow_sketch(e, z, self.W_s)
        q_input = torch.cat([s_t, z, chi_t], dim=-1)
        Q = self.W_Q_P(q_input).view(cfg.prefix_length, cfg.d_memory)             # (K_p, d_m)

        alpha = memory.prefix_attention(Q, chi_t, tau_t)                           # (K_p, N_m)
        P_hat = memory.read_values(alpha)                                          # (K_p, d_m)

        # Project up to d_model and refine
        x = self.value_to_model(P_hat)                                             # (K_p, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        return x, alpha
