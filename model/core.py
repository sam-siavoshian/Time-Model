"""Decoder-only Transformer core with LoRA adapters + Route-3 LayerNorm modulation.

8 layers, width 512, 8 heads, FFN 2048 per ARCHITECTURE_LOCKED.md.
LoRA adapters in layers 0-2. Route-3 LN modulation applied in layers 1-2.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.adapters import LoRALinear
from model.config import IPCNConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: IPCNConfig, layer_idx: int, prefix_len: int):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = d // cfg.n_heads
        rank = cfg.lora_rank if layer_idx in cfg.consolidated_layers else 0

        self.q_proj = LoRALinear(d, d, rank=rank, alpha=cfg.lora_alpha)
        self.k_proj = LoRALinear(d, d, rank=rank, alpha=cfg.lora_alpha)
        self.v_proj = LoRALinear(d, d, rank=rank, alpha=cfg.lora_alpha)
        self.o_proj = LoRALinear(d, d, rank=rank, alpha=cfg.lora_alpha)

        self.prefix_len = prefix_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (S, d_model). First prefix_len positions are prefix; rest are content.
        S, d = x.shape
        Q = self.q_proj(x).view(S, self.n_heads, self.d_head).transpose(0, 1)  # (H, S, d_head)
        K = self.k_proj(x).view(S, self.n_heads, self.d_head).transpose(0, 1)
        V = self.v_proj(x).view(S, self.n_heads, self.d_head).transpose(0, 1)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)              # (H, S, S)

        # Mask: prefix tokens are bidirectional among themselves; content tokens
        # are causal w.r.t. content and have full access to all prefix tokens.
        device = x.device
        mask = torch.zeros(S, S, device=device, dtype=torch.bool)
        K_p = self.prefix_len
        # Prefix-to-prefix: full (bidirectional). Already True via remaining defaults below.
        # Content-to-content: causal lower-triangular.
        for i in range(K_p, S):
            for j in range(K_p, i + 1):
                mask[i, j] = True
            # Content-to-prefix: full access
            for j in range(K_p):
                mask[i, j] = True
        for i in range(K_p):
            for j in range(K_p):
                mask[i, j] = True
        # Apply: -inf where NOT allowed
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ V                                                          # (H, S, d_head)
        out = out.transpose(0, 1).contiguous().view(S, d)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: IPCNConfig, layer_idx: int, prefix_len: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        d = cfg.d_model
        rank = cfg.lora_rank if layer_idx in cfg.consolidated_layers else 0

        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(cfg, layer_idx, prefix_len)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = LoRALinear(d, cfg.d_ffn, rank=rank, alpha=cfg.lora_alpha)
        self.fc2 = LoRALinear(cfg.d_ffn, d, rank=rank, alpha=cfg.lora_alpha)

        # Route-3: LayerNorm modulation from prefix broadcast b
        self.use_lnmod = layer_idx in cfg.lnmod_layers and cfg.use_route3_lnmod
        if self.use_lnmod:
            self.W_gamma_lnmod = nn.Linear(d, d, bias=False)
            self.W_beta_lnmod = nn.Linear(d, d, bias=False)
            self.alpha_film = cfg.alpha_film
        else:
            self.W_gamma_lnmod = None
            self.W_beta_lnmod = None

    def forward(self, x: torch.Tensor, b_broadcast: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (S, d_model). b_broadcast: (S, d_model) for content positions (with prefix positions zeroed/repeated).
        h = self.ln1(x)
        if self.use_lnmod and b_broadcast is not None:
            gamma = 1.0 + self.alpha_film * torch.tanh(self.W_gamma_lnmod(b_broadcast))
            beta = self.alpha_film * torch.tanh(self.W_beta_lnmod(b_broadcast))
            h = gamma * h + beta
        x = x + self.attn(h)

        h2 = self.ln2(x)
        ff = self.fc2(F.gelu(self.fc1(h2)))
        x = x + ff
        return x


class CoreTransformer(nn.Module):
    """Decoder-only stack that accepts a prefix-prepended sequence."""

    def __init__(self, cfg: IPCNConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Learned positional embeddings for chunk_length + prefix_length total positions
        self.pos_emb = nn.Embedding(cfg.chunk_length + cfg.prefix_length, cfg.d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, i, cfg.prefix_length) for i in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (L,) -> embeddings: (L, d_model). Positions added later inside forward()."""
        return self.tok_emb(input_ids)

    def forward(
        self,
        H0: torch.Tensor,                # (S, d_model) — already injected (prefix + tilde_e)
        b_full: Optional[torch.Tensor] = None,    # (S, d_model) — for Route-3 LN mod (prefix rows zero or replicated)
        return_hidden: bool = False,
    ):
        """Run the core stack. Returns (logits: (S, vocab), hidden_layers: list)."""
        S, d = H0.shape
        positions = torch.arange(S, device=H0.device)
        x = H0 + self.pos_emb(positions)
        layer_outputs = []
        for block in self.blocks:
            x = block(x, b_broadcast=b_full)
            if return_hidden:
                layer_outputs.append(x)
        x = self.ln_f(x)
        # Compute logits only on content positions (skip prefix)
        K_p = self.cfg.prefix_length
        content = x[K_p:]
        logits = self.lm_head(content)
        return logits, layer_outputs if return_hidden else None, x
