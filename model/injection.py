"""Three prefix injection routes.

Route 1 (mandatory): prepend prefix tokens to content sequence.
Route 2 (recommended): broadcast preconditioning of content embeddings via
                      gated residual b_tj.
Route 3 (optional, layers 1-2): LayerNorm modulation via FiLM-style Gamma/B.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import IPCNConfig


class BroadcastPreconditioner(nn.Module):
    """Route 2: token-specific prefix reads then gated residual on embeddings."""

    def __init__(self, cfg: IPCNConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # Token-side projection (to compute attention over prefix tokens)
        self.W_e = nn.Linear(d, d, bias=False)
        # Prefix-side projection
        self.W_p = nn.Linear(d, d, bias=False)
        # Gate input: [e; b; z]
        self.W_gamma = nn.Linear(d + d + cfg.d_temporal_state, d, bias=True)
        # Broadcast residual projection
        self.W_b = nn.Linear(d, d, bias=False)
        self.ln = nn.LayerNorm(d)

    def forward(
        self,
        e: torch.Tensor,                # (L, d_model)
        P: torch.Tensor,                # (K_p, d_model)
        z: torch.Tensor,                # (d_temporal_state,)
        lambda_pre: float,
    ):
        """Returns (e_tilde: (L, d_model), b: (L, d_model)) — b is per-token prefix read."""
        cfg = self.cfg
        L = e.shape[0]
        Q = self.W_e(e)                                                      # (L, d_model)
        K = self.W_p(P)                                                      # (K_p, d_model)
        scores = Q @ K.t() / (cfg.d_model ** 0.5)                            # (L, K_p)
        eta = F.softmax(scores, dim=-1)                                       # (L, K_p)
        b = eta @ P                                                           # (L, d_model)

        z_exp = z.unsqueeze(0).expand(L, -1)
        gate_in = torch.cat([e, b, z_exp], dim=-1)
        gamma = torch.sigmoid(self.W_gamma(gate_in))                          # (L, d_model)
        e_tilde = self.ln(e + lambda_pre * gamma * self.W_b(b))
        return e_tilde, b, gamma


def schedule_lambda_pre(step: int, cfg: IPCNConfig) -> float:
    """Linear anneal from lambda_pre_init at step 0 to lambda_pre_final at lambda_pre_anneal_steps."""
    if step >= cfg.lambda_pre_anneal_steps:
        return cfg.lambda_pre_final
    frac = step / cfg.lambda_pre_anneal_steps
    return cfg.lambda_pre_init + frac * (cfg.lambda_pre_final - cfg.lambda_pre_init)
