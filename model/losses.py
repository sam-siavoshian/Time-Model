"""IPCN training objective (9 terms).

Each term implemented as a stateless function. Composed by ipcn.py.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits: (L, V), targets: (L,) — next-token CE."""
    return F.cross_entropy(logits, targets)


def pre_influence_loss(
    u_t: torch.Tensor,                  # scalar — mean usefulness (loss_no_prefix - loss_with_prefix) over chunk
    gate_bar: torch.Tensor,             # scalar — mean broadcast gate magnitude
    rho_helped: float,
    tau_gate: float,
) -> torch.Tensor:
    """Penalize collapsed prefix gate when memory actually helped."""
    helped = (u_t > rho_helped).float()
    return helped * F.relu(tau_gate - gate_bar)


def precision_loss(u_t: torch.Tensor, gate_bar: torch.Tensor) -> torch.Tensor:
    """Suppress prefix when it hurts (U_t < 0)."""
    hurt = (u_t < 0).float()
    return hurt * gate_bar


def mem_predict_loss(u_hat: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(u_hat, u_true.detach())


def diversity_loss(keys: torch.Tensor) -> torch.Tensor:
    """keys: (N_m, d_m). Penalize off-diagonal cosine^2."""
    N = keys.shape[0]
    k = F.normalize(keys, dim=-1)
    sim = k @ k.t()                                              # (N, N)
    sim2 = sim * sim
    sim2 = sim2 - torch.diag(sim2.diag())
    return sim2.sum() / (N * N)


def slot_util_loss(usage: torch.Tensor, h_min_frac: float = 0.5) -> torch.Tensor:
    """Penalize utilization-entropy collapse below 0.5 * log(N)."""
    N = usage.shape[0]
    p = usage / (usage.sum() + 1e-6)
    p = p + 1e-12
    H_u = -(p * p.log()).sum()
    H_min = h_min_frac * math.log(N)
    return F.relu(torch.tensor(H_min, device=usage.device, dtype=usage.dtype) - H_u)


def evolution_self_predict_loss(
    z_hat: torch.Tensor,
    z_next: torch.Tensor,
    delta_mem_hat: torch.Tensor,
    delta_mem_true: torch.Tensor,
    weight_M: float = 0.5,
) -> torch.Tensor:
    L_z = F.mse_loss(z_hat, z_next.detach())
    L_M = F.mse_loss(delta_mem_hat, delta_mem_true.detach())
    return L_z + weight_M * L_M


def chronometric_loss(
    delta_tau_hat: torch.Tensor,                      # (B,) raw model output (treated as log-space prediction)
    delta_tau_true: torch.Tensor,                     # (B,) real elapsed minutes
    phase_hat: Optional[torch.Tensor] = None,         # (B, C)
    phase_true: Optional[torch.Tensor] = None,        # (B,) class labels
    future_mem_hat: Optional[torch.Tensor] = None,
    future_mem_true: Optional[torch.Tensor] = None,
    lambda_dur: float = 1.0,
    lambda_phase: float = 0.5,
    lambda_future: float = 0.5,
) -> torch.Tensor:
    """Predict log(1 + delta_tau) in log-space for numerical stability.

    Raw delta_tau values can be 1-65,536 minutes; raw-MSE explodes.
    The encoder already represents tau as log(1+tau), so prediction in
    log-space matches the encoding basis.
    """
    target_log = torch.log1p(delta_tau_true.clamp_min(0.0))
    L_dur = F.smooth_l1_loss(delta_tau_hat.squeeze(-1) if delta_tau_hat.dim() > 1 else delta_tau_hat, target_log)
    L_phase = torch.tensor(0.0, device=delta_tau_hat.device, dtype=delta_tau_hat.dtype)
    if phase_hat is not None and phase_true is not None:
        L_phase = F.cross_entropy(phase_hat, phase_true)
    L_future = torch.tensor(0.0, device=delta_tau_hat.device, dtype=delta_tau_hat.dtype)
    if future_mem_hat is not None and future_mem_true is not None:
        L_future = F.smooth_l1_loss(future_mem_hat, future_mem_true.detach())
    return lambda_dur * L_dur + lambda_phase * L_phase + lambda_future * L_future


def consolidation_kl(
    p_teacher_logits: torch.Tensor,                   # (B, V)
    p_student_logits: torch.Tensor,                   # (B, V)
) -> torch.Tensor:
    """KL(p_teacher || p_student). Teacher is stop-grad (computed under no_grad)."""
    p_T = F.softmax(p_teacher_logits.detach(), dim=-1)
    log_p_S = F.log_softmax(p_student_logits, dim=-1)
    log_p_T = F.log_softmax(p_teacher_logits.detach(), dim=-1)
    return (p_T * (log_p_T - log_p_S)).sum(dim=-1).mean()
