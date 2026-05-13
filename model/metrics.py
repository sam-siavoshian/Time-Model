"""Standalone metric functions per SPEC.tex §17.

These are NOT loss terms (those live in losses.py). These are diagnostic
metrics to be computed at eval time and logged for paper figures.

  - I_pre        Pre-computation influence index
  - H_P          Prefix entropy
  - H_u          Slot utilization entropy
  - V_t          Memory velocity
  - D_Omega      Adapter drift from init
"""

from __future__ import annotations

import math

import torch

from model.adapters import LoRALinear


@torch.no_grad()
def pre_computation_influence(
    H0_prefix: torch.Tensor,                         # (S, d_model) with prefix injection
    H0_zero: torch.Tensor,                           # (S, d_model) with zero prefix
    H_L_prefix: torch.Tensor,                        # (S, d_model) final hidden with prefix
    H_L_zero: torch.Tensor,                          # (S, d_model) final hidden with zero prefix
    eps: float = 1e-8,
) -> float:
    """SPEC.tex §17.2 I_pre = ||H0^prefix - H0^zero||_F / (||HL^prefix - HL^zero||_F + eps).

    Pass criterion: I_pre > 0.25 on tasks where memory is useful.
    Interpretation: how much of the prefix's effect lands at layer 0
    vs at the final layer. Late retrieval would give I_pre ≈ 0.
    """
    num = (H0_prefix - H0_zero).norm()
    denom = (H_L_prefix - H_L_zero).norm() + eps
    return float((num / denom).item())


@torch.no_grad()
def prefix_entropy(alpha_p: torch.Tensor) -> float:
    """SPEC.tex §17.2 H_P = -(1/K_p) sum_r sum_i alpha[r,i] log alpha[r,i].

    Mean across prefix queries. Low entropy = focused prefix.
    High entropy = scattered, weak prefix.
    """
    K_p = alpha_p.shape[0]
    H = -(alpha_p.clamp_min(1e-12) * alpha_p.clamp_min(1e-12).log()).sum(dim=-1)
    return float((H.sum() / K_p).item())


@torch.no_grad()
def slot_utilization_entropy(usage: torch.Tensor) -> tuple[float, float]:
    """SPEC.tex §17.2 H_u = -sum_i p_i log p_i where p_i = u_i / sum(u_j).

    Returns (H_u, max_entropy_if_uniform). Pass criterion (anti-collapse):
    H_u >= 0.5 * log(N_m).
    """
    p = usage / (usage.sum() + 1e-9)
    H_u = float(-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum().item())
    H_max = float(math.log(max(1, usage.numel())))
    return H_u, H_max


@torch.no_grad()
def memory_velocity(M_t: torch.Tensor, M_prev: torch.Tensor) -> float:
    """SPEC.tex §17.2 V_t = ||M_t - M_{t-1}||_F.

    Frobenius norm of slot-value delta. Used to detect collapse
    (V_t -> 0) or runaway (V_t exploding).
    """
    return float((M_t - M_prev).norm().item())


@torch.no_grad()
def adapter_drift(model: torch.nn.Module, init_snapshot: dict | None = None) -> float:
    """SPEC.tex §17.2 D_Omega = ||Omega_t - Omega_0||_F.

    If init_snapshot is None, this returns the norm of all LoRA matrices
    in the current model. A non-zero baseline means adapters have moved
    from their init (lora_B starts at 0, so D_Omega = ||lora_A||_F initially
    if lora_B != 0 mass has accumulated).
    """
    if init_snapshot is None:
        total = 0.0
        for module in model.modules():
            if isinstance(module, LoRALinear) and module.rank > 0:
                total += float((module.lora_A ** 2).sum().item())
                total += float((module.lora_B ** 2).sum().item())
        return float(total ** 0.5)
    # With snapshot: diff from init
    total = 0.0
    for module in model.modules():
        if isinstance(module, LoRALinear) and module.rank > 0:
            key = id(module)
            if key in init_snapshot:
                a0, b0 = init_snapshot[key]
                total += float(((module.lora_A - a0) ** 2).sum().item())
                total += float(((module.lora_B - b0) ** 2).sum().item())
    return float(total ** 0.5)


def snapshot_adapter_init(model: torch.nn.Module) -> dict:
    """Take a deep snapshot of all LoRA adapters at init time for D_Omega."""
    snap = {}
    for module in model.modules():
        if isinstance(module, LoRALinear) and module.rank > 0:
            snap[id(module)] = (module.lora_A.detach().clone(), module.lora_B.detach().clone())
    return snap
