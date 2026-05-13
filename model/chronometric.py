"""Deterministic chronometric encoder chi_t.

Maps real elapsed time tau (and delta_tau since last update) to a multi-scale
sinusoidal vector. Zero learned parameters. Same trick as positional encoding,
applied to real wall-clock instead of token position.

chi_t = [tau, delta_tau, psi(tau), psi(delta_tau), nu, gap]

where psi(tau) = [log(1+tau), sin(2*pi*tau/T_b), cos(2*pi*tau/T_b)] over 13 timescales.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from model.config import IPCNConfig


class ChronometricEncoder(nn.Module):
    """Deterministic time -> vector. No learned params."""

    def __init__(self, cfg: IPCNConfig):
        super().__init__()
        self.cfg = cfg
        # Buffer of timescales for vectorized computation
        self.register_buffer(
            "timescales",
            torch.tensor(cfg.timescales, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _psi(tau: torch.Tensor, timescales: torch.Tensor) -> torch.Tensor:
        """tau: (B,) -> psi(tau): (B, 3 * len(timescales))."""
        # tau: (B,), timescales: (T,)
        tau_unsq = tau.unsqueeze(-1)                         # (B, 1)
        scales = timescales.unsqueeze(0)                     # (1, T)
        angle = 2.0 * math.pi * tau_unsq / scales            # (B, T)
        log_term = torch.log1p(tau_unsq).expand_as(angle)    # (B, T)
        sin_term = torch.sin(angle)                          # (B, T)
        cos_term = torch.cos(angle)                          # (B, T)
        return torch.cat([log_term, sin_term, cos_term], dim=-1)  # (B, 3T)

    def forward(
        self,
        tau: torch.Tensor,
        delta_tau: torch.Tensor,
        event_density: Optional[torch.Tensor] = None,
        gap_flag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """tau, delta_tau: (B,) real minutes. event_density: (B,) optional.
        gap_flag: (B,) 0/1.
        Returns chi_t: (B, d_chronometric)."""
        B = tau.shape[0]
        ts = self.timescales.to(tau.dtype)
        psi_tau = self._psi(tau, ts)                         # (B, 3T)
        psi_dt = self._psi(delta_tau, ts)                    # (B, 3T)

        if event_density is None:
            event_density = torch.zeros(B, device=tau.device, dtype=tau.dtype)
        if gap_flag is None:
            gap_flag = torch.zeros(B, device=tau.device, dtype=tau.dtype)

        chi = torch.cat(
            [
                tau.unsqueeze(-1),                           # (B, 1)
                delta_tau.unsqueeze(-1),                     # (B, 1)
                psi_tau,                                     # (B, 3T)
                psi_dt,                                      # (B, 3T)
                event_density.unsqueeze(-1),                 # (B, 1)
                gap_flag.unsqueeze(-1),                      # (B, 1)
            ],
            dim=-1,
        )
        return chi                                           # (B, d_chronometric)
