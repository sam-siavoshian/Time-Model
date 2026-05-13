"""Episodic memory bank.

256 slots, each with key/value/dynamics state + scalars (age/usage/confidence/
plasticity/conflict) + temporal metadata (tau_write, tau_use, chi_slot).

Operates on a single conversation/stream (batch dimension 1 in v1; extend later).

Implements:
  - prefix attention read with all bias terms (confidence, usage, age, disuse,
    conflict, temporal compatibility)
  - write candidates (surprise/novelty/relevance/prefix-attribution)
  - slot assignment via softmax over slots
  - autonomous evolution with sparse neighbor graph + duration-sensitive update
  - silent-gap iteration (evolve without input)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import IPCNConfig


class MemoryBank(nn.Module):
    """Persistent episodic memory across chunks. Stateful per-stream."""

    def __init__(self, cfg: IPCNConfig):
        super().__init__()
        self.cfg = cfg
        N_m = cfg.n_slots
        d_m = cfg.d_memory

        # Slot state: registered as buffers (not learned by gradient, but updated
        # in-place by write/evolve).
        # Keys: random unit-norm so slot_assign has discriminative signal at init.
        init_k = torch.randn(N_m, d_m)
        init_k = init_k / init_k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("k", init_k)                                  # keys (unit-norm)
        self.register_buffer("v", torch.zeros(N_m, d_m))                   # values
        self.register_buffer("q", torch.zeros(N_m, d_m))                   # dynamics state
        self.register_buffer("age", torch.zeros(N_m))
        self.register_buffer("usage", torch.zeros(N_m))
        self.register_buffer("conf", torch.full((N_m,), 0.5))
        self.register_buffer("plast", torch.full((N_m,), 1.0))
        self.register_buffer("conflict", torch.zeros(N_m))
        self.register_buffer("tau_write", torch.zeros(N_m))
        self.register_buffer("tau_use", torch.zeros(N_m))
        self.register_buffer("chi_slot", torch.zeros(N_m, cfg.d_chronometric))

        # Learned components:
        # Write projections: hidden state -> key, value
        self.W_k_m = nn.Linear(cfg.d_model, d_m, bias=False)
        self.W_v_m = nn.Linear(cfg.d_model, d_m, bias=False)

        # Relevance head: [h_L; b; z] -> scalar
        self.relevance_head = nn.Linear(cfg.d_model + cfg.d_model + cfg.d_temporal_state, 1, bias=True)

        # Update-rate predictor input: [v; z; c; rho; a] -> scalar gate
        self.update_rate_head = nn.Linear(d_m + cfg.d_temporal_state + 3, 1, bias=True)

        # Lambda decay predictor for evolution
        self.lambda_head = nn.Linear(d_m + d_m + cfg.d_temporal_state + 2, 1, bias=True)

        # Evolution transition matrices
        self.W_A = nn.Linear(d_m, d_m, bias=False)
        self.W_phi = nn.Linear(d_m, d_m, bias=False)
        self.W_v_phi = nn.Linear(d_m, d_m, bias=False)
        self.W_q_phi = nn.Linear(d_m, d_m, bias=False)
        self.W_z_phi = nn.Linear(cfg.d_temporal_state, d_m, bias=False)

        # GRU for slot dynamics state q_t
        self.q_gru = nn.GRUCell(d_m + cfg.d_temporal_state + d_m, d_m)

        # Temporal compatibility projection
        self.W_chi = nn.Linear(cfg.d_chronometric, d_m, bias=False)
        self.W_chi_slot = nn.Linear(cfg.d_chronometric, d_m, bias=False)

    # ---------- State management ----------

    def reset(self):
        """Wipe all slots (for new stream / per-user load)."""
        with torch.no_grad():
            self.k.zero_()
            self.v.zero_()
            self.q.zero_()
            self.age.zero_()
            self.usage.zero_()
            self.conf.fill_(0.5)
            self.plast.fill_(1.0)
            self.conflict.zero_()
            self.tau_write.zero_()
            self.tau_use.zero_()
            self.chi_slot.zero_()

    def state_dict_dynamic(self) -> dict:
        """Snapshot dynamic memory state (for save/load)."""
        return {
            "k": self.k.clone(),
            "v": self.v.clone(),
            "q": self.q.clone(),
            "age": self.age.clone(),
            "usage": self.usage.clone(),
            "conf": self.conf.clone(),
            "plast": self.plast.clone(),
            "conflict": self.conflict.clone(),
            "tau_write": self.tau_write.clone(),
            "tau_use": self.tau_use.clone(),
            "chi_slot": self.chi_slot.clone(),
        }

    def load_state_dict_dynamic(self, d: dict):
        with torch.no_grad():
            for name, t in d.items():
                getattr(self, name).copy_(t)

    # ---------- Prefix attention read ----------

    def prefix_attention(
        self,
        queries: torch.Tensor,           # (K_p, d_m)
        chi_t: torch.Tensor,             # (d_chronometric,)
        tau_t: float,
    ) -> torch.Tensor:
        """Compute attention weights over all slots for K_p prefix queries.
        Returns alpha: (K_p, N_m)."""
        cfg = self.cfg
        N_m = cfg.n_slots
        d_m = cfg.d_memory

        # Update derived metrics that depend on current tau
        age = tau_t - self.tau_write                          # (N_m,)
        disuse = tau_t - self.tau_use                         # (N_m,)

        # Dot product score
        scores = queries @ self.k.t() / math.sqrt(d_m)        # (K_p, N_m)

        # Bias terms
        scores = scores + cfg.beta_c * self.conf
        scores = scores + cfg.beta_u * torch.log1p(self.usage)
        scores = scores - cfg.beta_a * age
        scores = scores - cfg.beta_d * disuse
        scores = scores - cfg.beta_delta * self.conflict

        # Temporal compatibility
        q_chi = self.W_chi(chi_t)                             # (d_m,)
        k_chi = self.W_chi_slot(self.chi_slot)                # (N_m, d_m)
        cos = F.cosine_similarity(
            q_chi.unsqueeze(0).expand(N_m, -1), k_chi, dim=-1
        )                                                     # (N_m,)
        scores = scores + cfg.beta_tau * cos

        alpha = F.softmax(scores, dim=-1)                     # (K_p, N_m)
        return alpha

    def read_values(self, alpha: torch.Tensor) -> torch.Tensor:
        """alpha: (K_p, N_m) -> raw prefix vectors P_hat: (K_p, d_m)."""
        return alpha @ self.v                                 # (K_p, d_m)

    # ---------- Write ----------

    def compute_write_scores(
        self,
        h_L: torch.Tensor,                                    # (L, d_model)
        b: torch.Tensor,                                      # (L, d_model)
        z: torch.Tensor,                                      # (d_temporal_state,)
        surprise: torch.Tensor,                               # (L,)
        novelty: torch.Tensor,                                # (L,)
        u_prefix: torch.Tensor,                               # (L,)
    ) -> torch.Tensor:
        """Write score per token. Returns omega: (L,) in (0,1)."""
        cfg = self.cfg
        L = h_L.shape[0]
        z_exp = z.unsqueeze(0).expand(L, -1)
        rel_input = torch.cat([h_L, b, z_exp], dim=-1)        # (L, d_model+d_model+d_z)
        relevance = self.relevance_head(rel_input).squeeze(-1)
        score = (
            cfg.lambda_s * surprise
            + cfg.lambda_n * novelty
            + cfg.lambda_r * relevance
            + cfg.lambda_p * u_prefix
        )
        return torch.sigmoid(score)

    def slot_assign(self, k_hat: torch.Tensor) -> torch.Tensor:
        """k_hat: (K_w, d_m). Returns beta: (K_w, N_m) — hard top-1 one-hot for v1.

        SPEC.tex §"Slot assignment": "Use top-1 assignment for debugging, then
        top-2 soft assignment if slot collapse is controlled." We start with
        hard top-1 so each candidate hits exactly one slot.
        """
        cfg = self.cfg
        k_norm = F.normalize(k_hat, dim=-1)                   # (K_w, d_m)
        K_norm = F.normalize(self.k, dim=-1)                  # (N_m, d_m)
        sim = k_norm @ K_norm.t()                             # (K_w, N_m)
        scores = (
            cfg.eta_sim * sim
            - cfg.eta_c * self.conf
            - cfg.eta_u * torch.log1p(self.usage)
            - cfg.eta_delta * self.conflict
        )                                                     # (K_w, N_m)
        best_slot = scores.argmax(dim=-1)                     # (K_w,)
        beta = F.one_hot(best_slot, num_classes=cfg.n_slots).to(scores.dtype)
        return beta

    @torch.no_grad()
    def write(
        self,
        h_L: torch.Tensor,                                    # (L, d_model)
        b: torch.Tensor,                                      # (L, d_model)
        z: torch.Tensor,                                      # (d_temporal_state,)
        surprise: torch.Tensor,                               # (L,)
        novelty: torch.Tensor,                                # (L,)
        u_prefix: torch.Tensor,                               # (L,)
        tau_t: float,
        chi_t: torch.Tensor,                                  # (d_chronometric,)
    ):
        """Apply the full write step in-place."""
        cfg = self.cfg
        L = h_L.shape[0]

        omega = self.compute_write_scores(h_L, b, z, surprise, novelty, u_prefix)  # (L,)
        # Pick top-K_w candidates
        K_w = min(cfg.n_write_candidates, L)
        top = torch.topk(omega, K_w)
        idx = top.indices                                     # (K_w,)
        cand_omega = top.values                               # (K_w,)
        h_cand = h_L[idx]                                     # (K_w, d_model)
        k_hat = self.W_k_m(h_cand)                            # (K_w, d_m)
        v_hat = self.W_v_m(h_cand)                            # (K_w, d_m)

        beta = self.slot_assign(k_hat)                        # (K_w, N_m)

        # Compute per-slot update direction (weighted by omega and beta)
        # k_target[i] = sum_j beta[j,i] * omega[j] * k_hat[j]
        weights = beta * cand_omega.unsqueeze(-1)             # (K_w, N_m)
        k_target = weights.t() @ k_hat                        # (N_m, d_m)
        v_target = weights.t() @ v_hat                        # (N_m, d_m)
        used_mass = weights.sum(dim=0)                        # (N_m,)

        # Update rate per slot
        # eta[i] = sigmoid(w_eta^T [v_i; z; c_i; rho_i; a_i]) * plast_i
        z_exp = z.unsqueeze(0).expand(cfg.n_slots, -1)
        age = tau_t - self.tau_write
        rate_input = torch.cat(
            [self.v, z_exp, self.conf.unsqueeze(-1), self.plast.unsqueeze(-1), age.unsqueeze(-1)],
            dim=-1,
        )
        eta = torch.sigmoid(self.update_rate_head(rate_input).squeeze(-1)) * self.plast  # (N_m,)
        # Only update slots that received nonzero mass
        eta = eta * (used_mass > 1e-6).float()

        # Apply update
        one_minus = 1.0 - eta
        new_k = one_minus.unsqueeze(-1) * self.k + eta.unsqueeze(-1) * k_target
        new_k = F.normalize(new_k, dim=-1)                    # renormalize keys
        new_v = one_minus.unsqueeze(-1) * self.v + eta.unsqueeze(-1) * v_target

        # Conflict: cosine distance between old v and new contribution
        contribution = v_target / (used_mass.unsqueeze(-1) + 1e-6)
        old_norm = F.normalize(self.v, dim=-1)
        contrib_norm = F.normalize(contribution, dim=-1)
        cos_old_new = (old_norm * contrib_norm).sum(dim=-1)
        delta_conflict = (1.0 - cos_old_new) * (used_mass > 1e-6).float()

        # Confidence update
        new_conf = torch.clamp(
            self.conf
            + cfg.xi_use * (used_mass > 1e-6).float()
            - cfg.xi_conf * delta_conflict
            - cfg.xi_age * age,
            0.0,
            1.0,
        )

        # Plasticity update (per-slot avg novelty over candidates that wrote here)
        # For simplicity in v1: novelty_bar = mean of novelty[idx] weighted by beta
        nov_per_cand = novelty[idx]                           # (K_w,)
        nov_per_slot = (weights * nov_per_cand.unsqueeze(-1)).sum(dim=0) / (used_mass + 1e-6)
        new_plast = torch.clamp(1.0 - new_conf + cfg.xi_nov * nov_per_slot, 0.0, 1.0)

        # Write all in-place
        self.k.copy_(new_k)
        self.v.copy_(new_v)
        self.conf.copy_(new_conf)
        self.plast.copy_(new_plast)
        self.conflict.add_((delta_conflict).detach())

        # Timestamp: slots that received nonzero mass get tau_write updated
        touched = (used_mass > 1e-6).float()
        self.tau_write.copy_(touched * tau_t + (1.0 - touched) * self.tau_write)
        # chi_slot for newly-decisively-written slots: copy chi_t
        chi_t_exp = chi_t.unsqueeze(0).expand(cfg.n_slots, -1)
        self.chi_slot.copy_(touched.unsqueeze(-1) * chi_t_exp + (1.0 - touched).unsqueeze(-1) * self.chi_slot)

    # ---------- Evolution ----------

    @torch.no_grad()
    def evolve(self, z: torch.Tensor, chi_t: torch.Tensor, delta_tau: float):
        """Apply autonomous evolution. Δτ-driven."""
        cfg = self.cfg
        d_m = cfg.d_memory
        N_m = cfg.n_slots
        dtau_clipped = min(max(delta_tau, 0.0), cfg.delta_tau_max)

        # Slot interaction graph: sparse top-K neighbors
        k_norm = F.normalize(self.k, dim=-1)
        sim = k_norm @ k_norm.t() / math.sqrt(d_m)            # (N_m, N_m)
        sim.fill_diagonal_(-1e9)                               # no self-loop in neighbor agg
        topk = torch.topk(sim, cfg.neighbor_k, dim=-1)
        neighbor_idx = topk.indices                            # (N_m, K)
        neighbor_weights = F.softmax(topk.values, dim=-1)      # (N_m, K)

        # Aggregate neighbor values
        neighbor_v = self.v[neighbor_idx]                      # (N_m, K, d_m)
        agg = (neighbor_weights.unsqueeze(-1) * neighbor_v).sum(dim=1)  # (N_m, d_m)
        agg = self.W_A(agg)                                    # (N_m, d_m)

        # Per-slot decay rate lambda
        z_exp = z.unsqueeze(0).expand(N_m, -1)
        age = self.age                                         # not used directly; use as proxy for decay weighting
        lambda_input = torch.cat(
            [self.v, self.q, z_exp, self.conf.unsqueeze(-1), self.plast.unsqueeze(-1)],
            dim=-1,
        )
        lam = torch.sigmoid(self.lambda_head(lambda_input).squeeze(-1))  # (N_m,)

        # Dynamics input
        dyn = torch.tanh(self.W_v_phi(self.v) + self.W_q_phi(self.q) + self.W_z_phi(z_exp))
        dyn = self.W_phi(dyn)

        delta_v = -lam.unsqueeze(-1) * self.v + agg + dyn      # (N_m, d_m)
        v_new = F.layer_norm(
            self.v + cfg.eps_dyn * dtau_clipped * delta_v,
            normalized_shape=(d_m,),
        )

        # Update q via GRU
        gru_input = torch.cat([v_new, z_exp, delta_v], dim=-1)
        q_new = self.q_gru(gru_input, self.q)

        self.v.copy_(v_new)
        self.q.copy_(q_new)

    # ---------- Usage signal for consolidation ----------

    @torch.no_grad()
    def update_usage(
        self,
        alpha_p: torch.Tensor,                                # (K_p, N_m)
        u_prefix_bar: float,                                  # scalar mean over chunk
        gate_bar: float,                                      # scalar mean prefix broadcast gate
    ):
        """Incremental usage counter update per spec."""
        cfg = self.cfg
        # Sum attention mass over all prefix queries
        used_mass = alpha_p.sum(dim=0)                        # (N_m,)
        increment = used_mass * max(0.0, u_prefix_bar) * gate_bar
        new_usage = cfg.lambda_u_decay * self.usage + increment
        self.usage.copy_(new_usage)
        # Update tau_use for slots that received any prefix mass
        # (we don't have current tau here; caller should set tau_use externally if needed)

    @torch.no_grad()
    def mark_used(self, alpha_p: torch.Tensor, tau_t: float, threshold: float = 1e-3):
        """Update tau_use for slots with nontrivial attention mass."""
        used_mass = alpha_p.sum(dim=0)                        # (N_m,)
        touched = (used_mass > threshold).float()
        self.tau_use.copy_(touched * tau_t + (1.0 - touched) * self.tau_use)
