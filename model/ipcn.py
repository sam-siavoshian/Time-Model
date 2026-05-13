"""Top-level IPCN model.

Implements forward_chunk() = the canonical inference algorithm from SPEC.tex:

  1. e_t       <- token_embeddings(X_t)
  2. s_t       <- shallow_sketch(e_t, z_{t-1})
  3. prefix queries Q_p over slots via temporal+confidence+usage biases
  4. P_t       <- PFC(P_hat, sketch, z, chi; Omega)
  5. b_tj      <- token_prefix_broadcast(e_tj, P_t)
  6. e_tilde   <- LN(e + lambda_pre * gate * W_b b)   (Route 2)
  7. H0        <- concat(P_t, e_tilde)
  8. logits, H_L <- CoreTransformer(H0; theta, Omega)   (Routes 1+3 inside)
  9. write candidates -> memory write
 10. z_t       <- update_temporal_self_state
 11. memory.evolve(z_t, chi_t, delta_tau)
 12. if consolidation_phase: run_consolidation_pass
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.chronometric import ChronometricEncoder
from model.config import IPCNConfig
from model.core import CoreTransformer
from model.injection import BroadcastPreconditioner, schedule_lambda_pre
from model.late_retrieval import LateRetrievalHead
from model.memory import MemoryBank
from model.pfc import PrefixFormingController


@dataclass
class ChunkOutput:
    logits: torch.Tensor                                     # (L, V)
    hidden_last: torch.Tensor                                # (S, d_model) — full sequence including prefix
    prefix: torch.Tensor                                     # (K_p, d_model)
    alpha_prefix: torch.Tensor                               # (K_p, N_m)
    b_broadcast: torch.Tensor                                # (L, d_model)
    gate: torch.Tensor                                       # (L, d_model)
    H0: Optional[torch.Tensor] = None                        # (S, d_model) — input to layer 0 if requested
    hidden_layers: Optional[list[torch.Tensor]] = None       # list of (S, d_model) per core layer


class IPCN(nn.Module):
    def __init__(self, cfg: IPCNConfig):
        super().__init__()
        self.cfg = cfg
        self.chrono = ChronometricEncoder(cfg)
        self.memory = MemoryBank(cfg)
        self.pfc = PrefixFormingController(cfg)
        self.broadcast = BroadcastPreconditioner(cfg)
        self.core = CoreTransformer(cfg)
        self.late_retrieval = LateRetrievalHead(cfg)         # A1 baseline path

        # Temporal self-state z_t
        self.register_buffer("z", torch.zeros(cfg.d_temporal_state))

        # GRU that updates z from per-chunk summary
        # b_t bundles: [h_bar, p_bar, ell_bar, H(alpha_pre), ||M_t - M_{t-1}||, c_bar, u_bar, delta_bar]
        bundle_dim = cfg.d_model + cfg.d_model + 1 + 1 + 1 + 1 + 1 + 1
        self.z_gru = nn.GRUCell(bundle_dim, cfg.d_temporal_state)

        # Memory-usefulness predictor (P_U): pooled sketch + pooled prefix + z -> scalar.
        # Inputs we pool: sketch (size built from W_s output), prefix mean, z.
        self.P_U = nn.Linear(cfg.d_model + cfg.d_temporal_state, 1, bias=True)

        # Self-state predictor (P_z): pooled memory + z -> next z. Pooled memory is mean(v).
        self.P_z = nn.Linear(cfg.d_memory + cfg.d_temporal_state, cfg.d_temporal_state, bias=True)

        # Delta memory predictor (P_M): pooled memory + z -> next pooled memory.
        self.P_M = nn.Linear(cfg.d_memory + cfg.d_temporal_state, cfg.d_memory, bias=True)

        # Chronometric prediction head (delta_tau_hat from final hidden state pooled)
        self.dtau_head = nn.Linear(cfg.d_model, 1, bias=True)

        # Training step counter for lambda_pre schedule
        self.register_buffer("train_step", torch.zeros(1, dtype=torch.long))

    # ---------- Public API ----------

    def reset_memory(self):
        self.memory.reset()
        self.z.zero_()

    def forward_chunk(
        self,
        input_ids: torch.Tensor,                             # (L,)
        tau_t: float,
        delta_tau: float,
        gap_flag: float = 0.0,
        event_density: float = 0.0,
        zero_prefix_for_ablation: bool = False,
        return_hidden_layers: bool = False,
    ) -> ChunkOutput:
        cfg = self.cfg
        device = input_ids.device
        L = input_ids.shape[0]

        # 1. Embed
        e = self.core.embed(input_ids)                       # (L, d_model)

        # 2. Chronometric vector
        chi_t = self.chrono(
            tau=torch.tensor([tau_t], device=device, dtype=torch.float32),
            delta_tau=torch.tensor([delta_tau], device=device, dtype=torch.float32),
            event_density=torch.tensor([event_density], device=device, dtype=torch.float32),
            gap_flag=torch.tensor([gap_flag], device=device, dtype=torch.float32),
        ).squeeze(0)                                          # (d_chronometric,)

        # 3-4. PFC builds prefix
        prefix, alpha_prefix = self.pfc(e, self.z, chi_t, self.memory, tau_t)  # prefix: (K_p, d_model)
        if zero_prefix_for_ablation or not cfg.enable_episodic_memory or cfg.enable_late_retrieval_only:
            prefix = torch.zeros_like(prefix)

        # 5-6. Route 2: broadcast preconditioning
        step = int(self.train_step.item())
        lam_pre = schedule_lambda_pre(step, cfg)
        if cfg.use_route2_broadcast:
            e_tilde, b, gate = self.broadcast(e, prefix, self.z, lam_pre)
        else:
            e_tilde = e
            b = torch.zeros_like(e)
            gate = torch.zeros_like(e)

        # 7. H0
        H0 = torch.cat([prefix, e_tilde], dim=0)             # (K_p + L, d_model)

        # 8. Core forward. Route-3 LN modulation inside.
        # b_full: same shape as H0. Prefix rows get zero modulation, content rows get b.
        if cfg.use_route3_lnmod:
            K_p = cfg.prefix_length
            b_full = torch.zeros(K_p + L, cfg.d_model, device=device, dtype=e.dtype)
            b_full[K_p:] = b
        else:
            b_full = None

        logits, hidden_layers, hidden_last = self.core(
            H0, b_full=b_full, return_hidden=return_hidden_layers
        )

        # Mark slots as used (for tau_use bookkeeping)
        with torch.no_grad():
            self.memory.mark_used(alpha_prefix.detach(), tau_t=tau_t)

        return ChunkOutput(
            logits=logits,
            hidden_last=hidden_last,
            prefix=prefix,
            alpha_prefix=alpha_prefix,
            b_broadcast=b,
            gate=gate,
            H0=H0 if return_hidden_layers else None,
            hidden_layers=hidden_layers,
        )

    # ---------- Per-chunk write + evolve + z update ----------

    def update_memory_and_state(
        self,
        out: ChunkOutput,
        surprise: torch.Tensor,                              # (L,)
        novelty: torch.Tensor,                               # (L,)
        u_prefix: torch.Tensor,                              # (L,)
        u_prefix_bar: float,
        gate_bar: float,
        tau_t: float,
        delta_tau: float,
        chi_t_vec: Optional[torch.Tensor] = None,
        do_evolve: bool = True,
    ):
        """Apply write -> z update -> evolve. Stateful."""
        cfg = self.cfg
        device = out.logits.device
        L = out.logits.shape[0]

        # Re-derive chi if not passed
        if chi_t_vec is None:
            chi_t_vec = self.chrono(
                tau=torch.tensor([tau_t], device=device, dtype=torch.float32),
                delta_tau=torch.tensor([delta_tau], device=device, dtype=torch.float32),
            ).squeeze(0)

        # Hidden state for content positions (drop prefix rows)
        h_L_content = out.hidden_last[cfg.prefix_length:]    # (L, d_model)

        # Snapshot memory for evolution self-prediction loss later
        prev_mem_snap = {
            "v": self.memory.v.clone(),
            "z": self.z.clone(),
        }

        # Write
        self.memory.write(
            h_L=h_L_content,
            b=out.b_broadcast,
            z=self.z,
            surprise=surprise,
            novelty=novelty,
            u_prefix=u_prefix,
            tau_t=tau_t,
            chi_t=chi_t_vec,
        )

        # Update usage counters (for consolidation eligibility)
        self.memory.update_usage(out.alpha_prefix.detach(), u_prefix_bar, gate_bar)

        # Update z
        with torch.no_grad():
            mem_velocity = (self.memory.v - prev_mem_snap["v"]).norm()
            alpha_entropy = -(out.alpha_prefix.clamp_min(1e-12) * out.alpha_prefix.clamp_min(1e-12).log()).sum(-1).mean()
            bundle = torch.cat(
                [
                    h_L_content.mean(dim=0),
                    out.prefix.mean(dim=0),
                    torch.tensor([surprise.mean().item()], device=device),
                    torch.tensor([alpha_entropy.item()], device=device),
                    torch.tensor([mem_velocity.item()], device=device),
                    torch.tensor([self.memory.conf.mean().item()], device=device),
                    torch.tensor([self.memory.usage.mean().item()], device=device),
                    torch.tensor([self.memory.conflict.mean().item()], device=device),
                ],
                dim=-1,
            )
        new_z = self.z_gru(bundle.unsqueeze(0), self.z.unsqueeze(0)).squeeze(0)
        with torch.no_grad():
            self.z.copy_(new_z.detach())

        if do_evolve and cfg.enable_evolution:
            self.memory.evolve(self.z, chi_t_vec, delta_tau)

        return prev_mem_snap

    # ---------- Eligibility for consolidation ----------

    def eligible_slots_for_consolidation(self) -> list[int]:
        """v1: replace age >= t_stable check with simpler 'slot has been written'.

        SPEC.tex calls for stability >= 512 chunks, but the mem.age buffer
        is not updated per-chunk currently. We use tau_write > 0 as a
        proxy: a slot is eligible if kappa > tau_cons AND it has been
        decisively written at least once. Tighter stability tracking is
        added in Phase 2+ when long-horizon consolidation matters.
        """
        cfg = self.cfg
        mem = self.memory
        kappa = (
            torch.log1p(mem.usage)
            * mem.conf
            * (1.0 - mem.conflict.clamp(0, 1))
            * (1.0 - mem.plast)
        )
        written = (mem.tau_write > 0)
        eligible = (kappa > cfg.tau_cons) & written
        return [i for i, ok in enumerate(eligible.tolist()) if ok]
