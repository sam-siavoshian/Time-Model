"""Locked IPCN hyperparameters.

Single source of truth. Pulled directly from SPEC.tex / ARCHITECTURE_LOCKED.md.
No deviation allowed without a pre-registration amendment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class IPCNConfig:
    # ----- Core transformer -----
    vocab_size: int = 50257                                  # GPT-2 BPE
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ffn: int = 2048
    chunk_length: int = 256                                  # L
    local_window: int = 1024                                 # local attention max
    dropout: float = 0.0

    # ----- Prefix-forming controller (PFC) -----
    pfc_n_layers: int = 2
    pfc_n_heads: int = 4
    pfc_d_model: int = 512
    pfc_d_ffn: int = 1024
    prefix_length: int = 32                                  # K_p

    # ----- Episodic memory -----
    n_slots: int = 256                                       # N_m
    d_memory: int = 256                                      # d_m
    d_temporal_state: int = 128                              # d_z
    n_write_candidates: int = 16                             # K_w
    neighbor_k: int = 8                                      # |N(i)|

    # ----- Chronometric encoding -----
    timescales: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536)

    # ----- LoRA -----
    lora_rank: int = 8                                       # start. 16 on scale-up.
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    consolidated_layers: Tuple[int, ...] = (0, 1, 2)         # core layers with LoRA
    consolidate_pfc: bool = True

    # ----- Injection routes -----
    use_route1_prepend: bool = True                          # mandatory
    use_route2_broadcast: bool = True                        # strongly recommended
    use_route3_lnmod: bool = True                            # optional, layers 1-2
    lambda_pre_init: float = 0.5                             # broadcast strength at step 0
    lambda_pre_final: float = 1.0                            # broadcast strength at step 100k
    lambda_pre_anneal_steps: int = 100_000
    alpha_film: float = 0.1                                  # LayerNorm modulation scale
    lnmod_layers: Tuple[int, ...] = (1, 2)

    # ----- Prefix-memory attention biases -----
    beta_c: float = 0.5                                      # confidence bias
    beta_u: float = 0.2                                      # log(1+usage) bias
    beta_a: float = 0.05                                     # age penalty
    beta_d: float = 0.03                                     # disuse penalty
    beta_delta: float = 0.4                                  # conflict penalty
    beta_tau: float = 0.25                                   # temporal compatibility bonus

    # ----- Write score weights -----
    lambda_s: float = 0.7                                    # surprise
    lambda_n: float = 0.5                                    # novelty
    lambda_r: float = 0.4                                    # relevance
    lambda_p: float = 0.8                                    # prefix attribution

    # ----- Slot assignment -----
    eta_sim: float = 2.0
    eta_c: float = 0.5
    eta_u: float = 0.2
    eta_delta: float = 0.7

    # ----- Confidence/plasticity updates -----
    xi_use: float = 0.05
    xi_conf: float = 0.10
    xi_age: float = 0.001
    xi_nov: float = 0.05

    # ----- Evolution dynamics -----
    eps_dyn: float = 0.1
    delta_tau_max: float = 65536.0                           # clip for stability

    # ----- Consolidation -----
    tau_cons: float = 3.0                                    # eligibility threshold
    t_stable: int = 512                                      # min stable chunks
    eta_cons: float = 1e-5                                   # LoRA update LR
    lambda_omega: float = 1e-4                               # adapter weight decay
    lambda_ewc: float = 1e-3                                 # EWC regularizer
    lambda_u_decay: float = 0.995                            # usage counter decay
    eps_drop: float = 0.02                                   # hard-consolidation accuracy gate
    kl_drift_threshold: float = 0.02                         # LM drift rollback
    contradiction_floor: float = 0.01                        # contradiction drop limit
    consolidation_frequency: int = 256                       # chunks between consolidation passes
    consolidation_adapter_steps: int = 3                     # SGD steps per consolidation batch

    # ----- Loss weights (default) -----
    w_lm: float = 1.0
    w_pre_influence: float = 0.02
    w_precision: float = 0.02
    w_mem_predict: float = 0.05
    w_diversity: float = 0.001
    w_slot_util: float = 0.001
    w_evolution: float = 0.02
    w_chrono: float = 0.03
    w_cons: float = 0.1                                      # during consolidation phases

    # ----- Pre-computational influence loss thresholds -----
    rho_helped: float = 0.03                                 # U_t > rho means prefix helped
    tau_gate: float = 0.2                                    # min mean prefix gate when memory helped

    # ----- Chronometric loss weights (inside L_chrono) -----
    lambda_dur: float = 1.0
    lambda_phase: float = 0.5
    lambda_future: float = 0.5

    # ----- Optimizer -----
    base_lr: float = 3e-4
    adapter_lr: float = 1e-5
    optimizer: str = "adamw"
    bf16: bool = True

    # ----- Backprop through chunks -----
    bptt_chunks: int = 4                                     # then detach memory

    # ----- Phase configs (step counts) -----
    phase0_steps: int = 100_000
    phase1_steps: int = 50_000
    phase2_steps: int = 50_000
    phase3_steps: int = 100_000

    # ----- Derived -----
    @property
    def d_chronometric(self) -> int:
        # 2 scalars (tau, delta_tau), psi(tau)+psi(delta_tau) at 13 scales x 3 features each, plus nu_t + gap_t
        return 2 + 2 * 3 * len(self.timescales) + 1 + 1

    @property
    def n_timescales(self) -> int:
        return len(self.timescales)


# Default instance, importable as: from model.config import DEFAULT
DEFAULT = IPCNConfig()
