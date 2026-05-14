"""IPCN wrapped around a pretrained Qwen 2.5 base model.

Track B from the day-one report. The 102M from-scratch model in Track A
lacked the representational capacity for synthetic rule-following; the
memory mechanism executed safely but had no behavior to migrate. Track B
ports the same IPCN scaffolding (PFC + memory bank + chronometric encoder
+ LoRA + consolidation) around a Qwen 2.5 base that already speaks
language fluently. The base is frozen; only the IPCN-specific modules
and a thin LoRA over Qwen's first few attention blocks train.

Architectural choices:
  - Base: Qwen 2.5 1.5B-Instruct (smallest fast iteration; can scale to
    3B / 7B once the smoke pipeline is green).
  - Memory bank: 128 slots * d_memory matching Qwen's hidden_size
    (1536 for 1.5B, 2048 for 3B, 3584 for 7B). Each slot stores a
    candidate next-state-bias vector that can be merged into Qwen's
    layer-0 activations.
  - PFC: 2-layer transformer over K_p=16 prefix tokens whose output is
    PREPENDED to Qwen's input embeddings. Qwen's causal attention then
    processes the (K_p + L) sequence; the K_p prefix tokens are the
    IPCN "involuntary prefix".
  - Chronometric encoder: same as Track A. Linear-projected to d_model.
  - LoRA: rank 8 on Qwen layers [0, 1] q/k/v/o + the lm_head (optional).
  - Consolidation: same teacher/student KL pattern, distilling slot
    behavior into LoRA + lm_head adapters.

This file is the WRAPPER class. Training, eval, and consolidation use
the existing pipeline (with adjustments for the bigger d_model and the
Qwen tokenizer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QwenIPCNConfig:
    """Config for the Qwen-wrapped IPCN. Independent from IPCNConfig so we
    can keep Track A and Track B side-by-side without confusing tooling.
    """
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # PFC over the involuntary prefix
    prefix_length: int = 16
    pfc_n_layers: int = 2
    pfc_n_heads: int = 8

    # Memory bank
    n_slots: int = 128
    d_memory: int = 0                                          # 0 = match base hidden_size

    # Chronometric encoder (matches Track A's design)
    timescales: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536)

    # LoRA
    lora_rank: int = 8
    lora_layers: Tuple[int, ...] = (0, 1)                      # which Qwen layers to adapt
    lora_targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    lora_lm_head: bool = True                                  # also LoRA the language model head

    # Memory write / read bias coefficients (mirror Track A defaults)
    beta_c: float = 0.5
    beta_u: float = 0.2
    beta_a: float = 0.1
    beta_d: float = 0.1
    beta_delta: float = 0.4
    beta_tau: float = 0.3

    eta_sim: float = 2.0
    eta_c: float = 0.5
    eta_u: float = 0.2
    eta_delta: float = 0.7

    # Write-score weights
    lambda_s: float = 1.0
    lambda_n: float = 1.0
    lambda_r: float = 0.5
    lambda_p: float = 0.5

    # Conf / plast update
    xi_use: float = 0.1
    xi_conf: float = 0.2
    xi_age: float = 0.001
    xi_nov: float = 0.1

    # Consolidation eligibility
    tau_cons: float = 1.0                                      # use 1.0 (not Track A's 3.0; we have shorter runs)
    consolidation_frequency: int = 256
    kl_drift_threshold: float = 0.05
    eps_drop: float = 0.02

    # Misc
    n_write_candidates: int = 4
    enable_episodic_memory: bool = True
    enable_evolution: bool = False                             # disabled in Track B for simplicity
    chunk_length: int = 256
    lambda_u_decay: float = 0.99


class _Chronometric(nn.Module):
    """Multi-scale sinusoidal encoding of elapsed time tau.

    Identical to Track A. Each tau_t value gets encoded across 13 time
    scales (2 -> 65536) as (sin, cos) plus a log(1+tau) channel. Output
    dimension = 2 * n_scales + 1 = 27.
    """
    def __init__(self, timescales: Tuple[int, ...]):
        super().__init__()
        self.register_buffer("scales", torch.tensor(timescales, dtype=torch.float32))

    @property
    def out_dim(self) -> int:
        return 2 * self.scales.numel() + 1

    def forward(self, tau: torch.Tensor, delta_tau: torch.Tensor) -> torch.Tensor:
        tau = tau.float()                                      # (B,) or scalar
        if tau.dim() == 0:
            tau = tau.unsqueeze(0)
        x = tau.unsqueeze(-1) / self.scales                    # (B, S)
        chi = torch.cat([torch.sin(x), torch.cos(x), torch.log1p(tau.unsqueeze(-1))], dim=-1)
        return chi                                             # (B, 2*S + 1)


class _LoRALinear(nn.Module):
    """Wraps an existing nn.Linear and adds a rank-r additive adapter:
    output = base(x) + (x @ A.T) @ B.T * scaling.

    The base.weight is frozen; only A, B train. Saves memory by NOT
    cloning the base; we replace forward with a custom one.
    """
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        in_f = base.in_features
        out_f = base.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # x: (..., in_features); compute (x @ A.T) @ B.T
        # Cast x to LoRA param dtype so the matmul works when the base is
        # in bf16 but LoRA params are in fp32 (mixed-precision pattern).
        lora_dtype = self.lora_A.dtype
        x_l = x.to(lora_dtype)
        adapter = F.linear(F.linear(x_l, self.lora_A), self.lora_B) * self.scaling
        return base_out + adapter.to(base_out.dtype)


class _MemoryBankQwen(nn.Module):
    """Episodic memory bank sized for Qwen's hidden dim.

    Same operations as model.memory.MemoryBank but parameterized for
    Track B's bigger d_memory. We keep this module self-contained so
    Track A and Track B can be loaded in the same process without
    config bleed.
    """
    def __init__(self, cfg: QwenIPCNConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        d_m = cfg.d_memory if cfg.d_memory > 0 else d_model
        self.d_memory = d_m
        N_m = cfg.n_slots
        # Slot state buffers
        init_k = torch.randn(N_m, d_m)
        init_k = init_k / init_k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("k", init_k)
        self.register_buffer("v", torch.zeros(N_m, d_m))
        self.register_buffer("age", torch.zeros(N_m))
        self.register_buffer("usage", torch.zeros(N_m))
        self.register_buffer("conf", torch.full((N_m,), 0.5))
        self.register_buffer("plast", torch.full((N_m,), 1.0))
        self.register_buffer("conflict", torch.zeros(N_m))
        self.register_buffer("tau_write", torch.zeros(N_m))
        self.register_buffer("tau_use", torch.zeros(N_m))

        # Write projections from d_model -> d_memory
        self.W_k_m = nn.Linear(d_model, d_m, bias=False)
        self.W_v_m = nn.Linear(d_model, d_m, bias=False)

    def reset(self):
        with torch.no_grad():
            new_k = torch.randn(self.cfg.n_slots, self.d_memory, device=self.k.device, dtype=self.k.dtype)
            new_k = new_k / new_k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            self.k.copy_(new_k)
            self.v.zero_()
            self.age.zero_()
            self.usage.zero_()
            self.conf.fill_(0.5)
            self.plast.fill_(1.0)
            self.conflict.zero_()
            self.tau_write.zero_()
            self.tau_use.zero_()

    @torch.no_grad()
    def write(self, h: torch.Tensor, tau_t: float):
        """Coarse write step: pick top-K_w hidden states by L2 norm, project
        to memory dim, slot-assign by argmax cosine, EMA update with eta=0.3.

        Simpler than Track A's full omega-gated step. The point in Track B
        is to demonstrate slot -> weight migration, not to re-litigate
        the write design.
        """
        L = h.shape[0]
        K_w = min(self.cfg.n_write_candidates, L)
        norms = h.norm(dim=-1)
        top = torch.topk(norms, K_w)
        cand = h[top.indices]
        k_hat = F.normalize(self.W_k_m(cand), dim=-1)
        v_hat = self.W_v_m(cand)
        K_norm = F.normalize(self.k, dim=-1)
        sim = k_hat @ K_norm.t()                              # (K_w, N_m)
        best = sim.argmax(dim=-1)
        beta = F.one_hot(best, num_classes=self.cfg.n_slots).float()
        used_mass = beta.sum(dim=0)                            # (N_m,)
        k_target = beta.t() @ k_hat
        v_target = beta.t() @ v_hat
        eta = 0.3 * (used_mass > 0).float()
        one_minus = 1.0 - eta
        new_k = one_minus.unsqueeze(-1) * self.k + eta.unsqueeze(-1) * k_target
        new_k = F.normalize(new_k, dim=-1)
        new_v = one_minus.unsqueeze(-1) * self.v + eta.unsqueeze(-1) * v_target
        self.k.copy_(new_k)
        self.v.copy_(new_v)
        touched = (used_mass > 0).float()
        self.tau_write.copy_(touched * tau_t + (1.0 - touched) * self.tau_write)
        self.usage.copy_(self.cfg.lambda_u_decay * self.usage + used_mass)


class _PFC(nn.Module):
    """Prefix-Forming Controller.

    Takes K_p learned prefix queries, attends over the memory bank (keys
    + values), and outputs K_p hidden states of dim d_model that will be
    prepended to the base model's input embeddings.
    """
    def __init__(self, cfg: QwenIPCNConfig, d_model: int, d_chrono: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        # Learned prefix query embeddings.
        self.prefix_queries = nn.Parameter(torch.randn(cfg.prefix_length, d_model) * 0.02)
        # Cross-attention from prefix queries to memory bank values.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(0, 0)                          # placeholder; overwritten below
        self.v_proj = nn.Linear(0, 0)                          # placeholder; overwritten below
        # Real key/value projections from memory bank entries to d_model.
        self.mem_k_proj = nn.Linear(0, d_model)                # placeholder, overwritten below
        self.mem_v_proj = nn.Linear(0, d_model)                # placeholder, overwritten below
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        # Chronometric injection projection
        self.chrono_proj = nn.Linear(d_chrono, d_model)

    def attach_memory_dim(self, d_memory: int):
        """Set up projections from memory bank dim to d_model. Called after
        we know the memory bank dim.
        """
        d_model = self.d_model
        self.mem_k_proj = nn.Linear(d_memory, d_model, bias=False).to(self.prefix_queries.device)
        self.mem_v_proj = nn.Linear(d_memory, d_model, bias=False).to(self.prefix_queries.device)

    def forward(
        self,
        memory_k: torch.Tensor,                                # (N_m, d_memory)
        memory_v: torch.Tensor,                                # (N_m, d_memory)
        chi_t: torch.Tensor,                                   # (d_chrono,)
    ) -> torch.Tensor:
        K_p = self.cfg.prefix_length
        d = self.d_model
        # Queries: learned prefix embeddings + chrono bias broadcast.
        chi_bias = self.chrono_proj(chi_t)                     # (d_model,)
        q = self.prefix_queries + chi_bias.unsqueeze(0)        # (K_p, d_model)
        q = self.q_proj(q)
        # Keys/values from memory bank projected to d_model
        k = self.mem_k_proj(memory_k)                          # (N_m, d_model)
        v = self.mem_v_proj(memory_v)                          # (N_m, d_model)
        attn = torch.softmax(q @ k.t() / (d ** 0.5), dim=-1)   # (K_p, N_m)
        prefix = attn @ v                                      # (K_p, d_model)
        prefix = self.ln(prefix + self.out_proj(prefix))       # residual
        return prefix


class QwenIPCN(nn.Module):
    """Frozen-base Qwen + IPCN scaffolding.

    The forward pass is:
      1. Tokenize -> input embeddings (Qwen's embed_tokens, frozen).
      2. Compute chronometric vector chi_t from tau_t.
      3. PFC produces K_p prefix hidden states from current memory bank.
      4. Concatenate (prefix, input_embeds) -> Qwen layers -> logits.
      5. LM loss only on the LAST L tokens (ignore the K_p prefix).
      6. After forward, write hidden states into memory bank.
    """
    def __init__(self, cfg: QwenIPCNConfig):
        super().__init__()
        self.cfg = cfg
        # Lazy-import transformers so users without it can still inspect
        # the rest of the codebase. ImportError surfaces only when the
        # wrapper is actually instantiated.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
        self.base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        # Freeze the entire base by default.
        for p in self.base.parameters():
            p.requires_grad_(False)
        # Discover hidden_size.
        hidden = self.base.config.hidden_size
        self.d_model = hidden
        if cfg.d_memory == 0:
            cfg.d_memory = hidden
        # Memory + chronometric + PFC
        self.memory = _MemoryBankQwen(cfg, hidden)
        self.chrono = _Chronometric(cfg.timescales)
        self.pfc = _PFC(cfg, hidden, self.chrono.out_dim)
        self.pfc.attach_memory_dim(cfg.d_memory)
        # Apply LoRA to selected layers + lm_head.
        self._apply_lora()

    def _apply_lora(self):
        """Find Qwen attention modules in self.cfg.lora_layers and wrap
        their q/k/v/o projections with _LoRALinear. Also optionally LoRA
        the lm_head.
        """
        cfg = self.cfg
        n_adapted = 0
        # Qwen's layers are at self.base.model.layers (list of decoder layers)
        try:
            layers = self.base.model.layers
        except AttributeError:
            # Some checkpoints expose self.base.transformer.h
            layers = self.base.transformer.h
        for li in cfg.lora_layers:
            block = layers[li]
            # Qwen 2.5 attention is at block.self_attn with q_proj, k_proj, v_proj, o_proj
            attn = getattr(block, "self_attn", None) or getattr(block, "attn", None)
            if attn is None:
                continue
            for proj_name in cfg.lora_targets:
                proj = getattr(attn, proj_name, None)
                if proj is None or not isinstance(proj, nn.Linear):
                    continue
                wrapped = _LoRALinear(proj, rank=cfg.lora_rank)
                setattr(attn, proj_name, wrapped)
                n_adapted += 1
        if cfg.lora_lm_head and hasattr(self.base, "lm_head"):
            self.base.lm_head = _LoRALinear(self.base.lm_head, rank=cfg.lora_rank)
            n_adapted += 1
        self._n_lora_modules = n_adapted

    def reset_memory(self):
        self.memory.reset()

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,                               # (L,) or (B, L)
        tau_t: float = 0.0,
        delta_tau: float = 0.0,
        return_hidden: bool = False,
    ):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, L = input_ids.shape
        device = input_ids.device

        # Embed inputs through Qwen's frozen embedding table.
        try:
            embed = self.base.model.embed_tokens(input_ids)
        except AttributeError:
            embed = self.base.transformer.wte(input_ids)
        # Chronometric + PFC prefix
        chi_t = self.chrono(
            torch.tensor(tau_t, device=device, dtype=torch.float32),
            torch.tensor(delta_tau, device=device, dtype=torch.float32),
        ).squeeze(0)                                           # (d_chrono,)
        prefix = self.pfc(self.memory.k, self.memory.v, chi_t)  # (K_p, d_model)
        prefix = prefix.to(embed.dtype).unsqueeze(0).expand(B, -1, -1)

        # Concatenate prefix + token embeddings
        hidden = torch.cat([prefix, embed], dim=1)             # (B, K_p+L, d_model)

        # Build attention mask so the prefix is visible to all tokens
        # but the LM loss only counts the original token positions.
        attn_mask = torch.ones(hidden.shape[:2], device=device, dtype=torch.long)

        # Run through the base model via inputs_embeds
        out = self.base(
            inputs_embeds=hidden,
            attention_mask=attn_mask,
            output_hidden_states=return_hidden,
            return_dict=True,
        )
        # Last L logits correspond to the original tokens
        logits = out.logits[:, -L:, :]                         # (B, L, vocab)
        result = {"logits": logits.squeeze(0) if B == 1 else logits}
        if return_hidden:
            # Strip the prefix from each layer's hidden state
            result["hidden_states"] = [h[:, -L:, :] for h in out.hidden_states]
        # Write last-layer hidden states (excluding the prefix) into memory.
        last_h = out.hidden_states[-1][:, -L:, :] if out.hidden_states else None
        if last_h is None:
            # Re-run with output_hidden_states to grab last hidden state.
            with torch.no_grad():
                out2 = self.base(
                    inputs_embeds=hidden,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_h = out2.hidden_states[-1][:, -L:, :]
        with torch.no_grad():
            self.memory.write(last_h[0].float(), tau_t=tau_t)
        return result


def build_qwen_ipcn(cfg: Optional[QwenIPCNConfig] = None) -> QwenIPCN:
    cfg = cfg or QwenIPCNConfig()
    return QwenIPCN(cfg)


if __name__ == "__main__":
    # Quick local sanity print
    cfg = QwenIPCNConfig()
    print(f"Config: base={cfg.base_model_name}, slots={cfg.n_slots}, K_p={cfg.prefix_length}, LoRA rank={cfg.lora_rank}")
