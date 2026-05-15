"""IPCN v2: cross-attention memory injection + differentiable writes + Identity-V.

Synthesis of four diagnostic reviews of v1-v4 (PAPER.md Section 21):
  - Prefix-prepend at layer 0 is rank-bounded (Petrov & Liang, NeurIPS 2024,
    arxiv 2310.19698) — proves it CANNOT redirect attention to memory tokens.
    Benchmark of 6 frozen-base memory architectures on Flan-T5 (arxiv
    2603.16413): prefix = 0.02% recall, cross-attention = 11.91% (600x).
  - `@torch.no_grad()` on _MemoryBankQwen.write made W_k_m and W_v_m never
    train. Memory bank was content-addressed via random Kaiming-init
    projections. This is fixed here by making writes differentiable and
    keeping a gradient-bearing snapshot across chunks.
  - Identity-V (Prometheus Mind, arxiv 2601.15324): when the fact value is
    a token, store lm_head.weight[token_id] as the slot value instead of a
    learned projection. Removes the encoder-decoder bottleneck and gets
    87.5% vs 0% on retrieval in their setup.
  - Cross-attention is injected at three Qwen layers (4, 12, 20) per
    LongMem (NeurIPS 2023, arxiv 2306.07174) and Memory Injections (arxiv
    2309.05605). Qwen specifically recalls facts via EARLY ATTENTION
    (arxiv 2509.08778), so we hit layer 4 hard.

API mirrors model/qwen_ipcn.py so the rest of the pipeline (train, recall
check) just swaps the import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QwenIPCNv2Config:
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Memory bank
    n_slots: int = 128
    d_memory: int = 0                                          # 0 = match base hidden_size

    # Cross-attention injection points: layers where memory gets fused in
    # via additive gated cross-attention. Qwen 2.5 1.5B has 28 layers
    # (0..27). Pick early/middle/late per LongMem + Qwen mechanistic-
    # interpretation (arxiv 2509.08778).
    inject_layers: Tuple[int, ...] = (4, 12, 20)

    # LoRA: keep narrow because cross-attn does the routing.
    lora_rank: int = 8
    lora_layers: Tuple[int, ...] = (4,)                        # just the layer Qwen uses for fact recall
    lora_targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    lora_lm_head: bool = True

    # If True, do NOT freeze the Qwen base. All ~1.5B params train via
    # AdamW. v9 escalation when frozen-base ran out of capacity to learn
    # memory routing. ~6 GB extra optimizer state; fits on GB10 130 GB.
    unfreeze_base: bool = False

    # Write step
    n_write_candidates: int = 4
    write_eta: float = 0.5                                     # EMA mix rate
    use_identity_v: bool = True                                # use lm_head row as slot value (Prometheus Mind)

    # Eligibility / consolidation (kept for downstream — not used in v5 alone)
    tau_cons: float = -1.0
    consolidation_frequency: int = 256
    kl_drift_threshold: float = 0.05
    eps_drop: float = 0.02

    # Misc
    chunk_length: int = 64
    lambda_u_decay: float = 0.99


class _LoRALinear(nn.Module):
    """Frozen base nn.Linear + rank-r additive adapter. Dtype-safe."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_l = x.to(self.lora_A.dtype)
        adapter = F.linear(F.linear(x_l, self.lora_A), self.lora_B) * self.scaling
        return base_out + adapter.to(base_out.dtype)


class _MemoryBankV2(nn.Module):
    """Differentiable-write memory bank with Identity-V option.

    Buffers store the "committed" memory state (non-differentiable, persists
    via state_dict). The forward graph reads from a gradient-bearing
    snapshot self._k_grad / self._v_grad that gets updated on every write
    and carried across chunks within a conversation.
    """

    def __init__(self, cfg: QwenIPCNv2Config, d_model: int, vocab_lookup: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        d_m = cfg.d_memory if cfg.d_memory > 0 else d_model
        self.d_model = d_model
        self.d_memory = d_m
        N_m = cfg.n_slots

        # Buffers (non-grad, persisted in state_dict).
        init_k = torch.randn(N_m, d_m)
        init_k = init_k / init_k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("k", init_k)
        self.register_buffer("v", torch.zeros(N_m, d_m))
        self.register_buffer("usage", torch.zeros(N_m))
        self.register_buffer("conf", torch.full((N_m,), 0.5))
        self.register_buffer("plast", torch.full((N_m,), 1.0))
        self.register_buffer("conflict", torch.zeros(N_m))
        self.register_buffer("tau_write", torch.zeros(N_m))
        self.register_buffer("tau_use", torch.zeros(N_m))

        # Differentiable snapshot. Reset every conversation; updated every write.
        self._k_grad: Optional[torch.Tensor] = None
        self._v_grad: Optional[torch.Tensor] = None

        # Learned write projections — NOW differentiable.
        self.W_k_m = nn.Linear(d_model, d_m, bias=False)
        self.W_v_m = nn.Linear(d_model, d_m, bias=False)

        # Identity-V uses the base model's lm_head as a value lookup
        # table; otherwise W_v_m projects hidden states to slot values.
        self.use_identity_v = cfg.use_identity_v
        self._vocab_lookup = vocab_lookup                      # optional handle to lm_head

    def reset(self):
        """Wipe slot state; called at conversation boundary."""
        with torch.no_grad():
            new_k = torch.randn(
                self.cfg.n_slots, self.d_memory,
                device=self.k.device, dtype=self.k.dtype,
            )
            new_k = new_k / new_k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            self.k.copy_(new_k)
            self.v.zero_()
            self.usage.zero_()
            self.conf.fill_(0.5)
            self.plast.fill_(1.0)
            self.conflict.zero_()
            self.tau_write.zero_()
            self.tau_use.zero_()
        self._k_grad = None
        self._v_grad = None

    def current_k(self) -> torch.Tensor:
        return self._k_grad if self._k_grad is not None else self.k

    def current_v(self) -> torch.Tensor:
        return self._v_grad if self._v_grad is not None else self.v

    def write(self, h: torch.Tensor, write_target_ids: Optional[torch.Tensor], tau_t: float):
        """Differentiable write.

        h: (L, d_model) hidden states of the current chunk, grad-bearing.
        write_target_ids: (L,) int. If use_identity_v=True, the slot value
            for a chosen position becomes lm_head.weight[target_id] so the
            memory directly encodes the answer token's output direction.
            (Prometheus Mind: 87.5% vs 0% gain.) If None, use W_v_m(h).

        Top-k_w candidates by L2 norm of h get written via argmax-cosine
        slot assignment + EMA mix. The result is stored in self._k_grad /
        self._v_grad (carries gradient) AND copied to self.k / self.v
        (non-grad buffer for persistence + state_dict).
        """
        L = h.shape[0]
        K_w = min(self.cfg.n_write_candidates, L)
        if K_w == 0:
            return
        with torch.no_grad():
            norms = h.float().norm(dim=-1)
            top_idx = torch.topk(norms, K_w).indices            # (K_w,)

        cand_h = h[top_idx]                                    # (K_w, d_model) grad
        k_hat = self.W_k_m(cand_h.to(self.W_k_m.weight.dtype))  # (K_w, d_m) grad

        if self.use_identity_v and self._vocab_lookup is not None and write_target_ids is not None:
            # Identity-V: slot value = lm_head row for the target token.
            tgt = write_target_ids[top_idx].clamp_min(0)        # (K_w,)
            # lm_head.weight shape: (vocab, d_model). Take rows, project to d_memory
            # if needed (here d_m = d_model so it's identity).
            lm_w = self._vocab_lookup.weight if hasattr(self._vocab_lookup, "weight") else self._vocab_lookup
            v_hat = lm_w[tgt].to(k_hat.dtype)                   # (K_w, d_model) detached from lm_head grad (lm_head is LoRA-wrapped if cfg.lora_lm_head)
            if v_hat.shape[-1] != self.d_memory:
                # If d_memory != d_model we'd need a projection. Assert match.
                raise RuntimeError(f"Identity-V requires d_memory == d_model, got {self.d_memory} vs {v_hat.shape[-1]}")
        else:
            v_hat = self.W_v_m(cand_h.to(self.W_v_m.weight.dtype))

        # Slot assignment: argmax cosine over current k state.
        cur_k = self.current_k()
        K_norm = F.normalize(cur_k.to(k_hat.dtype), dim=-1)
        k_hat_n = F.normalize(k_hat, dim=-1)
        sim = k_hat_n @ K_norm.t()                             # (K_w, N_m)
        best = sim.argmax(dim=-1)                              # (K_w,)
        beta = F.one_hot(best, num_classes=self.cfg.n_slots).to(k_hat.dtype)

        # Aggregate candidate writes into per-slot targets. used_mass tracks
        # whether a slot received any incoming write.
        used_mass = beta.sum(dim=0)                            # (N_m,)
        k_target = beta.t() @ k_hat                            # (N_m, d_m)
        v_target = beta.t() @ v_hat
        eta = self.cfg.write_eta * (used_mass > 0).to(k_hat.dtype)
        one_minus = (1.0 - eta).unsqueeze(-1)
        cur_k_typed = self.current_k().to(k_hat.dtype)
        cur_v_typed = self.current_v().to(v_hat.dtype)
        new_k = one_minus * cur_k_typed + eta.unsqueeze(-1) * k_target
        new_k = F.normalize(new_k, dim=-1)                     # always unit-norm keys
        new_v = one_minus * cur_v_typed + eta.unsqueeze(-1) * v_target

        # Carry across chunks (with grad).
        self._k_grad = new_k
        self._v_grad = new_v

        # Persist (no grad).
        with torch.no_grad():
            self.k.copy_(new_k.detach().to(self.k.dtype))
            self.v.copy_(new_v.detach().to(self.v.dtype))
            touched = (used_mass > 0).to(self.tau_write.dtype)
            self.tau_write.copy_(touched * tau_t + (1.0 - touched) * self.tau_write)
            self.usage.copy_(self.cfg.lambda_u_decay * self.usage + used_mass.to(self.usage.dtype))


class _CrossAttnMemory(nn.Module):
    """One injection point. Computes a memory-conditioned delta to add to
    the layer's hidden state, gated by a learned sigmoid.

      q = W_q(h)                       # (B, L, d)
      attn = softmax(q @ k_m.T / sqrt) # (B, L, N_m)
      delta = attn @ v_m               # (B, L, d)
      g = sigmoid(W_g(h))              # (B, L, 1)  (per-position gate)
      h' = h + g * W_o(delta)

    All projections trainable. The "g" gate lets the model learn to use
    memory ONLY at the recall-question positions.
    """

    def __init__(self, d_model: int, d_memory: int):
        super().__init__()
        self.d_model = d_model
        self.scale = d_memory ** -0.5
        self.W_q = nn.Linear(d_model, d_memory, bias=False)
        self.W_o = nn.Linear(d_memory, d_model, bias=False)
        self.W_g = nn.Linear(d_model, 1, bias=True)
        # Init gate near 0.5 so the cross-attn meaningfully contributes from
        # step 0. v5 used sigmoid(-3)=0.05 which made the memory pathway
        # invisible at init; combined with negative-control collapse, the
        # gate never grew. Sigmoid(0)=0.5 starts memory at half strength.
        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 0.0)

    def forward(
        self,
        h: torch.Tensor,                                       # (B, L, d_model)
        mem_k: torch.Tensor,                                   # (N_m, d_memory)
        mem_v: torch.Tensor,                                   # (N_m, d_memory)
    ) -> torch.Tensor:
        # Do the cross-attn in fp32 so gate gradients are stable; cast back
        # at the residual add. Mem K/V come in at h.dtype; coerce all to fp32.
        h_fp = h.float()
        q = self.W_q(h_fp)                                     # (B, L, d_memory) fp32
        mk = mem_k.float()
        mv = mem_v.float()
        attn = torch.softmax(q @ mk.t() * self.scale, dim=-1)
        delta = attn @ mv
        delta = self.W_o(delta)
        gate = torch.sigmoid(self.W_g(h_fp))
        return h + (gate * delta).to(h.dtype)


class _ShortcutHead(nn.Module):
    """Direct memory -> logit shortcut. Bypasses all 28 Qwen layers.

    At each token position, query the memory bank, attend to slots, fetch
    a weighted memory value vector. Project to vocab via the (LoRA-wrapped)
    lm_head and ADD to the base logits, scaled by a global learnable gate.

    Under Identity-V, memory.v[i] = lm_head.weight[target_token_i] so
    `shortcut @ lm_head.weight.T` directly produces a logit boost at the
    target token. The gate starts at sigmoid(2) = 0.88 so the shortcut
    dominates from step 0; the model can scale it down via gradient if
    memory ever points at the wrong slot.
    """

    def __init__(self, d_model: int, d_memory: int):
        super().__init__()
        self.scale = d_memory ** -0.5
        self.W_q = nn.Linear(d_model, d_memory, bias=False)
        self.gate_logit = nn.Parameter(torch.tensor(2.0))      # sigmoid(2)=0.88

    def forward(
        self,
        last_h: torch.Tensor,                                  # (B, L, d_model)
        mem_k: torch.Tensor,                                   # (N_m, d_memory)
        mem_v: torch.Tensor,                                   # (N_m, d_memory)
        lm_head: nn.Module,
    ) -> torch.Tensor:                                         # (B, L, vocab)
        h = last_h.float()
        q = self.W_q(h)                                        # (B, L, d_memory)
        mk = mem_k.float()
        mv = mem_v.float()
        attn = torch.softmax(q @ mk.t() * self.scale, dim=-1)  # (B, L, N_m)
        shortcut = attn @ mv                                    # (B, L, d_memory)
        # Run through lm_head (LoRA-wrapped) to get vocab-space logits.
        # lm_head is _LoRALinear: base.weight is bf16, lora_A/B are fp32.
        # The _LoRALinear forward handles internal dtype casting; we just
        # need to feed in the base weight's dtype.
        base_dtype = lm_head.base.weight.dtype if hasattr(lm_head, "base") else lm_head.weight.dtype
        shortcut_logits = lm_head(shortcut.to(base_dtype))
        gate = torch.sigmoid(self.gate_logit)
        return gate * shortcut_logits.float()


class QwenIPCNv2(nn.Module):
    """Cross-attention-injected IPCN around frozen Qwen 2.5.

    The wrapper attaches forward HOOKS on Qwen's decoder layers at the
    injection indices. Each hook post-processes the layer's output by
    adding a memory-conditioned delta. Hooks can be removed at any time.
    """

    def __init__(self, cfg: QwenIPCNv2Config):
        super().__init__()
        self.cfg = cfg

        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
        self.base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        if not cfg.unfreeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)
        else:
            print("[v9] base UNFROZEN. all ~1.5B params will receive gradient.")
            for p in self.base.parameters():
                p.requires_grad_(True)
        self.d_model = self.base.config.hidden_size
        if cfg.d_memory == 0:
            cfg.d_memory = self.d_model

        # Memory bank with optional Identity-V via lm_head weight rows.
        # NB: lm_head will be LoRA-wrapped below, so we pass the inner
        # base.lm_head BEFORE wrapping. After wrapping, the weight reference
        # still points to the original tied embedding.
        vocab_lookup = self.base.lm_head if hasattr(self.base, "lm_head") else None
        self.memory = _MemoryBankV2(cfg, self.d_model, vocab_lookup=vocab_lookup)

        # Cross-attention modules (one per injection layer).
        self.cross_attn = nn.ModuleDict({
            str(li): _CrossAttnMemory(self.d_model, cfg.d_memory)
            for li in cfg.inject_layers
        })

        # Apply LoRA to selected layers + lm_head.
        self._apply_lora()

        # Direct memory -> logit shortcut head (bypasses all 28 layers).
        self.shortcut = _ShortcutHead(self.d_model, cfg.d_memory)

        # Register forward hooks on the chosen decoder layers.
        self._hook_handles = []
        self._register_injection_hooks()

    def _apply_lora(self):
        cfg = self.cfg
        n_adapted = 0
        try:
            layers = self.base.model.layers
        except AttributeError:
            layers = self.base.transformer.h
        for li in cfg.lora_layers:
            block = layers[li]
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

    def _register_injection_hooks(self):
        try:
            layers = self.base.model.layers
        except AttributeError:
            layers = self.base.transformer.h
        for li in self.cfg.inject_layers:
            block = layers[li]
            xattn = self.cross_attn[str(li)]

            def make_hook(xattn_mod):
                def hook(module, args, output):
                    # Qwen's decoder layer returns a tuple where output[0]
                    # is the hidden state. We inject into output[0].
                    h = output[0] if isinstance(output, tuple) else output
                    mem_k = self.memory.current_k().to(h.dtype)
                    mem_v = self.memory.current_v().to(h.dtype)
                    h_new = xattn_mod(h, mem_k, mem_v)
                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new
                return hook

            handle = block.register_forward_hook(make_hook(xattn))
            self._hook_handles.append(handle)

    def reset_memory(self):
        self.memory.reset()

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        tau_t: float = 0.0,
        delta_tau: float = 0.0,
        write_target_ids: Optional[torch.Tensor] = None,
    ):
        """Run Qwen with cross-attention memory injection at the chosen
        layers, then update the memory bank from the last hidden state.

        write_target_ids: optional (L,) target token ids. If Identity-V is
            on, the write step picks the lm_head row for each target as
            the slot value. For training, pass the LABELS for the chunk so
            memory stores "the right answer direction" at the answer chunk.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, L = input_ids.shape
        device = input_ids.device

        out = self.base(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = out.logits                                    # (B, L, vocab)
        last_h = out.hidden_states[-1]                         # (B, L, d_model)

        # Direct memory -> logit shortcut. Adds a memory-attention-weighted
        # lm_head pass directly to the base's logits. Required because the
        # cross-attn injection at layers 4/12/20 gets washed out in the
        # 23-layer residual stream before reaching lm_head. v1-v7 all hit
        # delta=0 because of this dilution. The shortcut head is the
        # decisive fix: memory value -> lm_head -> output, no detour.
        mem_k_now = self.memory.current_k()
        mem_v_now = self.memory.current_v()
        shortcut_logits = self.shortcut(last_h, mem_k_now, mem_v_now, self.base.lm_head)
        logits = logits.float() + shortcut_logits

        # Write into memory using last hidden state (with grad).
        write_h = last_h[0].to(self.W_k_m_dtype())             # (L, d_model)
        wt = write_target_ids if write_target_ids is None else write_target_ids
        if wt is not None and wt.dim() > 1:
            wt = wt[0]
        self.memory.write(write_h, wt, tau_t=tau_t)

        return {"logits": logits.squeeze(0) if B == 1 else logits}

    def W_k_m_dtype(self):
        return self.memory.W_k_m.weight.dtype


def build_qwen_ipcn_v2(cfg: Optional[QwenIPCNv2Config] = None) -> QwenIPCNv2:
    cfg = cfg or QwenIPCNv2Config()
    return QwenIPCNv2(cfg)
