"""IPCN Track C: Qwen wrapped to EXPERIENCE TIME.

Track A (102M from scratch) and Track B (Qwen + memory routing) both
failed at memory-value retrieval -- the "what specific token" question.
The paper's actual claim is time experience, not value recall.

This module reframes the architecture around the FOUR operational
properties of time (PAPER.md §1):
  1. Causal ordering (A before B can influence future, not reverse)
  2. Duration measurement (Delta-tau is detectable)
  3. Multi-scale phase (same Delta-tau means different things at
     different points in a cycle)
  4. Persistence under no-input (tau advances even with no tokens)

The four behavioral gaps to close (PAPER.md §7):
  - No clock: model has no tau, only token position.
  - No silent-gap awareness: gaps between inputs invisible.
  - No self-rate awareness: model does not know its own throughput.
  - No behavioral pressure: deadlines do not change behavior.

Architecture:
  - Qwen 2.5 3B-Instruct (or 7B), frozen + LoRA (all 28 layers + head).
  - Chronometric encoder: 13-scale sinusoidal + log1p(tau) = 27-dim chi.
  - Layer-wise chrono injection: at EVERY decoder layer, an additive
    bias derived from chi gets summed into the layer's hidden state.
    The bias is per-position (same across all tokens of a chunk because
    tau is per-chunk). Gate init = sigmoid(0) = 0.5 so the chrono signal
    is half-strength from step 0.
  - Memory bank kept for tau_write timestamps so we can also test
    age-discount retrieval, but memory-to-output routing is no longer
    the headline claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QwenTimeConfig:
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    timescales: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536)
    inject_layers: Tuple[int, ...] = ()                        # () = inject at EVERY layer
    lora_rank: int = 8
    lora_layers: Tuple[int, ...] = ()                          # () = LoRA on ALL layers
    lora_targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    lora_lm_head: bool = True
    unfreeze_base: bool = False
    chunk_length: int = 256
    # Architectural ablation: "film" (default v15) or "additive".
    # "film" = h + alpha * (gamma * h + beta)        -- DiT AdaLN-Zero pattern
    # "additive" = h + alpha * (W_chi @ chi)         -- pure additive residual
    injection_type: str = "film"
    # When injection_type == "additive" and this is > 0, initialize
    # to_beta.bias to this constant so beta(chi) at step 0 is non-zero,
    # which gives d_out/d_alpha = beta != 0 and lets the additive variant
    # escape the AdaLN-Zero gradient trap. Used to test whether the W9
    # reviewer concern ("additive ablation tests only one init") is
    # resolved by a sane non-zero beta init. Default 0.0 = original AdaLN-Zero.
    additive_beta_init: float = 0.0


class _Chronometric(nn.Module):
    """Multi-scale sinusoidal + log1p(tau). Output dim = 2*N_scales + 1."""

    def __init__(self, timescales: Tuple[int, ...]):
        super().__init__()
        self.register_buffer("scales", torch.tensor(timescales, dtype=torch.float32))

    @property
    def out_dim(self) -> int:
        return 2 * self.scales.numel() + 1

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.float()
        if tau.dim() == 0:
            tau = tau.unsqueeze(0)
        x = tau.unsqueeze(-1) / self.scales
        return torch.cat([torch.sin(x), torch.cos(x), torch.log1p(tau.unsqueeze(-1))], dim=-1)


class _LoRALinear(nn.Module):
    """Dtype-safe LoRA wrapper."""

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


class _ChronoInjector(nn.Module):
    """AdaLN-Zero style FiLM injection per DiT (arxiv 2212.09748).

    h' = h + alpha * (gamma * h + beta)

    gamma, beta projected from chi_t. alpha is a per-channel learned
    scalar initialized to ZERO so the entire injection is identity at
    step 0 (no random noise added to a 28-layer residual stream). Once
    LoRA + the projectors learn useful directions, alpha rises and the
    chrono signal contributes.

    This replaces the prior additive-bias-with-gate-0.5 design which
    pumped untrained noise into every layer from step 0 -- the same
    failure surface that killed Track B v1-v4.
    """

    def __init__(self, d_model: int, d_chrono: int, injection_type: str = "film",
                 additive_beta_init: float = 0.0):
        super().__init__()
        self.injection_type = injection_type
        self.to_gamma = nn.Linear(d_chrono, d_model, bias=True)
        self.to_beta = nn.Linear(d_chrono, d_model, bias=True)
        # Per-channel residual gate, ZERO-init so injection is identity at step 0.
        self.alpha = nn.Parameter(torch.zeros(d_model))
        # CRITICAL init pattern from DiT (arxiv 2212.09748):
        #   alpha = 0     (residual gate)
        #   gamma = 1     (identity scale, BIAS not weight)
        #   beta  = 0     (shift)
        # At init: out = h + 0 * (1*h + 0) = h. PURE IDENTITY.
        # But d_out/d_alpha = gamma*h + beta = 1*h + 0 = h (NONZERO).
        # So alpha receives nonzero gradient and can grow.
        # v10 incorrectly zeroed gamma too -> d_out/d_alpha = 0 -> alpha
        # never moved off zero -> chrono signal never reached output ->
        # T4=0.0 negative-control failure across all tests.
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.constant_(self.to_gamma.bias, 1.0)             # gamma = 1 = identity scale
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)
        # W9 reviewer test: for the additive variant, allow a small non-zero
        # beta init so d_out/d_alpha = beta != 0 at step 0 and the additive
        # variant has a chance to train. Default 0.0 = original (which traps).
        if injection_type == "additive" and additive_beta_init != 0.0:
            nn.init.constant_(self.to_beta.bias, additive_beta_init)

    def forward(self, h: torch.Tensor, chi_t: torch.Tensor) -> torch.Tensor:
        chi_f = chi_t.float()
        gamma = self.to_gamma(chi_f)
        beta = self.to_beta(chi_f)
        h_f = h.float()
        if self.injection_type == "additive":
            # Pure additive residual ablation: ignore h-dependent scaling.
            # out = h + alpha * beta(chi). Comparable to GazeQwen-style.
            out = h_f + self.alpha[None, None, :] * beta[None, None, :]
        else:
            # FiLM (default v15): scale + shift h, gated by alpha.
            modulated = gamma[None, None, :] * h_f + beta[None, None, :]
            out = h_f + self.alpha[None, None, :] * modulated
        return out.to(h.dtype)


class QwenTime(nn.Module):
    """Frozen Qwen + per-layer chronometric injection + optional unfreeze."""

    def __init__(self, cfg: QwenTimeConfig):
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
        self.d_model = self.base.config.hidden_size

        self.chrono = _Chronometric(cfg.timescales)
        # Stack of chi vectors (one per active forward) so re-entrant calls
        # don't clobber each other. Hooks read the top of the stack.
        self._chi_stack: list = []

        # Find decoder layers.
        try:
            self._layers = self.base.model.layers
        except AttributeError:
            self._layers = self.base.transformer.h
        n_layers = len(self._layers)
        # Skip the LAST layer: RMSNorm + lm_head at the end would attenuate
        # the injected bias to near-noise. Inject up to penultimate.
        inject_layers = cfg.inject_layers if cfg.inject_layers else tuple(range(n_layers - 1))
        # One injector per layer (injection_type set per ablation).
        injection_type = getattr(cfg, "injection_type", "film")
        additive_beta_init = float(getattr(cfg, "additive_beta_init", 0.0))
        self.chrono_injectors = nn.ModuleDict({
            str(li): _ChronoInjector(self.d_model, self.chrono.out_dim,
                                     injection_type=injection_type,
                                     additive_beta_init=additive_beta_init)
            for li in inject_layers
        })
        self._inject_layers = inject_layers

        # Apply LoRA.
        self._apply_lora()

        # Register hooks.
        self._hook_handles = []
        self._register_chrono_hooks()

    def _apply_lora(self):
        cfg = self.cfg
        n_adapted = 0
        layers = self._layers
        n_layers = len(layers)
        lora_layers = cfg.lora_layers if cfg.lora_layers else tuple(range(n_layers))
        for li in lora_layers:
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
            # Prevent HF from re-tying lm_head.weight back to embed_tokens
            # on future state_dict loads (Qwen 2.5 ties by default).
            try:
                self.base.config.tie_word_embeddings = False
            except Exception:
                pass
            n_adapted += 1
        self._n_lora_modules = n_adapted

    def _register_chrono_hooks(self):
        for li in self._inject_layers:
            block = self._layers[li]
            injector = self.chrono_injectors[str(li)]

            def make_hook(inj):
                def hook(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if not self._chi_stack:
                        return output
                    h_new = inj(h, self._chi_stack[-1])
                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new
                return hook

            handle = block.register_forward_hook(make_hook(injector))
            self._hook_handles.append(handle)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        tau_t: float = 0.0,
    ):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        device = input_ids.device
        B = input_ids.shape[0]

        # Compute chi for this chunk's tau_t. Push onto stack so re-entrant
        # forwards (e.g. nested no_grad eval calls) don't clobber.
        chi = self.chrono(torch.tensor(tau_t, device=device, dtype=torch.float32))
        chi_t = chi.squeeze(0) if chi.dim() > 1 else chi
        self._chi_stack.append(chi_t)
        try:
            out = self.base(input_ids=input_ids, return_dict=True)
        finally:
            self._chi_stack.pop()

        logits = out.logits.squeeze(0) if B == 1 else out.logits
        return {"logits": logits}


def build_qwen_time(cfg: Optional[QwenTimeConfig] = None) -> QwenTime:
    cfg = cfg or QwenTimeConfig()
    return QwenTime(cfg)
