"""QwenTime: frozen Qwen with trainable chronometric injection.

The v15 default architecture uses Qwen 2.5 3B-Instruct, 15 fixed
timescales, a 31-dimensional chronometric encoder, AdaLN-Zero style FiLM
injection on every decoder layer except the final layer, and LoRA rank 8.

The implemented chronometric formula is intentionally:

    chi(tau) = [sin(tau / T_k), cos(tau / T_k), log1p(tau)]

for each timescale T_k. This code does not multiply by 2*pi; T_k is the
scale divisor in radians, so the sinusoid's mathematical period is
2*pi*T_k.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


V15_BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
V15_TIMESCALES: Tuple[int, ...] = (
    2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 86400, 604800
)
V15_CHUNK_LENGTH = 512
V15_LORA_RANK = 8
V15_INJECTION_TYPE = "film"
V15_DATA_MIX: Tuple[float, float, float] = (0.40, 0.30, 0.30)
V15_DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
QWEN_TIME_CONFIG_VERSION = 1
CHRONO_FORMULA = "sin(tau / T), cos(tau / T), log1p(tau)"


@dataclass
class QwenTimeConfig:
    base_model_name: str = V15_BASE_MODEL_NAME
    timescales: Tuple[int, ...] = V15_TIMESCALES
    inject_layers: Tuple[int, ...] = ()                        # () = v15: all decoder layers except final
    lora_rank: int = V15_LORA_RANK
    lora_layers: Tuple[int, ...] = ()                          # () = LoRA on ALL layers
    lora_targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    lora_lm_head: bool = True
    unfreeze_base: bool = False
    chunk_length: int = V15_CHUNK_LENGTH
    # Architectural ablation: "film" (default v15) or "additive".
    # "film" = h + alpha * (gamma * h + beta)        -- DiT AdaLN-Zero pattern
    # "additive" = h + alpha * (W_chi @ chi)         -- pure additive residual
    injection_type: str = V15_INJECTION_TYPE
    # When injection_type == "additive" and this is > 0, initialize
    # to_beta.bias to this constant so beta(chi) at step 0 is non-zero,
    # which gives d_out/d_alpha = beta != 0 and lets the additive variant
    # escape the AdaLN-Zero gradient trap. Used to test whether the W9
    # reviewer concern ("additive ablation tests only one init") is
    # resolved by a sane non-zero beta init. Default 0.0 = original AdaLN-Zero.
    additive_beta_init: float = 0.0
    # W9 reviewer fix: IA3 (Liu et al. 2022) PEFT baseline. When True,
    # _apply_lora is skipped and _apply_ia3 wraps k_proj, v_proj, and
    # the FFN intermediate (up_proj) with multiplicative scaling
    # vectors initialized to one. Trainable params per layer: dim(k) +
    # dim(v) + dim(ffn). For Qwen 2.5 3B with 36 layers, head_dim=128,
    # ffn_dim=11008: 36 * (128 + 128 + 11008) ~ 405K params total.
    use_ia3: bool = False


def qwen_time_config_dict(cfg: QwenTimeConfig) -> dict:
    """Primitive, checkpoint-safe representation of QwenTimeConfig."""
    data = asdict(cfg)
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in data.items()
    }


def qwen_time_checkpoint_metadata(
    cfg: QwenTimeConfig,
    resolved_inject_layers: Optional[Tuple[int, ...]] = None,
) -> dict:
    metadata = {
        "qwen_time_config_version": QWEN_TIME_CONFIG_VERSION,
        "chrono_formula": CHRONO_FORMULA,
        "cfg": qwen_time_config_dict(cfg),
    }
    if resolved_inject_layers is not None:
        metadata["resolved_inject_layers"] = list(resolved_inject_layers)
    return metadata


def resolve_inject_layers(cfg: QwenTimeConfig, n_layers: int) -> Tuple[int, ...]:
    """Resolve the v15 empty-layer sentinel to all layers except final."""
    if cfg.inject_layers:
        return tuple(int(x) for x in cfg.inject_layers)
    return tuple(range(max(n_layers - 1, 0)))


def config_mismatch_report(expected: dict, observed: dict) -> list[str]:
    """Return human-readable checkpoint config mismatches.

    The loader uses this for checkpoints that include config metadata.
    Historical checkpoints may only include a partial ``cfg`` dict, so the
    comparison is intentionally limited to keys that are present in the
    observed checkpoint.
    """
    mismatches: list[str] = []
    expected_cfg = expected.get("cfg", expected)
    observed_cfg = observed.get("cfg", observed)
    for key, observed_value in observed_cfg.items():
        if key not in expected_cfg:
            continue
        expected_value = expected_cfg[key]
        if isinstance(expected_value, tuple):
            expected_value = list(expected_value)
        if isinstance(observed_value, tuple):
            observed_value = list(observed_value)
        if expected_value != observed_value:
            mismatches.append(f"{key}: checkpoint={observed_value!r} current={expected_value!r}")

    expected_layers = expected.get("resolved_inject_layers")
    observed_layers = observed.get("resolved_inject_layers")
    if observed_layers is not None and expected_layers is not None and observed_layers != expected_layers:
        mismatches.append(
            f"resolved_inject_layers: checkpoint={observed_layers!r} current={expected_layers!r}"
        )
    return mismatches


class _Chronometric(nn.Module):
    """Multi-scale sin(tau/T), cos(tau/T) + log1p(tau).

    Output dim = 2*N_scales + 1. There is no 2*pi multiplier in this
    implementation; T is a divisor in radians, so period is 2*pi*T.
    """

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


class _IA3Linear(nn.Module):
    """IA3 (Liu et al. 2022) multiplicative-scaling wrapper.

    Wraps a frozen nn.Linear with a learnable per-output-feature scale
    vector initialized to one (identity at init). Used as a methodological-
    sibling PEFT baseline to LoRA (W9 reviewer fix).
    """

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        # Initialize to ONE so IA3 starts as identity transformation.
        # (Contrast LoRA which inits to zero / no effect.)
        self.ia3_l = nn.Parameter(torch.ones(base.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        return out * self.ia3_l.to(out.dtype)


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
        # v15 skips the LAST layer: RMSNorm + lm_head at the end would attenuate
        # the injected bias to near-noise. Inject up to penultimate by default.
        inject_layers = resolve_inject_layers(cfg, n_layers)
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

        # Apply LoRA (default) or IA3 (W9 baseline).
        if cfg.use_ia3:
            self._apply_ia3()
        else:
            self._apply_lora()

        # Register hooks.
        self._hook_handles = []
        self._register_chrono_hooks()

    def checkpoint_metadata(self) -> dict:
        return qwen_time_checkpoint_metadata(self.cfg, tuple(self._inject_layers))

    def _apply_ia3(self):
        """W9 fix: IA3 multiplicative-scaling PEFT baseline.
        Wraps k_proj, v_proj of every attention block, plus mlp.up_proj
        (the FFN intermediate gate) with _IA3Linear scaling vectors
        initialized to one (identity at init)."""
        cfg = self.cfg
        n_adapted = 0
        for li, block in enumerate(self._layers):
            attn = getattr(block, "self_attn", None) or getattr(block, "attn", None)
            if attn is not None:
                for proj_name in ("k_proj", "v_proj"):
                    proj = getattr(attn, proj_name, None)
                    if isinstance(proj, nn.Linear):
                        setattr(attn, proj_name, _IA3Linear(proj))
                        n_adapted += 1
            mlp = getattr(block, "mlp", None)
            if mlp is not None:
                up = getattr(mlp, "up_proj", None)
                if isinstance(up, nn.Linear):
                    mlp.up_proj = _IA3Linear(up)
                    n_adapted += 1
        self._n_ia3_modules = n_adapted
        self._n_lora_modules = 0  # for compatibility with trainer print

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
