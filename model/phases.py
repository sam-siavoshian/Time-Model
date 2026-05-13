"""Phase scheduler.

SPEC.tex training schedule:

  Phase 0 (sanity, 50-100k steps):
    - Memory + prefix + evolution ENABLED
    - LoRA adapters FROZEN (Omega frozen)
    - Goal: verify prefix affects layer 0, A3 > A1 on ambiguity

  Phase 1 (consolidation warmup, 50k steps):
    - PFC LoRA adapters ENABLED
    - Core-layer LoRA adapters still FROZEN
    - Test: removing slot after consolidation produces < 2% perf drop

  Phase 2 (early-core consolidation, 50k steps):
    - Core layers 1-2 LoRA adapters ENABLED (PFC already on)
    - Base weights still frozen

  Phase 3 (mixed LM, 100k steps):
    - All adapters as in Phase 2
    - Mix synthetic + real text
    - Track perplexity drift, fail if > 5% above baseline

Each phase has:
  - which LoRA modules accept gradients (toggle via .set_adapter_enabled())
  - which loss terms are active (toggle by zeroing the weight)
  - which data caches to draw from
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch.nn as nn

from model.adapters import LoRALinear
from model.config import IPCNConfig
from model.ipcn import IPCN


class Phase(Enum):
    PHASE_0_SANITY = 0
    PHASE_1_PFC_CONSOLIDATION = 1
    PHASE_2_EARLY_CORE_CONSOLIDATION = 2
    PHASE_3_MIXED_LM = 3


@dataclass
class PhaseConfig:
    name: str
    pfc_adapters_active: bool
    core_layer0_adapter_active: bool
    core_layer1_adapter_active: bool
    core_layer2_adapter_active: bool
    consolidation_loss_weight: float
    data_caches: list[str]                                   # cache prefixes to draw from
    max_steps: int


def get_phase_config(phase: Phase, cfg: IPCNConfig) -> PhaseConfig:
    if phase == Phase.PHASE_0_SANITY:
        return PhaseConfig(
            name="phase0_sanity",
            pfc_adapters_active=False,
            core_layer0_adapter_active=False,
            core_layer1_adapter_active=False,
            core_layer2_adapter_active=False,
            consolidation_loss_weight=0.0,
            data_caches=[
                "data/tokenized/latent_world/train_1k",
                "data/tokenized/latent_world/train_2k",
                "data/tokenized/ambiguity/train",
            ],
            max_steps=cfg.phase0_steps,
        )
    if phase == Phase.PHASE_1_PFC_CONSOLIDATION:
        return PhaseConfig(
            name="phase1_pfc_consolidation",
            pfc_adapters_active=True,
            core_layer0_adapter_active=False,
            core_layer1_adapter_active=False,
            core_layer2_adapter_active=False,
            consolidation_loss_weight=cfg.w_cons,
            data_caches=[
                "data/tokenized/latent_world/train_1k",
                "data/tokenized/latent_world/train_2k",
                "data/tokenized/ambiguity/train",
                "data/tokenized/consolidation/ladder_train",
            ],
            max_steps=cfg.phase1_steps,
        )
    if phase == Phase.PHASE_2_EARLY_CORE_CONSOLIDATION:
        return PhaseConfig(
            name="phase2_early_core",
            pfc_adapters_active=True,
            core_layer0_adapter_active=True,
            core_layer1_adapter_active=True,
            core_layer2_adapter_active=True,
            consolidation_loss_weight=cfg.w_cons,
            data_caches=[
                "data/tokenized/latent_world/train_1k",
                "data/tokenized/latent_world/train_2k",
                "data/tokenized/latent_world/train_4k",
                "data/tokenized/ambiguity/train",
                "data/tokenized/consolidation/ladder_train",
            ],
            max_steps=cfg.phase2_steps,
        )
    if phase == Phase.PHASE_3_MIXED_LM:
        return PhaseConfig(
            name="phase3_mixed_lm",
            pfc_adapters_active=True,
            core_layer0_adapter_active=True,
            core_layer1_adapter_active=True,
            core_layer2_adapter_active=True,
            consolidation_loss_weight=cfg.w_cons,
            data_caches=[
                "data/tokenized/latent_world/train_4k",
                "data/tokenized/latent_world/train_8k",
                "data/tokenized/ambiguity/train",
                "data/tokenized/consolidation/ladder_train",
                "data/tokenized/real_text/gutenberg",
            ],
            max_steps=cfg.phase3_steps,
        )
    raise ValueError(f"unknown phase {phase}")


def apply_phase(model: IPCN, pc: PhaseConfig):
    """Toggle LoRA adapters per phase config."""
    # Walk the model graph; identify which LoRALinear belongs where via module name.
    for full_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        if module.rank == 0:
            continue
        if "pfc" in full_name:
            module.set_adapter_enabled(pc.pfc_adapters_active)
            for p in module.adapter_parameters:
                p.requires_grad = pc.pfc_adapters_active
        elif full_name.startswith("core.blocks.0") or ".blocks.0." in full_name:
            module.set_adapter_enabled(pc.core_layer0_adapter_active)
            for p in module.adapter_parameters:
                p.requires_grad = pc.core_layer0_adapter_active
        elif full_name.startswith("core.blocks.1") or ".blocks.1." in full_name:
            module.set_adapter_enabled(pc.core_layer1_adapter_active)
            for p in module.adapter_parameters:
                p.requires_grad = pc.core_layer1_adapter_active
        elif full_name.startswith("core.blocks.2") or ".blocks.2." in full_name:
            module.set_adapter_enabled(pc.core_layer2_adapter_active)
            for p in module.adapter_parameters:
                p.requires_grad = pc.core_layer2_adapter_active

    # Base weights frozen at all phases per spec ("base network weights theta
    # are not updated during ordinary inference"). For training we DO update
    # base weights in Phase 0 because the model is from scratch and needs to
    # learn language at all. After Phase 0 we freeze base and only train
    # adapters.
    if pc.pfc_adapters_active or pc.core_layer0_adapter_active:
        # Phase 1+: freeze base weights
        for name, p in model.named_parameters():
            is_adapter = "lora_A" in name or "lora_B" in name
            if not is_adapter:
                p.requires_grad = False


def trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
