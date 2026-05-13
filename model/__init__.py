"""IPCN model package.

Modules:
  config         locked hyperparameters (single source of truth)
  chronometric   deterministic chi_t encoder
  memory         256-slot episodic memory bank + evolution
  adapters       LoRA wrappers
  pfc            prefix-forming controller
  injection      three injection routes (prepend / broadcast / LayerNorm)
  core           decoder-only Transformer (8 layers, width 512)
  losses         9-term training objective
  consolidation  teacher-student distillation into LoRA with rollback
  ipcn           top-level IPCN model wrapper
"""

from model.config import IPCNConfig

__all__ = ["IPCNConfig"]
