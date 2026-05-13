"""Checkpoint save / load for IPCN.

Saves:
  - all model weights (state_dict)
  - memory bank dynamic state (k, v, q, ages, usage, conf, plast, conflict,
    tau_write, tau_use, chi_slot, z)
  - train_step counter
  - optimizer state
  - rng state (cpu only for v1)
  - cfg

Resume:
  - load_checkpoint() returns (model, opt, train_step, cfg).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from model.config import IPCNConfig
from model.ipcn import IPCN


def save_checkpoint(
    path: str,
    model: IPCN,
    opt: torch.optim.Optimizer,
    cfg: IPCNConfig,
    train_step: int,
    extra: dict | None = None,
):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "cfg": asdict(cfg),
        "train_step": train_step,
        "rng_state_cpu": torch.get_rng_state(),
        "extra": extra or {},
    }
    torch.save(state, str(p))


def load_checkpoint(
    path: str,
    map_location: str = "cpu",
) -> tuple[IPCN, dict, IPCNConfig, int]:
    """Returns (model, opt_state_dict, cfg, train_step). Caller rebuilds the
    optimizer fresh and loads opt_state if desired."""
    state = torch.load(str(path), map_location=map_location, weights_only=False)
    cfg = IPCNConfig(**state["cfg"])
    model = IPCN(cfg)
    model.load_state_dict(state["model_state"])
    if "rng_state_cpu" in state:
        torch.set_rng_state(state["rng_state_cpu"])
    return model, state["opt_state"], cfg, int(state["train_step"])


def save_meta(path: str, meta: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)
