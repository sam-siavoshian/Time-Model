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
from dataclasses import asdict, fields
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


def _reconstruct_cfg(saved_cfg: dict) -> IPCNConfig:
    """Build IPCNConfig from a saved dict, robust to schema evolution.

    - Unknown keys silently dropped (old checkpoint, new code where field was renamed).
    - Missing keys use IPCNConfig defaults.
    - Fields declared as Tuple get coerced back from list (asdict converts to list).
    """
    known_fields = {f.name: f for f in fields(IPCNConfig)}
    filtered = {}
    dropped = []
    for k, v in saved_cfg.items():
        if k not in known_fields:
            dropped.append(k)
            continue
        # Coerce list -> tuple for tuple-typed fields. We detect by inspecting
        # the field's default (if it's a tuple, we want a tuple).
        default = known_fields[k].default
        if isinstance(v, list) and isinstance(default, tuple):
            v = tuple(v)
        filtered[k] = v
    if dropped:
        print(f"checkpoint: dropping {len(dropped)} unknown cfg keys: {dropped}")
    missing = [k for k in known_fields if k not in filtered]
    if missing:
        print(f"checkpoint: {len(missing)} cfg keys missing, using defaults: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return IPCNConfig(**filtered)


def load_checkpoint(
    path: str,
    map_location: str = "cpu",
    strict: bool = True,
) -> tuple[IPCN, dict, IPCNConfig, int]:
    """Returns (model, opt_state_dict, cfg, train_step). Caller rebuilds the
    optimizer fresh and loads opt_state if desired.

    strict: if True (default), model.load_state_dict fails on missing/extra keys.
        Set False ONLY when knowingly loading an older checkpoint into a model
        with added modules (e.g. late_retrieval head wasn't in pre-bbeaf79 ckpts).
    """
    state = torch.load(str(path), map_location=map_location, weights_only=False)
    cfg = _reconstruct_cfg(state["cfg"])
    model = IPCN(cfg)
    missing_keys, unexpected_keys = model.load_state_dict(state["model_state"], strict=False)
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"checkpoint load: missing {len(missing_keys)} keys, "
            f"unexpected {len(unexpected_keys)} keys.\n"
            f"  missing first 3: {missing_keys[:3]}\n"
            f"  unexpected first 3: {unexpected_keys[:3]}\n"
            f"  pass strict=False to override."
        )
    if not strict and (missing_keys or unexpected_keys):
        print(f"checkpoint load: missing={len(missing_keys)}, unexpected={len(unexpected_keys)} "
              f"(strict=False; new modules at init, old modules dropped)")
    if "rng_state_cpu" in state:
        torch.set_rng_state(state["rng_state_cpu"])
    return model, state["opt_state"], cfg, int(state["train_step"])


def save_meta(path: str, meta: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)
