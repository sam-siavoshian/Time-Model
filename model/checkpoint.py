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


def _collect_opt_param_names(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
) -> list[list[str]]:
    """For each optimizer param group, return the parameter NAMES (per
    model.named_parameters()) in the order opt sees them.

    Used so a future load can map old optimizer state to new parameters
    by name when phase transitions change the trainable set.
    """
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    out = []
    for group_idx, group in enumerate(opt.param_groups):
        per_group = []
        for in_idx, p in enumerate(group["params"]):
            per_group.append(id_to_name.get(id(p), f"<unnamed_g{group_idx}_p{in_idx}>"))
        out.append(per_group)
    return out


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
        "opt_param_names_per_group": _collect_opt_param_names(model, opt),
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

    The returned opt_state_dict has been enriched with a top-level
    "param_names_per_group" key (when the checkpoint was saved with the
    current save_checkpoint). Callers that want robust restore across
    phase transitions should pass it to restore_opt_state_by_name
    rather than calling opt.load_state_dict directly -- a direct load
    raises ValueError as soon as the trainable param set changes
    (Phase 0 -> Phase 1, Phase 2 -> Phase 3, etc.), and most callers
    currently swallow that error silently and start with fresh
    momentum.

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
    opt_state = state["opt_state"]
    # Carry the saved param-name layout into the returned opt_state under
    # a known key so a caller can route it to restore_opt_state_by_name.
    # Older checkpoints without this key fall back to legacy behavior.
    if "opt_param_names_per_group" in state:
        opt_state = dict(opt_state)
        opt_state["__param_names_per_group__"] = state["opt_param_names_per_group"]
    return model, opt_state, cfg, int(state["train_step"])


def restore_opt_state_by_name(
    new_opt: torch.optim.Optimizer,
    new_model: torch.nn.Module,
    saved_opt_state: dict,
) -> dict:
    """Reload optimizer state across phase transitions by parameter name.

    Standard torch.optim.Optimizer.load_state_dict matches saved state to
    current params by POSITION within param_groups. When phase transitions
    change the trainable param set (e.g. Phase 0 -> Phase 1 narrows from
    all-params to PFC+LoRA only), positions no longer line up and the
    load raises ValueError. Callers historically caught that and silently
    fell back to fresh momentum, throwing away any AdamW state for params
    that ARE still trainable in the new phase.

    This helper does a name-based match instead:
      - For each param in the new optimizer, find its name in the model.
      - Look up that name in saved_opt_state["__param_names_per_group__"].
      - If found AND the saved state tensor shape matches: copy state.
      - Otherwise: leave that param's state empty (fresh init).

    Returns a summary dict:
      {n_matched: int, n_missing: int, n_shape_mismatch: int,
       n_new_params: int, used_name_map: bool}
    """
    saved_names_per_group = saved_opt_state.get("__param_names_per_group__")
    used_name_map = saved_names_per_group is not None

    # Map each new param to its model name.
    id_to_name = {id(p): n for n, p in new_model.named_parameters()}
    new_params_flat: list[tuple[str, torch.nn.Parameter]] = []
    for group in new_opt.param_groups:
        for p in group["params"]:
            new_params_flat.append((id_to_name.get(id(p), "<unnamed>"), p))

    # Walk the OLD opt state by position; build name -> state map (when
    # we have names) or position -> state map (when we don't).
    old_state = saved_opt_state.get("state", {})
    if used_name_map:
        old_flat_names: list[str] = []
        for group_names in saved_names_per_group:
            old_flat_names.extend(group_names)
        name_to_state: dict[str, dict] = {}
        for old_idx, name in enumerate(old_flat_names):
            entry = old_state.get(old_idx)
            if entry is not None:
                # If duplicate names appear (shared param), prefer earliest.
                name_to_state.setdefault(name, entry)
    else:
        name_to_state = {}

    # Build a state-dict-compatible dict matching the NEW opt's layout.
    new_state_dict_template = new_opt.state_dict()
    new_state = {}
    n_matched = 0
    n_missing = 0
    n_shape_mismatch = 0
    for new_idx, (name, param) in enumerate(new_params_flat):
        old_entry = name_to_state.get(name) if used_name_map else None
        if old_entry is None:
            n_missing += 1
            continue
        # Verify shape compat. AdamW state has exp_avg, exp_avg_sq, step.
        compat = True
        for tname in ("exp_avg", "exp_avg_sq"):
            t = old_entry.get(tname)
            if t is None:
                compat = False
                break
            if tuple(t.shape) != tuple(param.shape):
                compat = False
                break
        if not compat:
            n_shape_mismatch += 1
            continue
        # Defensive copy so the saved dict is not mutated.
        new_state[new_idx] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in old_entry.items()}
        n_matched += 1
    new_state_dict_template["state"] = new_state
    new_opt.load_state_dict(new_state_dict_template)
    return {
        "n_matched": n_matched,
        "n_missing": n_missing,
        "n_shape_mismatch": n_shape_mismatch,
        "n_new_params": len(new_params_flat),
        "used_name_map": used_name_map,
    }


def save_meta(path: str, meta: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)
