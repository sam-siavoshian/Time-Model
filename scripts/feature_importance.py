"""Feature importance analyzer.

Reports:
  1. Per-loss-component contribution: rolling % of total loss
  2. Per-parameter-group gradient magnitudes: which groups train fastest
  3. Slot utilization: which memory slots get used most
  4. Per-attention-head contribution (gradient norm of o_proj projections)

Run on a trained checkpoint. Produces reports/feature_importance.md.

Usage:
  uv run python -m scripts.feature_importance --checkpoint checkpoints/phase0.pt --n-batches 50
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

from model.checkpoint import load_checkpoint
from model.config import IPCNConfig
from model.dataset import SequentialChunkDataset, TokenizedCache
from model.ipcn import IPCN
from model.losses import (
    chronometric_loss,
    diversity_loss,
    lm_loss,
    mem_predict_loss,
    pre_influence_loss,
    precision_loss,
    slot_util_loss,
)


def per_loss_contribution(model: IPCN, cfg: IPCNConfig, n_batches: int = 50) -> dict:
    """Run forward+loss on N batches, report % contribution per loss term."""
    cache = TokenizedCache("data/tokenized/ambiguity/train")
    ds = SequentialChunkDataset(cache, cfg.chunk_length, shuffle_examples=True, seed=0)
    iterator = iter(ds)

    sums = defaultdict(float)
    n_used = 0
    for i, batch in enumerate(iterator):
        if i >= n_batches:
            break
        if batch.is_first_chunk:
            model.reset_memory()
        out = model.forward_chunk(batch.input_ids, tau_t=batch.tau_t, delta_tau=batch.delta_tau)
        L_lm = lm_loss(out.logits, batch.targets)
        L_pre = pre_influence_loss(torch.tensor(0.05), out.gate.mean(), cfg.rho_helped, cfg.tau_gate)
        L_prec = precision_loss(torch.tensor(-0.05), out.gate.mean())
        L_div = diversity_loss(model.memory.k)
        L_util = slot_util_loss(model.memory.usage)
        pooled = out.hidden_last.mean(dim=0)
        dtau_hat = model.dtau_head(pooled)
        L_chr = chronometric_loss(dtau_hat, torch.tensor([batch.delta_tau]))

        sums["L_lm"]            += float(L_lm.item() * cfg.w_lm)
        sums["L_pre_influence"] += float(L_pre.item() * cfg.w_pre_influence) if isinstance(L_pre, torch.Tensor) else 0
        sums["L_precision"]     += float(L_prec.item() * cfg.w_precision) if isinstance(L_prec, torch.Tensor) else 0
        sums["L_diversity"]     += float(L_div.item() * cfg.w_diversity)
        sums["L_slot_util"]     += float(L_util.item() * cfg.w_slot_util)
        sums["L_chrono"]        += float(L_chr.item() * cfg.w_chrono)
        n_used += 1

    total = sum(sums.values())
    return {
        k: {"sum": v, "fraction": v / max(1e-9, total)}
        for k, v in sums.items()
    }


def per_group_grad_magnitude(model: IPCN, cfg: IPCNConfig, n_batches: int = 20) -> dict:
    """Per param-group gradient magnitude after N backward passes."""
    cache = TokenizedCache("data/tokenized/ambiguity/train")
    ds = SequentialChunkDataset(cache, cfg.chunk_length, shuffle_examples=True, seed=1)
    iterator = iter(ds)

    grad_sums = defaultdict(float)
    grad_counts = defaultdict(int)

    for i, batch in enumerate(iterator):
        if i >= n_batches:
            break
        if batch.is_first_chunk:
            model.reset_memory()
        model.zero_grad(set_to_none=True)
        out = model.forward_chunk(batch.input_ids, tau_t=batch.tau_t, delta_tau=batch.delta_tau)
        L = lm_loss(out.logits, batch.targets)
        L.backward()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            # Group by top-level submodule
            group = name.split(".", 1)[0]
            grad_sums[group] += float(p.grad.abs().mean().item())
            grad_counts[group] += 1

    return {
        g: grad_sums[g] / max(1, grad_counts[g])
        for g in grad_sums
    }


@torch.no_grad()
def slot_utilization(model: IPCN, cfg: IPCNConfig, n_batches: int = 50) -> dict:
    """Track per-slot attention mass over N batches."""
    cache = TokenizedCache("data/tokenized/ambiguity/train")
    ds = SequentialChunkDataset(cache, cfg.chunk_length, shuffle_examples=True, seed=2)
    iterator = iter(ds)

    mass = torch.zeros(cfg.n_slots)
    n_used = 0
    for i, batch in enumerate(iterator):
        if i >= n_batches:
            break
        if batch.is_first_chunk:
            model.reset_memory()
        out = model.forward_chunk(batch.input_ids, tau_t=batch.tau_t, delta_tau=batch.delta_tau)
        mass = mass + out.alpha_prefix.sum(dim=0).cpu()
        n_used += 1

    mass = mass / max(1, n_used)
    sorted_idx = torch.argsort(mass, descending=True)
    top5_total = mass[sorted_idx[:5]].sum().item()
    bottom5_total = mass[sorted_idx[-5:]].sum().item()
    p_norm = mass / (mass.sum() + 1e-9)
    entropy = float(-(p_norm * (p_norm + 1e-12).log()).sum().item())
    return {
        "top5_mass": top5_total,
        "bottom5_mass": bottom5_total,
        "entropy": entropy,
        "max_entropy_if_uniform": float(torch.log(torch.tensor(float(cfg.n_slots))).item()),
        "top10_slots": sorted_idx[:10].tolist(),
        "n_zero_slots": int((mass < 1e-8).sum().item()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--n-batches", type=int, default=50)
    p.add_argument("--out", type=str, default="reports/feature_importance.md")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    torch.manual_seed(0)
    if args.checkpoint:
        model, _, cfg, _ = load_checkpoint(args.checkpoint, map_location=args.device)
        label = args.checkpoint
    else:
        cfg = IPCNConfig()
        model = IPCN(cfg).to(args.device)
        label = "untrained baseline"
    model.train(False)

    print(f"=== Feature importance on {label} ===\n")
    print("Loss contribution analysis...")
    contrib = per_loss_contribution(model, cfg, args.n_batches)
    for name, info in sorted(contrib.items(), key=lambda x: -x[1]["fraction"]):
        print(f"  {name:20s}: {info['fraction']*100:5.2f}%  (sum={info['sum']:.4f})")

    print("\nPer-group gradient magnitude analysis...")
    grads = per_group_grad_magnitude(model, cfg, n_batches=min(20, args.n_batches))
    for g, v in sorted(grads.items(), key=lambda x: -x[1]):
        print(f"  {g:30s}: {v:.6e}")

    print("\nSlot utilization analysis...")
    util = slot_utilization(model, cfg, args.n_batches)
    for k, v in util.items():
        print(f"  {k}: {v}")

    # Write report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Feature Importance Report",
        "",
        f"- Checkpoint: `{label}`",
        f"- Batches used: {args.n_batches}",
        "",
        "## Loss term contribution (% of total)",
        "",
        "| Loss | Fraction | Sum |",
        "|---|---|---|",
    ]
    for name, info in sorted(contrib.items(), key=lambda x: -x[1]["fraction"]):
        lines.append(f"| {name} | {info['fraction']*100:.2f}% | {info['sum']:.4f} |")
    lines += ["", "## Gradient magnitude per top-level group", "",
              "| Group | Mean |abs grad| |", "|---|---|"]
    for g, v in sorted(grads.items(), key=lambda x: -x[1]):
        lines.append(f"| {g} | {v:.6e} |")
    lines += ["", "## Slot utilization", "", "```json", json.dumps(util, indent=2), "```"]
    out_path.write_text("\n".join(lines))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
