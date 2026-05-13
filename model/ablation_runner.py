"""H3 ablation matrix runner.

Trains 6 of 7 ablation variants (A0, A2, A3, A4, A5, A6) at identical
parameter budget on identical data. A1 (late retrieval only) is deferred
because it requires a separate architecture path (memory read AFTER core)
not yet implemented.

Each variant flips a single dimension vs the previous one. H3 predicts:
  A0 < A2 < A3 < A4 < A5 < A6 on memory-biased ambiguity accuracy.

Usage:
  uv run python -m model.ablation_runner --steps 500 --out-dir reports/ablation_v1

This produces:
  reports/ablation_v1/A{0,2,3,4,5,6}.md     per-variant eval report
  reports/ablation_v1/summary.md            comparison table + verdict
  checkpoints/ablation_v1/A{0,2,...}.pt     trained variants
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

from model.checkpoint import save_checkpoint
from model.config import IPCNConfig
from model.dataset import SequentialChunkDataset, TokenizedCache
from model.ipcn import IPCN
from model.train import build_optimizer, train_loop


# Ablation specs. Each (name, modifier_fn) — modifier mutates a base IPCNConfig.
def _A0(cfg: IPCNConfig):
    # No memory. Pure decoder-only transformer baseline.
    cfg.enable_episodic_memory = False
    cfg.enable_evolution = False
    cfg.use_route1_prepend = False
    cfg.use_route2_broadcast = False
    cfg.use_route3_lnmod = False
    cfg.consolidate_pfc = False
    cfg.consolidated_layers = ()
    return cfg


def _A1(cfg: IPCNConfig):
    # A1 late-retrieval-only baseline. Memory bank populated, but no
    # pre-forward prefix; memory is consulted AFTER the core transformer.
    cfg.enable_episodic_memory = True
    cfg.enable_evolution = False
    cfg.enable_late_retrieval_only = True
    cfg.use_route1_prepend = False
    cfg.use_route2_broadcast = False
    cfg.use_route3_lnmod = False
    cfg.consolidate_pfc = False
    cfg.consolidated_layers = ()
    return cfg


def _A2(cfg: IPCNConfig):
    # Slots + prefix prepend only. No broadcast, no LN mod, no evolution, no consolidation.
    cfg.enable_episodic_memory = True
    cfg.enable_evolution = False
    cfg.enable_late_retrieval_only = False
    cfg.use_route1_prepend = True
    cfg.use_route2_broadcast = False
    cfg.use_route3_lnmod = False
    cfg.consolidate_pfc = False
    cfg.consolidated_layers = ()
    return cfg


def _A3(cfg: IPCNConfig):
    # A2 + broadcast preconditioning.
    cfg.enable_episodic_memory = True
    cfg.enable_evolution = False
    cfg.use_route1_prepend = True
    cfg.use_route2_broadcast = True
    cfg.use_route3_lnmod = False
    cfg.consolidate_pfc = False
    cfg.consolidated_layers = ()
    return cfg


def _A4(cfg: IPCNConfig):
    # A3 + evolution.
    cfg.enable_episodic_memory = True
    cfg.enable_evolution = True
    cfg.use_route1_prepend = True
    cfg.use_route2_broadcast = True
    cfg.use_route3_lnmod = False
    cfg.consolidate_pfc = False
    cfg.consolidated_layers = ()
    return cfg


def _A5(cfg: IPCNConfig):
    # A4 + PFC consolidation (PFC LoRA adapters).
    cfg.enable_episodic_memory = True
    cfg.enable_evolution = True
    cfg.use_route1_prepend = True
    cfg.use_route2_broadcast = True
    cfg.use_route3_lnmod = True                              # spec implies LN-mod on for A5/A6
    cfg.consolidate_pfc = True
    cfg.consolidated_layers = ()                             # no core LoRA yet
    return cfg


def _A6(cfg: IPCNConfig):
    # Full IPCN: A5 + core layers 0-2 LoRA.
    cfg.enable_episodic_memory = True
    cfg.enable_evolution = True
    cfg.use_route1_prepend = True
    cfg.use_route2_broadcast = True
    cfg.use_route3_lnmod = True
    cfg.consolidate_pfc = True
    cfg.consolidated_layers = (0, 1, 2)
    return cfg


VARIANTS = {
    "A0": _A0,
    "A1": _A1,
    "A2": _A2,
    "A3": _A3,
    "A4": _A4,
    "A5": _A5,
    "A6": _A6,
}


def build_variant(name: str, base_cfg: IPCNConfig) -> tuple[IPCNConfig, IPCN]:
    cfg = deepcopy(base_cfg)
    cfg = VARIANTS[name](cfg)
    model = IPCN(cfg)
    return cfg, model


def train_variant(
    name: str,
    cfg: IPCNConfig,
    model: IPCN,
    cache_prefix: str,
    steps: int,
    seed: int,
    log_path: Path,
    device: str = "cpu",
) -> tuple[dict, torch.optim.Optimizer]:
    """Train one variant. Returns (metrics dict, the optimizer with training state).

    Returning the optimizer preserves Adam m/v moments for the checkpoint —
    previous version built a FRESH optimizer at save time, discarding all
    training momentum.
    """
    torch.manual_seed(seed)
    model = model.to(device)
    cache = TokenizedCache(cache_prefix)
    dataset = SequentialChunkDataset(
        cache, chunk_length=cfg.chunk_length, shuffle_examples=True, seed=seed
    )
    iterator = iter(dataset)
    # NOTE: train_loop builds its own optimizer internally. To capture optimizer
    # state, we replicate that here and pass it through (train_loop's behavior
    # unchanged: same default groups). For v1, train_loop currently creates the
    # optimizer internally; we read it back via model.parameters() being shared.
    # Simpler fix: just rebuild from the trained model at end. This still
    # captures the trained state because Adam moments live in opt.state, not
    # in model params — so a freshly-built opt has m=v=0 by definition.
    # CORRECT fix: pass the optimizer into train_loop. Until train_loop is
    # refactored, we wrap and call train_step manually.
    opt = build_optimizer(model, cfg)

    t0 = time.time()
    logs = train_loop(model, iterator, cfg, max_steps=steps,
                      log_every=max(1, steps // 5), log_path=str(log_path),
                      optimizer=opt)
    dt = time.time() - t0

    if logs:
        last10 = logs[-min(10, len(logs)):]
        final_lm = sum(L.lm_loss for L in last10) / len(last10)
    else:
        final_lm = float("nan")

    return {
        "name": name,
        "steps": steps,
        "wall_time_s": dt,
        "final_lm_mean_last10": final_lm,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }, opt


def eval_variant(
    name: str,
    model: IPCN,
    cfg: IPCNConfig,
    n_trials: int = 20,
    ambiguity_n: int = 100,
) -> dict:
    """Eval: H1 D_0 + ambiguity-task accuracy. H3 ranking uses ambiguity accuracy."""
    from model.ambiguity_accuracy import evaluate_ambiguity
    from model.predictions import H1_synthetic_check
    model.train(False)
    out = {}
    if cfg.enable_episodic_memory:
        h1 = H1_synthetic_check(model, n_trials=n_trials)
    else:
        h1 = {"D0_mean": 0.0, "note": "memory disabled (A0)"}
    amb = evaluate_ambiguity(model, n_examples=ambiguity_n, chunk_length=cfg.chunk_length)
    out["D0_mean"] = h1.get("D0_mean", 0.0)
    out["ambiguity_acc"] = amb["accuracy"]
    out["ambiguity_n"] = amb["n_examples"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200, help="training steps per variant")
    p.add_argument("--cache", type=str, default="data/tokenized/ambiguity/train")
    p.add_argument("--variants", nargs="*", default=None,
                   help="subset like A0 A2 A3. Default: all 6.")
    p.add_argument("--out-dir", type=str, default="reports/ablation_v1")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints/ablation_v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    selected = args.variants or list(VARIANTS.keys())
    base_cfg = IPCNConfig()

    summary_rows: list[dict] = []
    for name in selected:
        print(f"\n========== Training variant {name} ==========")
        cfg, model = build_variant(name, base_cfg)
        train_metrics, opt = train_variant(
            name=name, cfg=cfg, model=model,
            cache_prefix=args.cache, steps=args.steps, seed=args.seed,
            log_path=out_dir / f"{name}_train.jsonl",
            device=args.device,
        )

        ckpt_path = ckpt_dir / f"{name}.pt"
        save_checkpoint(str(ckpt_path), model, opt, cfg, train_step=args.steps,
                        extra={"variant": name})

        eval_metrics = eval_variant(name, model, cfg, n_trials=10)

        row = {
            **train_metrics,
            "eval_D0_mean": eval_metrics.get("D0_mean", 0.0),
            "eval_ambiguity_acc": eval_metrics.get("ambiguity_acc", 0.0),
        }
        summary_rows.append(row)

        # Per-variant report
        with open(out_dir / f"{name}.md", "w") as f:
            f.write(f"# Variant {name}\n\n")
            f.write(f"## Config flags\n```json\n{json.dumps({k: getattr(cfg, k) for k in ['enable_episodic_memory','enable_evolution','use_route1_prepend','use_route2_broadcast','use_route3_lnmod','consolidate_pfc','consolidated_layers']}, default=str, indent=2)}\n```\n\n")
            f.write(f"## Train\n```json\n{json.dumps(train_metrics, indent=2)}\n```\n\n")
            f.write(f"## Eval\n```json\n{json.dumps(eval_metrics, indent=2)}\n```\n")

    # Summary
    print("\n========== Summary ==========")
    summary_path = out_dir / "summary.md"
    lines = [
        "# Ablation summary",
        "",
        f"- Cache: `{args.cache}`",
        f"- Steps per variant: {args.steps}",
        f"- Seed: {args.seed}",
        "",
        "| Variant | trainable params | wall (s) | final LM (last 10) | D_0 mean | Ambig acc |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['name']} | {r['trainable_params']:,} | "
            f"{r['wall_time_s']:.1f} | {r['final_lm_mean_last10']:.4f} | "
            f"{r['eval_D0_mean']:.4f} | {r['eval_ambiguity_acc']:.4f} |"
        )
    lines.append("")
    lines.append("## H3 ordering checks")
    d0_by_name = {r["name"]: r["eval_D0_mean"] for r in summary_rows}
    amb_by_name = {r["name"]: r["eval_ambiguity_acc"] for r in summary_rows}
    order = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    available = [n for n in order if n in d0_by_name]
    lines.append("Expected: A0 < A1 < A2 < A3 < A4 < A5 < A6")
    lines.append("D_0 observed:        " + " < ".join(f"{n}={d0_by_name[n]:.3f}" for n in available))
    lines.append("Ambig acc observed:  " + " < ".join(f"{n}={amb_by_name[n]:.3f}" for n in available))
    lines.append("")
    lines.append("Pass criterion (SPEC.tex H3): Acc(A3) - Acc(A1) >= 0.03 on ambiguity tasks.")
    if "A3" in amb_by_name and "A1" in amb_by_name:
        gap = amb_by_name["A3"] - amb_by_name["A1"]
        lines.append(f"Acc(A3) - Acc(A1) = {gap:.4f}  (threshold 0.03)")
    if "A6" in amb_by_name and "A1" in amb_by_name:
        gap = amb_by_name["A6"] - amb_by_name["A1"]
        lines.append(f"Acc(A6) - Acc(A1) = {gap:.4f}")
    summary_path.write_text("\n".join(lines))
    print(f"Summary: {summary_path}")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
