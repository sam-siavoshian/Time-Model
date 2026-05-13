"""Run all 7 falsifiable predictions in sequence. Produce a Markdown report.

Usage:
  uv run python -m model.eval_all --checkpoint checkpoints/phase0.pt --out reports/phase0_eval.md
  uv run python -m model.eval_all   (untrained baseline)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from model.checkpoint import load_checkpoint
from model.config import IPCNConfig
from model.ipcn import IPCN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--pre-checkpoint", type=str, default=None,
                   help="for H4 only: model BEFORE consolidation")
    p.add_argument("--out", type=str, default="reports/eval_all.md")
    p.add_argument("--n-trials", type=int, default=20,
                   help="trials per test (smaller = faster)")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    torch.manual_seed(0)
    if args.checkpoint:
        model, _, cfg, train_step = load_checkpoint(args.checkpoint, map_location=args.device)
        ckpt_label = args.checkpoint
    else:
        cfg = IPCNConfig()
        model = IPCN(cfg).to(args.device)
        train_step = 0
        ckpt_label = "untrained baseline"
    model.train(False)

    pre_model = None
    if args.pre_checkpoint:
        pre_model, _, _, _ = load_checkpoint(args.pre_checkpoint, map_location=args.device)
        pre_model.train(False)

    results: dict[str, dict] = {}

    # H1
    print("Running H1 (D_0 memory-swap)...")
    from model.predictions import H1_synthetic_check
    results["H1"] = H1_synthetic_check(model, n_trials=args.n_trials)

    # H2
    print("Running H2 (linear probe)...")
    from model.h2_probe import H2_probe_check
    results["H2"] = H2_probe_check(model, n_pairs=min(args.n_trials, 50))

    # H3 — full ablation matrix training is multi-run; skip in this runner
    results["H3"] = {
        "status": "deferred",
        "note": "H3 requires training A0-A6 variants. Run separately via run_phase --ablation.",
    }

    # H4
    if args.pre_checkpoint:
        print("Running H4 (CTI)...")
        from model.h4_cti import compute_CTI
        rules = []
        with open("data/consolidation/ladder_train.jsonl") as f:
            for i, line in enumerate(f):
                if i >= args.n_trials:
                    break
                rules.append(json.loads(line))
        results["H4"] = compute_CTI(pre_model, model, rules, chunk_length=cfg.chunk_length)
    else:
        results["H4"] = {
            "status": "skipped",
            "note": "H4 requires --pre-checkpoint (model BEFORE consolidation) plus current checkpoint.",
        }

    # H5
    print("Running H5 (silent-gap evolution)...")
    from model.h5_evolution import H5_check
    streams = []
    with open("data/latent_world/test_16k.jsonl") as f:
        for i, line in enumerate(f):
            if i >= min(args.n_trials, 10):
                break
            streams.append(json.loads(line))
    results["H5"] = H5_check(model, streams, chunk_length=cfg.chunk_length)

    # H6 — use engineered chronometric_pairs dataset
    print("Running H6 (chronometric pairs)...")
    from model.predictions import H6_pairs_check
    results["H6"] = H6_pairs_check(model, n_pairs=min(args.n_trials, 100))

    # H7
    print("Running H7 (contradiction pairs)...")
    from model.h7_contradiction import H7_check
    results["H7"] = H7_check(model, n_pairs=args.n_trials, chunk_length=cfg.chunk_length)

    # ---------- Render report ----------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# IPCN Eval Report")
    lines.append("")
    lines.append(f"- Checkpoint: `{ckpt_label}`")
    lines.append(f"- Train step: {train_step}")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Trials per test: {args.n_trials}")
    lines.append("")
    lines.append("## Pre-registered predictions")
    lines.append("")
    lines.append("| # | Metric | Result | Threshold | Pass |")
    lines.append("|---|---|---|---|---|")
    r1 = results["H1"]; lines.append(f"| H1 | D_0 mean | {r1.get('D0_mean', 0):.4f} | > 0.1 | {'PASS' if r1.get('passes') else 'FAIL'} |")
    r2 = results["H2"]; lines.append(f"| H2 | probe_acc max(H0,H1) | {max(r2.get('probe_acc_H0', 0), r2.get('probe_acc_H1', 0)):.4f} | >= 0.80 | {'PASS' if r2.get('passes_either') else 'FAIL'} |")
    lines.append(f"| H3 | ablation order A0..A6 | -- | -- | DEFERRED |")
    if "CTI_mean" in results["H4"]:
        r4 = results["H4"]; lines.append(f"| H4 | CTI mean | {r4.get('CTI_mean', 0):.4f} | > 0.70 | {'PASS' if r4.get('passes') else 'FAIL'} |")
    else:
        lines.append(f"| H4 | CTI mean | -- | -- | SKIPPED |")
    r5 = results["H5"]; lines.append(f"| H5 | Acc(evolve) - Acc(static) | {r5.get('gap', 0):.4f} | >= 0.15 | {'PASS' if r5.get('passes') else 'FAIL'} |")
    r6 = results["H6"]; lines.append(f"| H6 | KL(real, ablated) | {r6.get('KL_mean', 0):.6f} | nonzero | {'NONZERO' if r6.get('KL_mean', 0) > 0 else 'ZERO'} |")
    r7 = results["H7"]; lines.append(f"| H7 | KL_amb / KL_exp | {r7.get('KL_amb_mean', 0):.4f} / {r7.get('KL_exp_mean', 0):.4f} | >= 0.5 / <= 0.1 | {'PASS' if r7.get('passes_overall') else 'FAIL'} |")
    lines.append("")

    lines.append("## Raw results")
    lines.append("")
    for h, r in results.items():
        lines.append(f"### {h}")
        lines.append("```json")
        lines.append(json.dumps(r, indent=2))
        lines.append("```")
        lines.append("")

    out.write_text("\n".join(lines))
    print(f"\nReport written to {out}")
    # Also dump JSON
    with open(str(out).replace(".md", ".json"), "w") as f:
        json.dump({"checkpoint": ckpt_label, "train_step": train_step, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
