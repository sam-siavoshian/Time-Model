"""Aggressive falsification of v11 time-conditional behavior.

If v11's 4/5 pass is REAL (model learned a continuous tau -> behavior map),
turning off the chrono signal must DESTROY the behavior. If results barely
shift, v11 passed via template-matching artifacts (e.g. tokenization tricks,
prompt cues) and the paper claim is dead.

Conditions:
  A. v11 normal (baseline, replicated from check)
  B. alpha=0   -> chrono identity injection. Pure base + LoRA, no time.
  C. random tau at eval -> chrono signal disconnected from prompt context.
                            Model still sees A tau, but the wrong one.
  D. tau=0 always -> chrono signal pinned at zero (sin=0, cos=1, log=0).

We re-run T1 (clock consistency) under each condition. Expected:
  T1 pearson_r:  A: >=0.85   B: <=0.2    C: <=0.3    D: <=0.3
A high while B/C/D collapse = chrono signal is causally driving behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time

import torch

from model.qwen_time import QwenTime, QwenTimeConfig, build_qwen_time
from model.qwen_time_check import (
    load_trainable,
    greedy_decode,
    parse_duration_to_seconds,
)


PROMPT = (
    "<|im_start|>user\nHow long has it been since we started?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def pearson(pred, truth) -> float:
    if len(pred) < 4:
        return float("nan")
    mp, mt = statistics.mean(pred), statistics.mean(truth)
    num = sum((p - mp) * (t - mt) for p, t in zip(pred, truth))
    denom = (sum((p - mp) ** 2 for p in pred) *
             sum((t - mt) ** 2 for t in truth)) ** 0.5
    return num / denom if denom > 0 else 0.0


def t1_under_intervention(model, device, taus, eval_tau_override=None,
                          n_per_tau=4) -> dict:
    """Standard T1 but route each forward through eval_tau_override(true_tau)."""
    pred = []
    truth = []
    log_errs = []
    examples = []
    for true_tau in taus:
        for _ in range(n_per_tau):
            eval_tau = eval_tau_override(true_tau) if eval_tau_override else true_tau
            resp = greedy_decode(model, PROMPT, tau_t=eval_tau, device=device)
            sec = parse_duration_to_seconds(resp)
            if sec == sec and sec > 0:
                pred.append(sec)
                truth.append(true_tau)
                log_errs.append(abs(math.log10(sec) - math.log10(true_tau)))
            if len(examples) < 6:
                examples.append({"true_tau": true_tau, "eval_tau": eval_tau,
                                 "resp": resp, "parsed": sec})
    return {
        "n": len(pred),
        "pearson_r": pearson(pred, truth),
        "log_mae": statistics.mean(log_errs) if log_errs else float("nan"),
        "examples": examples,
    }


def snapshot_alpha(model):
    return {k: inj.alpha.detach().clone() for k, inj in model.chrono_injectors.items()}


def restore_alpha(model, snap):
    for k, inj in model.chrono_injectors.items():
        with torch.no_grad():
            inj.alpha.copy_(snap[k])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="reports/qwen_time_v11_falsify.json")
    p.add_argument("--base", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--seed", type=int, default=99)
    args = p.parse_args()

    rng = random.Random(args.seed)
    cfg = QwenTimeConfig()
    cfg.base_model_name = args.base
    print(f"Loading {cfg.base_model_name}...")
    model = build_qwen_time(cfg)
    model = model.to(args.device)
    print(f"Loading {args.checkpoint}...")
    load_trainable(model, args.checkpoint)
    model.train(False)

    # OOD taus (distinct from training grid)
    taus = []
    for _ in range(8):
        taus.append(math.exp(rng.uniform(math.log(2.0), math.log(14 * 86400.0))))
    print(f"  eval taus: {[round(t,1) for t in taus]}")

    # A: trained, no intervention
    print("\n=== A: v11 normal ===")
    t0 = time.time()
    A = t1_under_intervention(model, args.device, taus, eval_tau_override=None)
    print(f"  r={A['pearson_r']:.3f}, log_mae={A['log_mae']:.3f}")

    # B: alpha=0
    print("\n=== B: alpha=0 (chrono OFF) ===")
    snap = snapshot_alpha(model)
    for inj in model.chrono_injectors.values():
        with torch.no_grad():
            inj.alpha.zero_()
    B = t1_under_intervention(model, args.device, taus, eval_tau_override=None)
    print(f"  r={B['pearson_r']:.3f}, log_mae={B['log_mae']:.3f}")
    restore_alpha(model, snap)

    # C: random tau (eval tau swapped to a different random tau)
    print("\n=== C: random tau (eval tau != true tau) ===")
    def randomize(_t):
        return math.exp(rng.uniform(math.log(1.0), math.log(7 * 86400.0)))
    C = t1_under_intervention(model, args.device, taus, eval_tau_override=randomize)
    print(f"  r={C['pearson_r']:.3f}, log_mae={C['log_mae']:.3f}")

    # D: tau pinned to zero
    print("\n=== D: tau=0 pinned ===")
    D = t1_under_intervention(model, args.device, taus,
                              eval_tau_override=lambda _t: 0.0)
    print(f"  r={D['pearson_r']:.3f}, log_mae={D['log_mae']:.3f}")

    # E: tau=tau but alpha frozen at random sign (perturbation, sanity)
    print("\n=== E: alpha sign flipped (perturbation) ===")
    snap = snapshot_alpha(model)
    for inj in model.chrono_injectors.values():
        with torch.no_grad():
            inj.alpha.mul_(-1.0)
    E = t1_under_intervention(model, args.device, taus, eval_tau_override=None)
    print(f"  r={E['pearson_r']:.3f}, log_mae={E['log_mae']:.3f}")
    restore_alpha(model, snap)

    verdict = {
        "A_normal_r": A["pearson_r"],
        "B_alpha_off_r": B["pearson_r"],
        "C_random_tau_r": C["pearson_r"],
        "D_tau_zero_r": D["pearson_r"],
        "E_alpha_flipped_r": E["pearson_r"],
        "A_normal_log_mae": A["log_mae"],
        "B_alpha_off_log_mae": B["log_mae"],
        "PASS_chrono_causal": (
            A["pearson_r"] > 0.7
            and B["pearson_r"] < 0.3
            and C["pearson_r"] < 0.4
            and D["pearson_r"] < 0.4
        ),
    }
    print("\n=== VERDICT ===")
    for k, v in verdict.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "verdict": verdict,
            "A_normal": A,
            "B_alpha_off": B,
            "C_random_tau": C,
            "D_tau_zero": D,
            "E_alpha_flipped": E,
            "taus": taus,
        }, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
