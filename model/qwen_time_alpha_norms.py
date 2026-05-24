"""Per-layer alpha-norm dump + dominant-layer flip experiment.

Reviewer-mandated mech-interp follow-up (§24.7.9):
1. Dump |alpha| L2 norm per layer to identify which layers contribute most.
2. Flip ONLY the top-k dominant layers (vs random k) and rerun T1.
   Prediction: if dominant-layer-vote is real, top-k flip inverts r,
   random-k flip mostly preserves r.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics

import torch

from model.qwen_time import QwenTime, QwenTimeConfig, build_qwen_time
from model.qwen_time_check import (
    load_trainable,
    greedy_decode,
    parse_duration_to_seconds,
)
from model.qwen_time_extra_controls import (
    pearson,
    half_flip_alphas,
    restore_alphas,
    CLOCK_PROMPTS_PARAPHRASED,
)


def t1_under_state(model, device, taus, n_per_tau=3):
    prompt = CLOCK_PROMPTS_PARAPHRASED[0]
    pred, truth = [], []
    for tau in taus:
        for _ in range(n_per_tau):
            resp = greedy_decode(model, prompt, tau_t=tau, device=device)
            sec = parse_duration_to_seconds(resp)
            if sec == sec and sec > 0:
                pred.append(sec); truth.append(tau)
    return {"n": len(pred), "pearson_r": pearson(pred, truth)}


def flip_specific_layers(model, layer_keys):
    snapshot = {}
    layer_keys_set = set(layer_keys)
    for k, inj in model.chrono_injectors.items():
        snapshot[k] = inj.alpha.detach().clone()
        if k in layer_keys_set:
            with torch.no_grad():
                inj.alpha.mul_(-1.0)
    return snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="reports/alpha_norms.json")
    p.add_argument("--base", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--timescales", type=str, default="")
    p.add_argument("--top-k", type=int, default=8)
    args = p.parse_args()

    cfg = QwenTimeConfig()
    cfg.base_model_name = args.base
    if args.timescales:
        cfg.timescales = tuple(int(x) for x in args.timescales.split(","))
    print(f"Loading {cfg.base_model_name}...")
    model = build_qwen_time(cfg)
    model = model.to(args.device)
    print(f"Loading {args.checkpoint}...")
    load_trainable(model, args.checkpoint)
    model.train(False)

    print("\n=== Per-layer alpha L2 norm ===")
    norms = {}
    for k, inj in model.chrono_injectors.items():
        norms[k] = float(inj.alpha.detach().abs().mean().item())
    sorted_layers = sorted(norms.items(), key=lambda kv: -kv[1])
    print(f"  Top 10 dominant layers (by mean |alpha|):")
    for li, n in sorted_layers[:10]:
        print(f"    L{li}: {n:.5f}")
    print(f"  Bottom 5 layers:")
    for li, n in sorted_layers[-5:]:
        print(f"    L{li}: {n:.5f}")

    top_k_layers = [k for k, _ in sorted_layers[:args.top_k]]
    bot_k_layers = [k for k, _ in sorted_layers[-args.top_k:]]

    rng = random.Random(7777)
    taus = [math.exp(rng.uniform(math.log(2.0), math.log(14 * 86400.0)))
            for _ in range(10)]

    print(f"\n=== T1 with TOP-{args.top_k} dominant layers flipped ===")
    snap = flip_specific_layers(model, top_k_layers)
    r_top = t1_under_state(model, args.device, taus)
    restore_alphas(model, snap)
    print(f"  r={r_top['pearson_r']:+.4f} n={r_top['n']}")

    print(f"\n=== T1 with BOTTOM-{args.top_k} layers flipped ===")
    snap = flip_specific_layers(model, bot_k_layers)
    r_bot = t1_under_state(model, args.device, taus)
    restore_alphas(model, snap)
    print(f"  r={r_bot['pearson_r']:+.4f} n={r_bot['n']}")

    # Random k as control
    rng_ctrl = random.Random(99)
    random_layers = rng_ctrl.sample(list(norms.keys()), args.top_k)
    print(f"\n=== T1 with RANDOM-{args.top_k} layers flipped (control) ===")
    snap = flip_specific_layers(model, random_layers)
    r_rand = t1_under_state(model, args.device, taus)
    restore_alphas(model, snap)
    print(f"  r={r_rand['pearson_r']:+.4f} n={r_rand['n']}")

    out = {
        "per_layer_mean_abs_alpha": norms,
        "sorted_by_dominance": sorted_layers,
        "top_k": args.top_k,
        "top_k_layers": top_k_layers,
        "bottom_k_layers": bot_k_layers,
        "random_k_layers": random_layers,
        "T1_top_k_flipped": r_top,
        "T1_bottom_k_flipped": r_bot,
        "T1_random_k_flipped": r_rand,
        "verdict_dominant_layers_invert":
            r_top["pearson_r"] < -0.5 and r_bot["pearson_r"] > 0.5,
    }
    print("\n=== VERDICT ===")
    print(f"  top-k flip -> r={r_top['pearson_r']:+.3f}")
    print(f"  bot-k flip -> r={r_bot['pearson_r']:+.3f}")
    print(f"  random-k   -> r={r_rand['pearson_r']:+.3f}")
    print(f"  dominant_layers_invert: {out['verdict_dominant_layers_invert']}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
