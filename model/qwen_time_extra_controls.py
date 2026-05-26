"""Reviewer-rigor extra controls (2026-05-23 audit).

Three additional experiments hostile reviewers flagged as missing:

1. F. Half-layer alpha sign flip (random ~50% of layers flipped).
   Reviewer attack: "alpha-flip r=-0.9998 just confirms chrono is
   monotone in tau. A true single-coherent-scalar-axis would predict
   half-flip gives r near 0, not strongly negative." We test that.

2. Paraphrase T1 check.
   Reviewer attack: "T1 prompt is verbatim in training data. Model
   memorizes the formatter vocabulary, not duration." We test against
   10 paraphrased prompts the model never saw.

3. Teacher-forced T4.
   Reviewer attack: "Per-position KL grows 0.18 -> 27 because greedy
   decode commits to different first tokens at different tau, then
   downstream logits diverge from autoregressive drift not chrono use."
   Teacher-forced T4 holds the first N tokens constant across tau and
   measures KL at later positions. If teacher-forced KL stays small,
   the multi-pos growth was drift not chrono routing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics

import torch
import torch.nn.functional as F

from model.qwen_time import QwenTime, QwenTimeConfig, build_qwen_time
from model.qwen_time_check import (
    load_trainable,
    greedy_decode,
    parse_duration_to_seconds,
)


CLOCK_PROMPTS_PARAPHRASED = [
    "<|im_start|>user\nHow long has it been since we started?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nFrom when we began until now, what is the elapsed duration?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWhat's the total time so far?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nHow many seconds or minutes have gone by?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nReport the wall-clock time since we kicked off.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nDuration check: how much time has passed?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nSay how long this conversation has been going.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nTime elapsed?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nIn human terms, how long ago did we start?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nGive me a rough estimate of how long we've been at this.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nThis chat has lasted approximately how long?<|im_end|>\n<|im_start|>assistant\n",
]


def pearson(pred, truth):
    if len(pred) < 4:
        return float("nan")
    mp, mt = statistics.mean(pred), statistics.mean(truth)
    num = sum((p - mp) * (t - mt) for p, t in zip(pred, truth))
    denom = (sum((p - mp) ** 2 for p in pred) *
             sum((t - mt) ** 2 for t in truth)) ** 0.5
    return num / denom if denom > 0 else 0.0


# Half-layer alpha flip helpers
def half_flip_alphas(model, fraction=0.5, seed=42):
    rng = random.Random(seed)
    snapshot = {}
    layer_keys = list(model.chrono_injectors.keys())
    n_flip = int(len(layer_keys) * fraction)
    flip_set = set(rng.sample(layer_keys, n_flip))
    for k, inj in model.chrono_injectors.items():
        snapshot[k] = inj.alpha.detach().clone()
        if k in flip_set:
            with torch.no_grad():
                inj.alpha.mul_(-1.0)
    return snapshot, flip_set


def restore_alphas(model, snapshot):
    for k, inj in model.chrono_injectors.items():
        with torch.no_grad():
            inj.alpha.copy_(snapshot[k])


def t1_under_alpha_state(model, device, taus, n_per_tau=3):
    prompt = CLOCK_PROMPTS_PARAPHRASED[0]
    pred, truth = [], []
    for tau in taus:
        for _ in range(n_per_tau):
            resp = greedy_decode(model, prompt, tau_t=tau, device=device)
            sec = parse_duration_to_seconds(resp)
            if sec == sec and sec > 0:
                pred.append(sec); truth.append(tau)
    return {"n": len(pred), "pearson_r": pearson(pred, truth),
            "n_unique_tau": len(set(truth))}


def run_alpha_flip_battery(model, device):
    rng = random.Random(7777)
    taus = [math.exp(rng.uniform(math.log(2.0), math.log(14 * 86400.0)))
            for _ in range(10)]
    results = {}
    print(f"  taus: {[round(t, 1) for t in taus]}")
    print("  A. normal alphas")
    results["A_normal"] = t1_under_alpha_state(model, device, taus)
    print("  B. all alphas flipped")
    snap, _ = half_flip_alphas(model, fraction=1.0, seed=42)
    results["B_all_flipped"] = t1_under_alpha_state(model, device, taus)
    restore_alphas(model, snap)
    print("  C. half alphas flipped (seed 42)")
    snap, flip_c = half_flip_alphas(model, fraction=0.5, seed=42)
    results["C_half_flipped_42"] = t1_under_alpha_state(model, device, taus)
    results["C_half_flipped_42"]["flipped_layers"] = sorted(flip_c)
    restore_alphas(model, snap)
    print("  D. half alphas flipped (seed 7)")
    snap, flip_d = half_flip_alphas(model, fraction=0.5, seed=7)
    results["D_half_flipped_7"] = t1_under_alpha_state(model, device, taus)
    results["D_half_flipped_7"]["flipped_layers"] = sorted(flip_d)
    restore_alphas(model, snap)
    print("  E. third alphas flipped")
    snap, flip_e = half_flip_alphas(model, fraction=0.33, seed=1)
    results["E_third_flipped"] = t1_under_alpha_state(model, device, taus)
    results["E_third_flipped"]["flipped_layers"] = sorted(flip_e)
    restore_alphas(model, snap)
    return results


# Paraphrase T1
def t1_paraphrase_check(model, device, n_per_tau=3):
    """Paraphrase T1 with FULL response logging (v2). Reviewer attack:
    paraphrase r=+0.996 identical to 6 decimals could mean model outputs
    bit-identical strings ignoring prompt. Log all responses to verify.
    Reports: per-prompt r, per-(prompt,tau) response text, and
    response-identity matrix (frac of (i,j) prompt pairs producing
    identical response per tau)."""
    rng = random.Random(31337)
    taus = [math.exp(rng.uniform(math.log(2.0), math.log(7 * 86400.0)))
            for _ in range(8)]
    per_prompt = []
    responses_by_tau = {round(t, 3): [] for t in taus}
    for pi, prompt in enumerate(CLOCK_PROMPTS_PARAPHRASED):
        pred, truth, resps = [], [], []
        for tau in taus:
            for _ in range(n_per_tau):
                resp = greedy_decode(model, prompt, tau_t=tau, device=device)
                sec = parse_duration_to_seconds(resp)
                resps.append({"tau": tau, "resp": resp, "parsed": sec})
                if sec == sec and sec > 0:
                    pred.append(sec); truth.append(tau)
                    if pi < 11:  # log all for first 11 prompts
                        responses_by_tau[round(tau, 3)].append(
                            {"prompt_idx": pi, "resp": resp})
        is_anchor = (pi == 0)
        per_prompt.append({
            "prompt_idx": pi,
            "is_anchor_prompt": is_anchor,
            "prompt": prompt.split("user\n")[1].split("<|im_end|>")[0],
            "n": len(pred),
            "n_unique_tau": len(set(truth)),
            "pearson_r": pearson(pred, truth),
            "responses": resps[:8],  # sample first 8 (one per tau)
        })
        tag = "ANCHOR" if is_anchor else "para "
        print(f"  [{tag}] r={per_prompt[-1]['pearson_r']:+.3f} n={len(pred)}  "
              f"q={per_prompt[-1]['prompt'][:60]}")
    rs = [p["pearson_r"] for p in per_prompt if p["pearson_r"] == p["pearson_r"]]
    # Compute response-identity matrix: at each tau, frac of prompts that
    # produced the SAME response string as the anchor prompt
    identity_per_tau = {}
    for tau_key, resps in responses_by_tau.items():
        if len(resps) < 2:
            continue
        anchor_resp = next((r["resp"] for r in resps if r["prompt_idx"] == 0), None)
        if anchor_resp is None:
            continue
        n_match = sum(1 for r in resps if r["resp"] == anchor_resp)
        identity_per_tau[tau_key] = {
            "n_prompts": len(resps),
            "n_matching_anchor": n_match,
            "fraction_identical_to_anchor": n_match / len(resps),
            "anchor_resp": anchor_resp[:80],
        }
    all_id = [v["fraction_identical_to_anchor"] for v in identity_per_tau.values()]
    return {
        "per_prompt": per_prompt,
        "anchor_r": per_prompt[0]["pearson_r"],
        "paraphrase_r_mean": statistics.mean(rs[1:]) if len(rs) > 1 else float("nan"),
        "paraphrase_r_std": statistics.stdev(rs[1:]) if len(rs) > 2 else float("nan"),
        "n_paraphrases": len(per_prompt) - 1,
        "response_identity_per_tau": identity_per_tau,
        "fraction_identical_mean": statistics.mean(all_id) if all_id else float("nan"),
    }


# Prompt-baseline injection support (W3 fix). When INJECT_PROMPT_TAU is
# set globally from --inject-prompt CLI, each tau gets [elapsed: X]
# prepended to its prompt. Each tau therefore sees a DIFFERENT prompt
# string, so we must re-tokenize per tau (cannot share the ids object).
INJECT_PROMPT_TAU = False


def _tau_text(tau: float) -> str:
    if tau < 60:
        return f"[elapsed: {tau:.1f}s]"
    if tau < 3600:
        return f"[elapsed: {tau/60:.1f}m]"
    if tau < 86400:
        h = int(tau // 3600); m = int((tau % 3600) // 60)
        return f"[elapsed: {h}h {m}m]"
    d = int(tau // 86400); h = int((tau % 86400) // 3600)
    return f"[elapsed: {d}d {h}h]"


def _maybe_inject_prompt(prompt: str, tau: float) -> str:
    if not INJECT_PROMPT_TAU:
        return prompt
    marker = "<|im_start|>user\n"
    if marker not in prompt:
        return prompt
    return prompt.replace(marker, f"{marker}{_tau_text(float(tau))} ", 1)


# Teacher-forced T4
@torch.no_grad()
def teacher_forced_kl(model, prompt, taus, n_positions, device):
    tok = model.tokenizer
    # Anchor uses taus[0]. For prompt-baseline mode, re-tokenize per tau
    # since each tau gets a different injected prompt string.
    anchor_prompt = _maybe_inject_prompt(prompt, taus[0])
    ids = tok.encode(anchor_prompt, return_tensors="pt").squeeze(0).to(device)
    cur = ids
    anchor_trajectory = []
    for _ in range(n_positions):
        out = model(cur, tau_t=taus[0])
        nid = int(out["logits"][-1].argmax().item())
        anchor_trajectory.append(nid)
        cur = torch.cat([cur, torch.tensor([nid], device=device)])
    per_position_kls = [[] for _ in range(n_positions)]
    cur_anchor = ids
    anchor_dists = []
    for pos in range(n_positions):
        out = model(cur_anchor, tau_t=taus[0])
        anchor_dists.append(F.log_softmax(out["logits"][-1].float().cpu(), dim=-1))
        cur_anchor = torch.cat([cur_anchor, torch.tensor([anchor_trajectory[pos]], device=device)])
    for tau in taus[1:]:
        # Re-tokenize for this tau's injected prompt (prompt-baseline mode)
        tau_prompt = _maybe_inject_prompt(prompt, tau)
        cur_t = tok.encode(tau_prompt, return_tensors="pt").squeeze(0).to(device)
        for pos in range(n_positions):
            out = model(cur_t, tau_t=tau)
            ld = F.log_softmax(out["logits"][-1].float().cpu(), dim=-1)
            p_a = anchor_dists[pos].exp(); p_b = ld.exp()
            kl = 0.5 * ((p_a * (anchor_dists[pos] - ld)).sum() +
                        (p_b * (ld - anchor_dists[pos])).sum()).item()
            per_position_kls[pos].append(kl)
            cur_t = torch.cat([cur_t, torch.tensor([anchor_trajectory[pos]], device=device)])
    return [statistics.mean(p) if p else 0.0 for p in per_position_kls]


def t4_teacher_forced(model, device, n_positions=8):
    prompts = [
        "<|im_start|>user\nHello.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat time is it?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nGreetings.<|im_end|>\n<|im_start|>assistant\n",
    ]
    taus = [15.0, 3600.0, 86400.0]
    all_per_pos = [[] for _ in range(n_positions)]
    for p in prompts:
        kls = teacher_forced_kl(model, p, taus, n_positions, device)
        for pos, k in enumerate(kls):
            all_per_pos[pos].append(k)
    means = [statistics.mean(p) if p else 0.0 for p in all_per_pos]
    return {
        "teacher_forced_per_position_mean_kl": means,
        "teacher_forced_mean_kl": statistics.mean(means),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="reports/extra_controls.json")
    p.add_argument("--base", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--timescales", type=str, default="")
    p.add_argument("--inject-prompt", action="store_true",
                   help="Prepend [elapsed: X] to each prompt before "
                        "teacher-forced KL. Use when evaluating "
                        "prompt-baseline ckpts (W3 fix).")
    args = p.parse_args()

    global INJECT_PROMPT_TAU
    if args.inject_prompt:
        INJECT_PROMPT_TAU = True
        print("  prompt-injection mode: ON (W3 fix for prompt-baseline T4 teacher-forced)")

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

    print("\n=== 1/3 Half-layer alpha flip control ===")
    flip = run_alpha_flip_battery(model, args.device)
    for cond, r in flip.items():
        layers = r.get("flipped_layers", "all" if "all" in cond else "none")
        print(f"  {cond}: r={r['pearson_r']:+.4f}  n={r['n']}  uniq_tau={r['n_unique_tau']}  "
              f"layers_flipped={layers if isinstance(layers, str) else len(layers)}")

    print("\n=== 2/3 Paraphrase T1 check (10 unseen prompts) ===")
    para = t1_paraphrase_check(model, args.device)
    print(f"  anchor r={para['anchor_r']:+.3f}")
    pm = para['paraphrase_r_mean']
    ps = para['paraphrase_r_std']
    pm_s = f"{pm:+.3f}" if pm == pm else "NaN"
    ps_s = f"{ps:.3f}" if ps == ps else "NaN"
    print(f"  paraphrase r mean={pm_s} +/- {ps_s}  n={para['n_paraphrases']}")

    print("\n=== 3/3 Teacher-forced T4 ===")
    tf = t4_teacher_forced(model, args.device)
    print(f"  teacher-forced per-position KL: "
          f"{[f'{x:.2f}' for x in tf['teacher_forced_per_position_mean_kl']]}")
    print(f"  teacher-forced mean KL: {tf['teacher_forced_mean_kl']:.3f}")

    pm_ok = (pm == pm and pm >= 0.5)
    out = {
        "half_flip_battery": flip,
        "paraphrase_t1": para,
        "teacher_forced_t4": tf,
        "verdicts": {
            "PASS_anchor_T1_replicates": para["anchor_r"] >= 0.7,
            "PASS_paraphrase_T1_holds": pm_ok,
            "PASS_half_flip_kills_signal":
                abs(flip["C_half_flipped_42"]["pearson_r"]) < 0.5
                and abs(flip["D_half_flipped_7"]["pearson_r"]) < 0.5,
            "PASS_teacher_forced_T4_chrono_present":
                tf["teacher_forced_mean_kl"] >= 0.05,
        },
    }
    print("\n=== VERDICTS ===")
    for k, v in out["verdicts"].items():
        print(f"  {k}: {v}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
