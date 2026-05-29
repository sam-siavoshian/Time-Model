"""Teacher-forced T4 with TOKEN LABELS at each position.

§24.7.7 teacher-forced KL has spikes at positions 3,5,6 and zeros at
positions 1,2,7. Reviewer asks: which TOKENS are at those positions?
Likely positions 3-6 are number+unit tokens ("3", " hours", etc.) while
1,2,7 are scaffolding ("has", "been", "."). This script labels each
position with the actual decoded token to confirm.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.nn.functional as F

from model.qwen_time import QwenTimeConfig, build_qwen_time
from model.qwen_time_check import load_trainable


@torch.no_grad()
def teacher_forced_kl_labeled(model, prompt, taus, n_positions, device):
    tok = model.tokenizer
    ids = tok.encode(prompt, return_tensors="pt").squeeze(0).to(device)
    cur = ids
    anchor_traj = []
    for _ in range(n_positions):
        out = model(cur, tau_t=taus[0])
        nid = int(out["logits"][-1].argmax().item())
        anchor_traj.append(nid)
        cur = torch.cat([cur, torch.tensor([nid], device=device)])
    # Decode token labels
    token_strs = [tok.decode([nid]) for nid in anchor_traj]
    # Per-position KL
    per_pos = [[] for _ in range(n_positions)]
    cur_anchor = ids
    anchor_dists = []
    for pos in range(n_positions):
        out = model(cur_anchor, tau_t=taus[0])
        anchor_dists.append(F.log_softmax(out["logits"][-1].float().cpu(), dim=-1))
        cur_anchor = torch.cat([cur_anchor, torch.tensor([anchor_traj[pos]], device=device)])
    for tau in taus[1:]:
        cur_t = ids
        for pos in range(n_positions):
            out = model(cur_t, tau_t=tau)
            ld = F.log_softmax(out["logits"][-1].float().cpu(), dim=-1)
            pa = anchor_dists[pos].exp(); pb = ld.exp()
            kl = 0.5 * ((pa * (anchor_dists[pos] - ld)).sum() +
                        (pb * (ld - anchor_dists[pos])).sum()).item()
            per_pos[pos].append(kl)
            cur_t = torch.cat([cur_t, torch.tensor([anchor_traj[pos]], device=device)])
    return {
        "tokens": token_strs,
        "kls_per_pos": [statistics.mean(p) if p else 0.0 for p in per_pos],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--base", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--timescales", type=str, default="")
    p.add_argument("--n-positions", type=int, default=10)
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

    # Clock prompt to study number-vs-scaffolding pattern
    prompts = [
        ("clock", "<|im_start|>user\nHow long has it been since we started?<|im_end|>\n<|im_start|>assistant\n"),
        ("hello", "<|im_start|>user\nHello.<|im_end|>\n<|im_start|>assistant\n"),
    ]
    taus = [15.0, 3600.0, 86400.0]

    out = {}
    for name, prompt in prompts:
        print(f"\n=== Prompt: {name} ===")
        r = teacher_forced_kl_labeled(model, prompt, taus, args.n_positions, args.device)
        print(f"  anchor trajectory tokens (tau={taus[0]}):")
        for pos, (tok_str, kl) in enumerate(zip(r["tokens"], r["kls_per_pos"])):
            marker = "<<<" if kl > 1.0 else "   "
            print(f"    pos {pos}: KL={kl:7.3f}  token={repr(tok_str)[:30]:30s}  {marker}")
        out[name] = r

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
