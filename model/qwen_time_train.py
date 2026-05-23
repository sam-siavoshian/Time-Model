"""Trainer for QwenTime. Passes per-conversation tau_t into the model.

Each conversation has its own tau_t (per gen_*_conversation in
qwen_time_data.py). The forward pass uses that tau to set the chrono
encoder vector, which is injected at every layer.

LM loss is masked to the assistant's response tokens only (answer-mask
pattern from Track B). The whole response is one chunk per conversation;
no cross-chunk memory in this trainer (Track C focuses on time, not
memory routing).

Usage:
  uv run python -m model.qwen_time_train --data data/qwen_time/train_v1.jsonl \
      --steps 8000 --device cuda --out checkpoints/qwen_time_v1.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model.qwen_time import QwenTime, QwenTimeConfig, build_qwen_time


def stream_records(path: str, shuffle: bool = True, seed: int = 0):
    with open(path) as f:
        lines = f.readlines()
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(lines)
    for line in lines:
        yield json.loads(line)


def make_chunk(tokenizer, rec, max_len: int):
    prefix_ids = tokenizer.encode(rec["prefix_text"], return_tensors="pt").squeeze(0)
    answer_ids = tokenizer.encode(rec["answer_text"], return_tensors="pt", add_special_tokens=False).squeeze(0)
    ids = torch.cat([prefix_ids, answer_ids])
    mask = torch.zeros(ids.shape[0], dtype=torch.bool)
    mask[prefix_ids.shape[0]:] = True
    if ids.shape[0] > max_len:
        # Keep the tail so the answer-mask remains valid.
        ids = ids[-max_len:]
        mask = mask[-max_len:]
    return ids, mask


def train_step(model, opt, ids, mask, tau_t, device):
    ids = ids.to(device)
    mask = mask.to(device)
    out = model(ids, tau_t=float(tau_t))
    logits = out["logits"]
    targets = ids[1:].clone()
    targets[~mask[1:]] = -100
    if (targets >= 0).sum().item() == 0:
        return {"loss": float("nan"), "grad_norm": 0.0, "ppl": float("nan"), "n_target": 0}
    loss = F.cross_entropy(logits[:-1], targets, ignore_index=-100)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    opt.step()
    return {
        "loss": float(loss.item()),
        "grad_norm": float(grad_norm.item()),
        "ppl": math.exp(min(float(loss.item()), 20)),
        "n_target": int((targets >= 0).sum().item()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-path", type=str, default="logs/qwen_time.jsonl")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", type=str, default="checkpoints/qwen_time.pt")
    p.add_argument("--chunk-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--unfreeze-base", action="store_true")
    p.add_argument("--timescales", type=str, default="",
                   help="Comma-separated chrono timescales in seconds. Empty = use QwenTimeConfig default.")
    p.add_argument("--freeze-alpha", action="store_true",
                   help="Lock per-layer chrono alpha gates at 0 throughout training. "
                        "This is the LoRA-only ablation: chrono encoder + projectors "
                        "still exist and receive gradients but cannot influence the "
                        "residual stream. Used to falsify the architectural claim by "
                        "comparing v15 (alpha learns) vs alpha=0-frozen.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    cfg = QwenTimeConfig()
    cfg.base_model_name = args.base
    cfg.chunk_length = args.chunk_length
    cfg.unfreeze_base = args.unfreeze_base
    if args.timescales:
        cfg.timescales = tuple(int(x) for x in args.timescales.split(","))
        print(f"  Override timescales: {cfg.timescales}")
    print(f"Loading QwenTime ({cfg.base_model_name})...")
    t0 = time.time()
    model = build_qwen_time(cfg)
    model = model.to(args.device)
    print(f"  loaded in {time.time() - t0:.1f}s")
    n_train = model.trainable_parameter_count()
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.3f}%)")
    print(f"  chrono injectors: {len(model.chrono_injectors)}")
    print(f"  LoRA modules: {model._n_lora_modules}")

    if args.freeze_alpha:
        # LoRA-only ablation: lock all per-layer alpha gates at 0 and
        # remove them from the optimizer. Chrono encoder + projectors
        # still exist but cannot affect the residual stream.
        for inj in model.chrono_injectors.values():
            with torch.no_grad():
                inj.alpha.zero_()
            inj.alpha.requires_grad_(False)
        print(f"  FROZE all {len(model.chrono_injectors)} chrono alpha gates at 0 (LoRA-only ablation)")

    if args.unfreeze_base:
        # Split LR: LoRA + injectors at args.lr, base params at 1/100th
        # to prevent catastrophic forgetting / weight collapse.
        lora_or_inj = []
        base_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in n or "chrono" in n or "alpha" in n or "to_gamma" in n or "to_beta" in n:
                lora_or_inj.append(p)
            else:
                base_params.append(p)
        print(f"  unfrozen base: split LR groups -- lora/inj={len(lora_or_inj)} base={len(base_params)}")
        opt = torch.optim.AdamW(
            [{"params": lora_or_inj, "lr": args.lr},
             {"params": base_params, "lr": args.lr * 0.01}],
            weight_decay=0.01,
        )
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    log_f = open(args.log_path, "w", buffering=1)
    print(f"Training {args.steps} steps. Log -> {args.log_path}")
    step = 0
    iterator = stream_records(args.data, seed=args.seed)
    t_train = time.time()
    for rec in iterator:
        if step >= args.steps:
            break
        try:
            ids, mask = make_chunk(model.tokenizer, rec, args.chunk_length)
        except Exception as e:
            continue
        out = train_step(model, opt, ids, mask, rec.get("tau_t", 0.0), args.device)
        out.update({"step": step, "mode": rec.get("mode", "?"), "tau_t": rec.get("tau_t", 0.0), "time": time.time()})
        log_f.write(json.dumps(out) + "\n")
        if step % args.log_every == 0:
            print(
                f"step={step:6d} | loss={out['loss']:7.4f} ppl={out.get('ppl', 0):8.1f} "
                f"grad={out['grad_norm']:6.3f} mode={out['mode']:10s} tau={out['tau_t']:.1f} tgt={out.get('n_target', 0):3d}"
            )
        step += 1
    log_f.write(json.dumps({"event": "training_complete", "step": step,
                            "max_steps": args.steps, "reason": "max_steps",
                            "time": time.time()}) + "\n")
    log_f.close()

    save = {
        "trainable_state": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "cfg": cfg.__dict__,
        "train_step": step,
    }
    tmp = Path(args.out).with_name(Path(args.out).name + f".tmp.{os.getpid()}")
    torch.save(save, str(tmp))
    os.replace(str(tmp), args.out)
    print(f"Saved checkpoint to {args.out}")
    print(f"Total wall time: {time.time() - t_train:.1f}s")


if __name__ == "__main__":
    main()
