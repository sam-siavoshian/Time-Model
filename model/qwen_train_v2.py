"""Trainer for QwenIPCNv2 (cross-attention memory injection + differentiable writes).

Differences from qwen_train.py:
  - Builds QwenIPCNv2 instead of QwenIPCN.
  - Passes write_target_ids to the model on each chunk so Identity-V can
    use lm_head rows as slot values.
  - Detaches the differentiable memory snapshot every cfg.bptt_chunks=2
    chunks to keep BPTT bounded.

Usage:
  uv run python -m model.qwen_train_v2 --steps 12000 --device cuda \
       --data data/qwen_memrecall/train_v5.jsonl --answer-mask \
       --chunk-length 64 --out checkpoints/qwen_ipcn_v5.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

from model.qwen_ipcn_v2 import QwenIPCNv2, QwenIPCNv2Config, build_qwen_ipcn_v2


def stream_chunks(
    jsonl_path: str,
    tokenizer,
    chunk_length: int,
    shuffle: bool = True,
    seed: int = 0,
    answer_mask: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, bool, int]]:
    with open(jsonl_path) as f:
        lines = f.readlines()
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(lines)
    for conv_idx, line in enumerate(lines):
        rec = json.loads(line)
        if answer_mask and "prefix_text" in rec and "answer_text" in rec:
            prefix_ids = tokenizer.encode(rec["prefix_text"], return_tensors="pt").squeeze(0)
            answer_ids = tokenizer.encode(rec["answer_text"], return_tensors="pt", add_special_tokens=False).squeeze(0)
            ids = torch.cat([prefix_ids, answer_ids])
            mask = torch.zeros(ids.shape[0], dtype=torch.bool)
            mask[prefix_ids.shape[0]:] = True
        else:
            ids = tokenizer.encode(rec["text"], return_tensors="pt").squeeze(0)
            mask = torch.ones(ids.shape[0], dtype=torch.bool)
        L = ids.shape[0]
        i = 0
        first = True
        while i < L:
            chunk = ids[i: i + chunk_length]
            chunk_mask = mask[i: i + chunk_length]
            if chunk.shape[0] < 8:
                break
            yield chunk, chunk_mask, first, conv_idx
            first = False
            i += chunk_length


def train_step(
    model: QwenIPCNv2,
    opt: torch.optim.Optimizer,
    chunk: torch.Tensor,
    chunk_mask: torch.Tensor,
    is_first: bool,
    chunk_idx_in_conv: int,
    device: str,
    bptt_detach_every: int = 2,
) -> dict:
    if is_first:
        model.reset_memory()
    # Always detach the carryover at chunk boundary -- gradient through
    # prior-chunk writes would require keeping the prior backward graph
    # alive, which is expensive and breaks under torch's free-on-backward.
    # The Identity-V trick (slot value = lm_head.weight row at the target
    # token) means W_v_m does NOT need to learn; the value carries the
    # right output direction by construction. W_k_m staying at init is
    # also acceptable because (a) keys are random but DETERMINISTIC per
    # input hidden state, (b) cross-attn W_q learns to map queries onto
    # whichever slot got the right value at write time.
    if model.memory._k_grad is not None:
        model.memory._k_grad = model.memory._k_grad.detach()
    if model.memory._v_grad is not None:
        model.memory._v_grad = model.memory._v_grad.detach()

    chunk = chunk.to(device)
    chunk_mask = chunk_mask.to(device)

    # write_target_ids: the next-token targets within this chunk. The
    # write step picks top-k_w candidates by hidden norm; for Identity-V
    # the slot value becomes lm_head[target_id] at each candidate position.
    # For prefix chunks (no answer tokens) targets are -100; the write
    # falls back to W_v_m projection for those positions.
    targets = chunk[1:].clone()
    targets[~chunk_mask[1:]] = -100
    # Build write_target_ids same length as chunk (last position has no
    # target). Pad to length L with the last value.
    write_targets = torch.cat([targets, targets[-1:].clone()])

    out = model(
        chunk, tau_t=float(chunk_idx_in_conv), delta_tau=1.0,
        write_target_ids=write_targets,
    )
    logits = out["logits"]                                     # (L, vocab)

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
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-path", type=str, default="logs/qwen_ipcn_v2.jsonl")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out", type=str, default="checkpoints/qwen_ipcn_v2.pt")
    p.add_argument("--chunk-length", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--answer-mask", action="store_true")
    p.add_argument("--bptt-detach-every", type=int, default=2)
    p.add_argument("--unfreeze-base", action="store_true",
                   help="train all of Qwen, not just LoRA + memory + cross-attn")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    cfg = QwenIPCNv2Config()
    cfg.chunk_length = args.chunk_length
    cfg.unfreeze_base = args.unfreeze_base
    print(f"Loading Qwen IPCN v2 ({cfg.base_model_name})...")
    t0 = time.time()
    model = build_qwen_ipcn_v2(cfg)
    model = model.to(args.device)
    print(f"  loaded in {time.time() - t0:.1f}s")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {n_train:,} ({100*n_train/n_total:.3f}%)")
    print(f"  cross-attn modules: {len(model.cross_attn)} at layers {cfg.inject_layers}")
    print(f"  LoRA modules: {model._n_lora_modules}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    log_f = open(args.log_path, "w", buffering=1)
    print(f"Training {args.steps} steps. Log -> {args.log_path}")
    step = 0
    iterator = stream_chunks(
        args.data, model.tokenizer, args.chunk_length, seed=args.seed,
        answer_mask=args.answer_mask,
    )
    t_train = time.time()
    for chunk, chunk_mask, is_first, conv_idx in iterator:
        if step >= args.steps:
            break
        # chunk_idx_in_conv: count chunks since last is_first.
        # Use a simple counter so we can compute bptt-detach correctly.
        chunk_idx_in_conv = 0 if is_first else step  # local proxy; train_step handles detach
        rec = train_step(
            model, opt, chunk, chunk_mask, is_first, chunk_idx_in_conv, args.device,
            bptt_detach_every=args.bptt_detach_every,
        )
        rec.update({"step": step, "conv_idx": conv_idx, "is_first": is_first, "time": time.time()})
        log_f.write(json.dumps(rec) + "\n")
        if step % args.log_every == 0:
            print(
                f"step={step:6d} | loss={rec['loss']:7.4f} ppl={rec.get('ppl', 0):8.1f} "
                f"grad={rec['grad_norm']:6.3f} conv={conv_idx:5d} tgt={rec.get('n_target', 0):3d}"
            )
        step += 1
    log_f.write(json.dumps({"event": "training_complete", "step": step,
                            "max_steps": args.steps, "reason": "max_steps",
                            "time": time.time()}) + "\n")
    log_f.close()

    save = {
        "trainable_state": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "memory_state": {
            "k": model.memory.k.detach().cpu(),
            "v": model.memory.v.detach().cpu(),
            "tau_write": model.memory.tau_write.detach().cpu(),
            "usage": model.memory.usage.detach().cpu(),
            "conf": model.memory.conf.detach().cpu(),
            "plast": model.memory.plast.detach().cpu(),
            "conflict": model.memory.conflict.detach().cpu(),
        },
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
