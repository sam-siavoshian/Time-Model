"""Train Qwen IPCN on memorize-recall conversations.

Pipeline:
  1. Load Qwen IPCN wrapper (frozen base + trainable IPCN modules).
  2. Stream conversations from data/qwen_memrecall/train.jsonl.
  3. Tokenize each conversation with Qwen tokenizer.
  4. Split into chunks of cfg.chunk_length tokens. Each chunk is a
     forward+backward step; memory bank persists ACROSS chunks of the
     same conversation; reset at conversation boundary.
  5. LM loss on next-token prediction across all chunks. Memory writes
     happen automatically inside forward().
  6. Periodic consolidation passes after a warmup period.
  7. Save checkpoints.

Why this works (when Track A didn't): the memorize-recall data is
designed so the FACT is OUT OF CONTEXT by the time the recall question
arrives. The model HAS to use the memory bank to answer. If memory
ablation breaks recall, CTI > 0.

Usage:
  uv run python -m model.qwen_train --steps 5000 --device cuda \
       --out checkpoints/qwen_ipcn_v1.pt
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

from model.qwen_ipcn import QwenIPCN, QwenIPCNConfig, build_qwen_ipcn


def stream_chunks(
    jsonl_path: str,
    tokenizer,
    chunk_length: int,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterator[tuple[torch.Tensor, bool, int]]:
    """Yields (chunk_token_ids, is_first_chunk_of_conv, conv_idx) tuples.

    Each conversation is tokenized, then split into non-overlapping chunks
    of chunk_length tokens. The is_first_chunk flag tells the trainer
    when to reset the memory bank.
    """
    with open(jsonl_path) as f:
        lines = f.readlines()
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(lines)
    for conv_idx, line in enumerate(lines):
        rec = json.loads(line)
        ids = tokenizer.encode(rec["text"], return_tensors="pt").squeeze(0)
        # Walk over chunks
        L = ids.shape[0]
        i = 0
        first = True
        while i < L:
            chunk = ids[i: i + chunk_length]
            if chunk.shape[0] < 8:  # tiny tail chunk, skip
                break
            yield chunk, first, conv_idx
            first = False
            i += chunk_length


def train_step(
    model: QwenIPCN,
    opt: torch.optim.Optimizer,
    chunk: torch.Tensor,
    is_first: bool,
    chunk_idx_in_conv: int,
    device: str,
) -> dict:
    if is_first:
        model.reset_memory()
    chunk = chunk.to(device)
    # tau_t roughly = chunk_idx, delta_tau = 1
    out = model(chunk, tau_t=float(chunk_idx_in_conv), delta_tau=1.0)
    logits = out["logits"]                                     # (L, vocab)
    # Next-token LM loss: predict ids[1:] from logits[:-1]
    loss = F.cross_entropy(logits[:-1], chunk[1:], ignore_index=-100)
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
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/qwen_memrecall/train.jsonl")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-path", type=str, default="logs/qwen_ipcn.jsonl")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--out", type=str, default="checkpoints/qwen_ipcn.pt")
    p.add_argument("--chunk-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    cfg = QwenIPCNConfig()
    cfg.chunk_length = args.chunk_length
    print(f"Loading Qwen IPCN ({cfg.base_model_name})...")
    t0 = time.time()
    model = build_qwen_ipcn(cfg)
    model = model.to(args.device)
    print(f"  loaded in {time.time() - t0:.1f}s")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  trainable params: {sum(p.numel() for p in trainable_params):,}")

    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    log_f = open(args.log_path, "w", buffering=1)
    print(f"Training {args.steps} steps. Logging -> {args.log_path}")
    print(f"  data: {args.data}")
    step = 0
    iterator = stream_chunks(args.data, model.tokenizer, args.chunk_length, seed=args.seed)
    t_train = time.time()
    for chunk, is_first, conv_idx in iterator:
        if step >= args.steps:
            break
        # chunk_idx_in_conv (rough): just use step modulo for tau; conv_idx for re-anchoring
        rec = train_step(model, opt, chunk, is_first, step % 32, args.device)
        rec.update({"step": step, "conv_idx": conv_idx, "is_first": is_first, "time": time.time()})
        log_f.write(json.dumps(rec) + "\n")
        if step % args.log_every == 0:
            print(
                f"step={step:6d} | loss={rec['loss']:7.4f} ppl={rec['ppl']:8.1f} "
                f"grad={rec['grad_norm']:6.3f} conv={conv_idx:5d}"
            )
        step += 1
    log_f.write(json.dumps({"event": "training_complete", "step": step,
                            "max_steps": args.steps, "reason": "max_steps",
                            "time": time.time()}) + "\n")
    log_f.close()

    # Save checkpoint: only the trainable params + memory bank state.
    save = {
        "trainable_state": {
            n: p.detach().cpu()
            for n, p in model.named_parameters() if p.requires_grad
        },
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
