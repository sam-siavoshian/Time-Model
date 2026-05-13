"""H5 silent-gap evolution: evolving memory beats static on long silent gaps.

Compare answer accuracy on Latent World questions that depend on hidden
state having evolved during the silent gap (decay, delayed transitions):

  Acc(IPCN_evolve)  with cfg.enable_evolution=True
  Acc(IPCN_static) with cfg.enable_evolution=False (memory frozen after writes)

Pass: Acc(evolve) - Acc(static) >= 0.15 on 64k+ context streams with
512+ silent minutes.

For prototype we run on smaller streams; the structural test is the
delta, not absolute accuracy.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import mean

import tiktoken
import torch
import torch.nn.functional as F

from model.checkpoint import load_checkpoint
from model.config import IPCNConfig
from model.h7_contradiction import render_memory_block
from model.ipcn import IPCN
from model.latent_world_loader import LatentWorldChunkDataset


@torch.no_grad()
def _score_stream(
    model: IPCN,
    enc,
    stream: dict,
    chunk_length: int,
) -> float:
    """Feed the stream's events as chunks, then score answers to questions.

    Simple accuracy: predict first answer token at the position where the
    Q: prompt ends.
    """
    model.reset_memory()
    # Render full stream text the way training did
    parts = ["<|stream|>"]
    for ev in stream["events"]:
        parts.append(ev["text"])
    parts.append("<|questions|>")
    correct = 0
    total = 0
    # Process events as a single context, chunked
    body_text = "\n".join(parts)
    tokens = enc.encode(body_text)
    # Feed event chunks
    prev_tau = 0.0
    chunk_i = 0
    for i in range(0, len(tokens), chunk_length):
        seg = tokens[i:i + chunk_length]
        if len(seg) < 2:
            break
        chunk_t = torch.tensor(seg, dtype=torch.long)
        if chunk_t.numel() < chunk_length:
            chunk_t = F.pad(chunk_t, (0, chunk_length - chunk_t.numel()), value=0)
        # Approximate tau by the last event we've seen in this chunk; for
        # simplicity scale linearly with chunk_i.
        out = model.forward_chunk(chunk_t, tau_t=float(chunk_i + 1) * 100.0, delta_tau=100.0)
        chunk_i += 1
    # Score each question
    for q in stream["questions"]:
        prompt = f"Q: {q['text']}\nA:"
        prompt_toks = torch.tensor(enc.encode(prompt), dtype=torch.long)
        if prompt_toks.numel() < chunk_length:
            prompt_toks_padded = F.pad(prompt_toks, (0, chunk_length - prompt_toks.numel()), value=0)
        else:
            prompt_toks_padded = prompt_toks[:chunk_length]
        out = model.forward_chunk(prompt_toks_padded, tau_t=10000.0, delta_tau=1.0)
        valid_len = (prompt_toks_padded != 0).sum().item()
        pred = out.logits[valid_len - 1].argmax().item()
        ans_toks = enc.encode(" " + q["answer"].strip())
        if not ans_toks:
            continue
        if pred == ans_toks[0]:
            correct += 1
        total += 1
    return correct / max(1, total)


def H5_check(
    model: IPCN,
    streams: list[dict],
    chunk_length: int,
) -> dict:
    enc = tiktoken.get_encoding("gpt2")
    cfg = model.cfg

    # Evolve arm
    cfg.enable_evolution = True
    acc_evolve = mean([_score_stream(model, enc, s, chunk_length) for s in streams])

    # Static arm
    cfg.enable_evolution = False
    acc_static = mean([_score_stream(model, enc, s, chunk_length) for s in streams])

    # Restore
    cfg.enable_evolution = True

    gap = acc_evolve - acc_static
    return {
        "n_streams": len(streams),
        "Acc_evolve": acc_evolve,
        "Acc_static": acc_static,
        "gap": gap,
        "threshold": 0.15,
        "passes": gap >= 0.15,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--streams", type=str, default="data/latent_world/test_16k.jsonl")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    torch.manual_seed(0)
    if args.checkpoint:
        model, _, cfg, _ = load_checkpoint(args.checkpoint, map_location=args.device)
    else:
        cfg = IPCNConfig()
        model = IPCN(cfg).to(args.device)
    model.train(False)

    streams = []
    with open(args.streams) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            streams.append(json.loads(line))

    print(f"=== H5 silent-gap evolution (n={len(streams)} streams) ===")
    r = H5_check(model, streams, chunk_length=cfg.chunk_length)
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
