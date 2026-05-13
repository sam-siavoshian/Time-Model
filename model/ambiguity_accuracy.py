"""Ambiguity-task accuracy harness.

For each AmbiguityExample (data/ambiguity/valid.jsonl):
  - Render as text: memory + input + question + options
  - Append "Answer: " to the prompt
  - Run forward_chunk
  - Predict the option letter (A/B/C/D) as the next token
  - Compare against ex["correct_answer"]

This is the actual accuracy metric for H3 prediction ordering.

Usage:
  uv run python -m model.ambiguity_accuracy --n 200
  uv run python -m model.ambiguity_accuracy --checkpoint checkpoints/phase0.pt --n 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import tiktoken
import torch
import torch.nn.functional as F

from model.checkpoint import load_checkpoint
from model.config import IPCNConfig
from model.ipcn import IPCN


def _render_for_accuracy(ex: dict) -> tuple[str, str]:
    """Render the example so the prompt ends right where the answer letter goes.
    Returns (prompt_text, expected_letter)."""
    lines = ["<|memory|>"]
    for f in ex["phase1_memory"]:
        lines.append(f"- {f}")
    lines.append("<|input|>")
    lines.append(ex["phase2_input"])
    lines.append("<|question|>")
    lines.append(ex["phase3_question"])
    for k, opt in ex["options"].items():
        lines.append(f"{k}. {opt}")
    prompt = "\n".join(lines) + "\n<|answer|>"
    return prompt, ex["correct_answer"]


@torch.no_grad()
def score_one(model: IPCN, enc, prompt: str, correct_letter: str, chunk_length: int) -> bool:
    """Tokenize prompt, predict next token, compare against the first token of correct_letter."""
    model.reset_memory()
    tokens = enc.encode(prompt)
    # Take the LAST chunk_length tokens (so the answer position is at the end)
    if len(tokens) > chunk_length:
        tokens = tokens[-chunk_length:]
    n_valid = len(tokens)
    if n_valid < chunk_length:
        tokens = tokens + [0] * (chunk_length - n_valid)
    in_t = torch.tensor(tokens, dtype=torch.long)
    out = model.forward_chunk(in_t, tau_t=1.0, delta_tau=1.0)
    pred = out.logits[n_valid - 1].argmax().item()
    # Compare against the first token of "A" / "B" / "C" / "D"
    # Note: tiktoken encodes "A" as one token. Use whichever GPT-2 produces.
    target_tokens = {letter: enc.encode(letter) for letter in ["A", "B", "C", "D"]}
    if correct_letter not in target_tokens:
        return False
    target_id = target_tokens[correct_letter][0]
    return pred == target_id


def evaluate_ambiguity(
    model: IPCN,
    n_examples: int = 200,
    path: str = "data/ambiguity/valid.jsonl",
    chunk_length: int | None = None,
) -> dict:
    enc = tiktoken.get_encoding("gpt2")
    if chunk_length is None:
        chunk_length = model.cfg.chunk_length
    examples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n_examples:
                break
            examples.append(json.loads(line))

    correct = 0
    per_family: dict[str, list[int]] = {}
    for ex in examples:
        prompt, expected = _render_for_accuracy(ex)
        ok = score_one(model, enc, prompt, expected, chunk_length)
        if ok:
            correct += 1
        per_family.setdefault(ex["family"], []).append(1 if ok else 0)
    accuracy = correct / max(1, len(examples))

    family_stats = {
        fam: {"n": len(vals), "acc": mean(vals)}
        for fam, vals in per_family.items()
    }
    return {
        "n_examples": len(examples),
        "accuracy": accuracy,
        "by_family": family_stats,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--path", type=str, default="data/ambiguity/valid.jsonl")
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

    print(f"=== Ambiguity accuracy (n={args.n}) ===")
    r = evaluate_ambiguity(model, n_examples=args.n, path=args.path, chunk_length=cfg.chunk_length)
    print(f"Overall accuracy: {r['accuracy']:.4f}")
    print(f"Random baseline: {1.0/4:.4f}")
    for fam, stats in r["by_family"].items():
        print(f"  {fam}: n={stats['n']} acc={stats['acc']:.4f}")


if __name__ == "__main__":
    main()
