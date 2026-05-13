"""Prefix Integrity Test (SPEC.tex §16.4).

Four conditions per ambiguous input:
  1. correct prefix     — built from the example's own memory state
  2. zero prefix        — all-zero prefix vector (memory ablated)
  3. shuffled prefix    — prefix from a DIFFERENT example, same family
  4. adversarial prefix — prefix built from the CONTRADICTING memory state

Predictions:
  - correct prefix improves ambiguity accuracy
  - zero prefix degrades accuracy
  - shuffled prefix hurts, but precision gating should suppress some damage
  - adversarial prefix should be defeated by explicit-evidence inputs
    (link to H7 prediction)

Output: 4 accuracy numbers + comparison.
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
from model.h7_contradiction import feed_memory_then_input, render_memory_block
from model.ipcn import IPCN


@torch.no_grad()
def _forward_with_prefix_override(
    model: IPCN,
    input_ids: torch.Tensor,
    prefix_override: torch.Tensor | None,
    tau_t: float = 1.0,
    delta_tau: float = 1.0,
) -> torch.Tensor:
    """Forward chunk but inject a manually supplied prefix instead of PFC output.

    prefix_override:
      None  -> normal PFC-built prefix
      'zero' (tensor of zeros) -> ablation
      arbitrary tensor (K_p, d_model) -> shuffled / adversarial
    Returns logits.
    """
    if prefix_override is None:
        out = model.forward_chunk(input_ids, tau_t=tau_t, delta_tau=delta_tau)
        return out.logits
    # Use the zero-prefix ablation path with override applied via a hack:
    # we set zero_prefix_for_ablation, which produces a zero prefix, then
    # manually overlay the override values onto model.broadcast and
    # core flow. Simpler: bypass by directly calling components.
    cfg = model.cfg
    device = input_ids.device
    e = model.core.embed(input_ids)
    chi_t = model.chrono(
        tau=torch.tensor([tau_t], device=device, dtype=torch.float32),
        delta_tau=torch.tensor([delta_tau], device=device, dtype=torch.float32),
    ).squeeze(0)
    # Use the override directly as the prefix (skip PFC)
    prefix = prefix_override
    from model.injection import schedule_lambda_pre
    lam_pre = schedule_lambda_pre(int(model.train_step.item()), cfg)
    if cfg.use_route2_broadcast:
        e_tilde, b, _ = model.broadcast(e, prefix, model.z, lam_pre)
    else:
        e_tilde = e
        b = torch.zeros_like(e)
    H0 = torch.cat([prefix, e_tilde], dim=0)
    K_p = cfg.prefix_length
    L = e.shape[0]
    b_full = None
    if cfg.use_route3_lnmod:
        b_full = torch.zeros(K_p + L, cfg.d_model, device=device, dtype=e.dtype)
        b_full[K_p:] = b
    logits, _, _ = model.core(H0, b_full=b_full, return_hidden=False)
    return logits


def _score(logits: torch.Tensor, n_valid: int, correct_letter: str, enc) -> bool:
    pred = logits[n_valid - 1].argmax().item()
    target_id = enc.encode(correct_letter)[0]
    return pred == target_id


@torch.no_grad()
def prefix_integrity_test(
    model: IPCN,
    pairs_path: str = "data/contradiction_pairs/pairs.jsonl",
    n_pairs: int = 50,
    chunk_length: int = 256,
) -> dict:
    """Run all 4 conditions on n_pairs contradiction pairs."""
    enc = tiktoken.get_encoding("gpt2")
    cfg = model.cfg

    pairs = []
    with open(pairs_path) as f:
        for i, line in enumerate(f):
            if i >= n_pairs:
                break
            pairs.append(json.loads(line))

    acc_correct = []
    acc_zero = []
    acc_shuffled = []
    acc_adversarial = []

    # Pre-compute reference prefixes for each pair (build by feeding memory A)
    cached_prefixes_a = []
    cached_prefixes_b = []
    for pair in pairs:
        # Build prefix under memory_a_facts
        model.reset_memory()
        mem_text_a = render_memory_block(pair["memory_a_facts"])
        mem_toks_a = torch.tensor(enc.encode(mem_text_a), dtype=torch.long)
        if mem_toks_a.numel() < chunk_length:
            mem_toks_a = F.pad(mem_toks_a, (0, chunk_length - mem_toks_a.numel()), value=0)
        else:
            mem_toks_a = mem_toks_a[:chunk_length]
        out_a = model.forward_chunk(mem_toks_a, tau_t=0.0, delta_tau=0.0)
        cached_prefixes_a.append(out_a.prefix.detach().clone())

        # Same for memory_b_facts
        model.reset_memory()
        mem_text_b = render_memory_block(pair["memory_b_facts"])
        mem_toks_b = torch.tensor(enc.encode(mem_text_b), dtype=torch.long)
        if mem_toks_b.numel() < chunk_length:
            mem_toks_b = F.pad(mem_toks_b, (0, chunk_length - mem_toks_b.numel()), value=0)
        else:
            mem_toks_b = mem_toks_b[:chunk_length]
        out_b = model.forward_chunk(mem_toks_b, tau_t=0.0, delta_tau=0.0)
        cached_prefixes_b.append(out_b.prefix.detach().clone())

    # Now run 4 conditions per pair (using memory_a as the "correct" reference)
    for i, pair in enumerate(pairs):
        # Encode the ambiguous input
        inp_toks = torch.tensor(enc.encode(pair["ambiguous_input"]), dtype=torch.long)
        if inp_toks.numel() < chunk_length:
            n_valid = inp_toks.numel()
            inp_toks = F.pad(inp_toks, (0, chunk_length - inp_toks.numel()), value=0)
        else:
            inp_toks = inp_toks[:chunk_length]
            n_valid = chunk_length

        correct_letter = pair["correct_under_memory_a"]

        # 1. Correct prefix (memory_a-built)
        logits = _forward_with_prefix_override(model, inp_toks, cached_prefixes_a[i])
        acc_correct.append(_score(logits, n_valid, correct_letter, enc))

        # 2. Zero prefix
        zero_pref = torch.zeros_like(cached_prefixes_a[i])
        logits = _forward_with_prefix_override(model, inp_toks, zero_pref)
        acc_zero.append(_score(logits, n_valid, correct_letter, enc))

        # 3. Shuffled prefix (from a different pair)
        shuf_idx = (i + 7) % len(cached_prefixes_a)
        logits = _forward_with_prefix_override(model, inp_toks, cached_prefixes_a[shuf_idx])
        acc_shuffled.append(_score(logits, n_valid, correct_letter, enc))

        # 4. Adversarial prefix (memory_b — contradicting)
        logits = _forward_with_prefix_override(model, inp_toks, cached_prefixes_b[i])
        acc_adversarial.append(_score(logits, n_valid, correct_letter, enc))

    return {
        "n_pairs": len(pairs),
        "acc_correct":      mean(acc_correct)      if acc_correct      else 0.0,
        "acc_zero":         mean(acc_zero)         if acc_zero         else 0.0,
        "acc_shuffled":     mean(acc_shuffled)     if acc_shuffled     else 0.0,
        "acc_adversarial":  mean(acc_adversarial)  if acc_adversarial  else 0.0,
        "expected_ordering": "correct >= shuffled > adversarial; zero < correct",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
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

    print(f"=== Prefix Integrity Test (n={args.n} pairs) ===")
    r = prefix_integrity_test(model, n_pairs=args.n, chunk_length=cfg.chunk_length)
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
