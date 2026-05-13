"""H7 contradiction-pair eval.

For each pair in data/contradiction_pairs/pairs.jsonl:
  - Tokenize memory_a_facts, memory_b_facts, ambiguous_input, explicit_input.
  - For arm "amb_a": reset memory, feed mem_a_facts as a chunk to populate memory bank,
    then forward on ambiguous_input. Capture output distribution p(y|amb, M_A).
  - Repeat for mem_b on ambiguous_input -> p(y|amb, M_B).
  - Same for explicit_input.
  - Compute symmetric KL: KL_amb = symKL(p(amb,A), p(amb,B)); KL_exp = symKL(p(exp,A), p(exp,B)).
  - Pass: KL_amb >= 0.5 AND KL_exp <= 0.1.

For untrained model, expect both KLs near zero (memory not yet wired into output).
Run after Phase 0 to test H7.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import tiktoken
import torch
import torch.nn.functional as F

from model.config import IPCNConfig
from model.ipcn import IPCN


def render_memory_block(facts: list[str]) -> str:
    return "<|memory|>\n" + "\n".join(f"- {f}" for f in facts) + "\n<|input|>\n"


def render_input(text: str) -> str:
    return text


@torch.no_grad()
def feed_memory_then_input(
    model: IPCN,
    enc,
    facts: list[str],
    input_text: str,
    tau_offset: float = 0.0,
    chunk_length: int = 256,
    reset_seed: int | None = None,
) -> torch.Tensor:
    """Reset model, feed memory chunk, then forward input chunk. Return logits
    averaged over the input chunk.

    reset_seed: if set, fix torch.manual_seed before reset so the random
    initial slot keys are reproducible across paired (mem_a/mem_b) calls.
    Without this, each call has different random init and KL between paired
    arms includes init noise — bad for an eval metric.
    """
    if reset_seed is not None:
        torch.manual_seed(reset_seed)
    model.reset_memory()
    mem_text = render_memory_block(facts)
    mem_toks = torch.tensor(enc.encode(mem_text), dtype=torch.long)
    # Pad/trim mem chunk to chunk_length
    if mem_toks.numel() < chunk_length:
        pad = chunk_length - mem_toks.numel()
        mem_toks = F.pad(mem_toks, (0, pad), value=0)
    else:
        mem_toks = mem_toks[:chunk_length]

    # First chunk: feed memory facts. Trigger a write step via update_memory_and_state
    # using a dummy surprise/novelty (we need memory to actually populate).
    mem_out = model.forward_chunk(mem_toks, tau_t=tau_offset, delta_tau=0.0)
    # Quick write step so the memory bank holds these facts
    L = mem_toks.shape[0]
    h_content = mem_out.hidden_last[model.cfg.prefix_length:]
    k_hat = model.memory.W_k_m(h_content)
    k_n = F.normalize(k_hat, dim=-1)
    K_n = F.normalize(model.memory.k, dim=-1)
    sim = k_n @ K_n.t()
    novelty = 1.0 - sim.max(dim=-1).values
    ce = torch.zeros(L)  # neutral surprise
    chi_t = model.chrono(
        tau=torch.tensor([tau_offset]), delta_tau=torch.tensor([0.0]),
    ).squeeze(0)
    model.memory.write(
        h_L=h_content, b=mem_out.b_broadcast, z=model.z, surprise=ce,
        novelty=novelty, u_prefix=torch.zeros(L),
        tau_t=tau_offset, chi_t=chi_t,
    )

    # Second chunk: input
    inp_toks = torch.tensor(enc.encode(input_text), dtype=torch.long)
    if inp_toks.numel() < chunk_length:
        pad = chunk_length - inp_toks.numel()
        inp_toks = F.pad(inp_toks, (0, pad), value=0)
    else:
        inp_toks = inp_toks[:chunk_length]
    in_out = model.forward_chunk(inp_toks, tau_t=tau_offset + 1.0, delta_tau=1.0)
    return in_out.logits.mean(dim=0)


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    log_p_a = F.log_softmax(logits_a, dim=-1)
    log_p_b = F.log_softmax(logits_b, dim=-1)
    p_a = log_p_a.exp()
    p_b = log_p_b.exp()
    kl = 0.5 * ((p_a * (log_p_a - log_p_b)).sum() + (p_b * (log_p_b - log_p_a)).sum())
    return kl.item()


@torch.no_grad()
def H7_check(
    model: IPCN,
    pairs_path: str = "data/contradiction_pairs/pairs.jsonl",
    n_pairs: int = 50,
    chunk_length: int = 256,
) -> dict:
    enc = tiktoken.get_encoding("gpt2")
    pairs = []
    with open(pairs_path) as f:
        for i, line in enumerate(f):
            if i >= n_pairs:
                break
            pairs.append(json.loads(line))

    kl_amb_vals = []
    kl_exp_vals = []
    for i, pair in enumerate(pairs):
        # Use a deterministic seed per pair so the four memory-bank inits
        # below all start identical. KL then reflects ONLY the memory state
        # difference, not init noise.
        seed = 2_000_000 + i
        logits_amb_a = feed_memory_then_input(
            model, enc, pair["memory_a_facts"], pair["ambiguous_input"],
            chunk_length=chunk_length, reset_seed=seed,
        )
        logits_amb_b = feed_memory_then_input(
            model, enc, pair["memory_b_facts"], pair["ambiguous_input"],
            chunk_length=chunk_length, reset_seed=seed,
        )
        logits_exp_a = feed_memory_then_input(
            model, enc, pair["memory_a_facts"], pair["explicit_input"],
            chunk_length=chunk_length, reset_seed=seed,
        )
        logits_exp_b = feed_memory_then_input(
            model, enc, pair["memory_b_facts"], pair["explicit_input"],
            chunk_length=chunk_length, reset_seed=seed,
        )
        kl_amb_vals.append(symmetric_kl(logits_amb_a, logits_amb_b))
        kl_exp_vals.append(symmetric_kl(logits_exp_a, logits_exp_b))

    return {
        "n_pairs": len(pairs),
        "KL_amb_mean": mean(kl_amb_vals),
        "KL_amb_std": stdev(kl_amb_vals) if len(kl_amb_vals) > 1 else 0.0,
        "KL_exp_mean": mean(kl_exp_vals),
        "KL_exp_std": stdev(kl_exp_vals) if len(kl_exp_vals) > 1 else 0.0,
        "threshold_amb": 0.5,
        "threshold_exp": 0.1,
        "passes_amb": mean(kl_amb_vals) >= 0.5,
        "passes_exp": mean(kl_exp_vals) <= 0.1,
        "passes_overall": (mean(kl_amb_vals) >= 0.5) and (mean(kl_exp_vals) <= 0.1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=str, default="data/contradiction_pairs/pairs.jsonl")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    torch.manual_seed(0)
    if args.checkpoint:
        from model.checkpoint import load_checkpoint
        model, _, cfg, _ = load_checkpoint(args.checkpoint, map_location=args.device)
    else:
        cfg = IPCNConfig()
        model = IPCN(cfg).to(args.device)
    model.train(False)

    print(f"=== H7 contradiction-pair eval (n={args.n}) ===")
    r = H7_check(model, pairs_path=args.pairs, n_pairs=args.n, chunk_length=cfg.chunk_length)
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
