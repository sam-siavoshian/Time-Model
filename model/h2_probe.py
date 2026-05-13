"""H2 linear probe: can layer 0/1 hidden states decode memory-conditioned interpretation?

Procedure:
  For each contradiction pair:
    - Reset model. Feed memory_a_facts as chunk 1 (writes to memory bank).
    - Forward on ambiguous_input as chunk 2 with return_hidden_layers=True.
    - Extract H^0 (input to layer 0) and H^1 (output of layer 0) hidden states.
    - Pool to a single vector per (sample, layer) — e.g. mean of content tokens.
    - Repeat with memory_b_facts.
    - Label = 0 (memory A) or 1 (memory B).

  Split into train/test by pair_id (disjoint pairs across splits).
  Train logistic regression on (H_pooled, label) per layer.
  Report accuracy. H2 passes if either layer achieves >= 80%.

For untrained model: probes should already work somewhat because prefix
construction is deterministic-ish given memory bank values. Real claim
strengthens after Phase 0 training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from model.config import IPCNConfig
from model.h7_contradiction import render_memory_block
from model.ipcn import IPCN


@torch.no_grad()
def _feed_memory_then_extract(
    model: IPCN,
    enc,
    facts: list[str],
    input_text: str,
    chunk_length: int = 256,
    tau_t_input: float = 1.0,
    reset_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (H0_pooled, H1_pooled) — mean over content tokens at each layer.

    reset_seed: fix torch.manual_seed before reset for reproducible random
    slot keys across paired calls. Critical for H2: probe accuracy comparison
    between (mem_a, mem_b) requires matched init noise.
    """
    if reset_seed is not None:
        torch.manual_seed(reset_seed)
    model.reset_memory()
    # Feed memory chunk
    mem_text = render_memory_block(facts)
    mem_toks = torch.tensor(enc.encode(mem_text), dtype=torch.long)
    if mem_toks.numel() < chunk_length:
        mem_toks = F.pad(mem_toks, (0, chunk_length - mem_toks.numel()), value=0)
    else:
        mem_toks = mem_toks[:chunk_length]
    mem_out = model.forward_chunk(mem_toks, tau_t=0.0, delta_tau=0.0)
    # Quick write to populate memory bank
    L = mem_toks.shape[0]
    h_content = mem_out.hidden_last[model.cfg.prefix_length:]
    k_hat = model.memory.W_k_m(h_content)
    k_n = F.normalize(k_hat, dim=-1)
    K_n = F.normalize(model.memory.k, dim=-1)
    sim = k_n @ K_n.t()
    novelty = 1.0 - sim.max(dim=-1).values
    chi_t = model.chrono(
        tau=torch.tensor([0.0]), delta_tau=torch.tensor([0.0]),
    ).squeeze(0)
    model.memory.write(
        h_L=h_content, b=mem_out.b_broadcast, z=model.z, surprise=torch.zeros(L),
        novelty=novelty, u_prefix=torch.zeros(L), tau_t=0.0, chi_t=chi_t,
    )

    # Forward on input with return_hidden_layers
    inp_toks = torch.tensor(enc.encode(input_text), dtype=torch.long)
    if inp_toks.numel() < chunk_length:
        inp_toks = F.pad(inp_toks, (0, chunk_length - inp_toks.numel()), value=0)
    else:
        inp_toks = inp_toks[:chunk_length]
    out = model.forward_chunk(
        inp_toks, tau_t=tau_t_input, delta_tau=1.0, return_hidden_layers=True
    )
    K_p = model.cfg.prefix_length
    # H0: input to layer 0 (after injection). Mean over content positions only.
    H0_pooled = out.H0[K_p:].mean(dim=0)                      # (d_model,)
    # H1: output of layer 0 (first entry in hidden_layers).
    if out.hidden_layers and len(out.hidden_layers) > 0:
        H1_pooled = out.hidden_layers[0][K_p:].mean(dim=0)
    else:
        H1_pooled = H0_pooled
    return H0_pooled.detach().cpu(), H1_pooled.detach().cpu()


def _logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_epochs: int = 200,
    lr: float = 0.1,
    weight_decay: float = 0.01,
) -> float:
    """Tiny pure-torch logistic regression. Returns test accuracy."""
    with torch.enable_grad():
        d = X_train.shape[1]
        w = torch.zeros(d, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        Xt = torch.from_numpy(X_train).float()
        yt = torch.from_numpy(y_train).float()
        Xe = torch.from_numpy(X_test).float()
        ye = torch.from_numpy(y_test).float()
        opt = torch.optim.AdamW([w, b], lr=lr, weight_decay=weight_decay)
        for _ in range(n_epochs):
            logits = Xt @ w + b
            loss = F.binary_cross_entropy_with_logits(logits, yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        preds = (Xe @ w + b > 0).float()
        return float((preds == ye).float().mean())


@torch.no_grad()
def H2_probe_check(
    model: IPCN,
    pairs_path: str = "data/contradiction_pairs/pairs.jsonl",
    n_pairs: int = 100,
    test_frac: float = 0.25,
    seed: int = 0,
) -> dict:
    enc = tiktoken.get_encoding("gpt2")
    pairs = []
    with open(pairs_path) as f:
        for i, line in enumerate(f):
            if i >= n_pairs:
                break
            pairs.append(json.loads(line))

    X0_list = []
    X1_list = []
    y_list = []
    for i, pair in enumerate(pairs):
        seed = 3_000_000 + i
        H0_a, H1_a = _feed_memory_then_extract(
            model, enc, pair["memory_a_facts"], pair["ambiguous_input"],
            chunk_length=model.cfg.chunk_length, reset_seed=seed,
        )
        H0_b, H1_b = _feed_memory_then_extract(
            model, enc, pair["memory_b_facts"], pair["ambiguous_input"],
            chunk_length=model.cfg.chunk_length, reset_seed=seed,
        )
        X0_list.append(H0_a.numpy()); y_list.append(0)
        X0_list.append(H0_b.numpy()); y_list.append(1)
        X1_list.append(H1_a.numpy())
        X1_list.append(H1_b.numpy())

    X0 = np.stack(X0_list)
    X1 = np.stack(X1_list)
    y = np.array(y_list, dtype=np.float32)

    # Train/test split by pair_id (each pair contributes 2 rows; split pair-disjoint)
    n_pairs_real = len(pairs)
    n_test_pairs = max(1, int(test_frac * n_pairs_real))
    rng = np.random.RandomState(seed)
    pair_idx = rng.permutation(n_pairs_real)
    test_pairs = set(pair_idx[:n_test_pairs])
    train_pairs = set(pair_idx[n_test_pairs:])
    train_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)
    for i in range(n_pairs_real):
        if i in train_pairs:
            train_mask[2 * i] = True; train_mask[2 * i + 1] = True
        else:
            test_mask[2 * i] = True; test_mask[2 * i + 1] = True

    acc0 = _logistic_regression(X0[train_mask], y[train_mask], X0[test_mask], y[test_mask])
    acc1 = _logistic_regression(X1[train_mask], y[train_mask], X1[test_mask], y[test_mask])

    return {
        "n_pairs": n_pairs_real,
        "n_train_examples": int(train_mask.sum()),
        "n_test_examples": int(test_mask.sum()),
        "probe_acc_H0": acc0,
        "probe_acc_H1": acc1,
        "threshold": 0.80,
        "passes_H0": acc0 >= 0.80,
        "passes_H1": acc1 >= 0.80,
        "passes_either": (acc0 >= 0.80) or (acc1 >= 0.80),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=str, default="data/contradiction_pairs/pairs.jsonl")
    p.add_argument("--n", type=int, default=50)
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

    print(f"=== H2 linear probe (n={args.n} pairs) ===")
    r = H2_probe_check(model, pairs_path=args.pairs, n_pairs=args.n)
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
