"""Evaluation harness for the 7 falsifiable predictions.

For now this implements H1 (D_0 layer-0 memory-swap diff) and H6 (Δτ ablation)
as shape-correct functions. Other predictions require trained model checkpoints
and structured test sets, added incrementally.

H1 (memory must change layer 0): for two memory states M_A and M_B and the
same input X,
    D_0 = ||H_0^A − H_0^B||_F / (||E(X)||_F + ε)
IPCN predicts D_0 > 0.1 on memory-biased inputs. Late retrieval predicts ≈ 0.

H6 (chronometric substrate): for paired streams with identical tokens and two
different Δτ values, compare output distributions on duration-sensitive vs
insensitive questions. Predicts Acc(real) − Acc(ablated) ≥ 0.10 on sensitive,
KL ≤ 0.1 on insensitive.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model.config import IPCNConfig
from model.injection import schedule_lambda_pre
from model.ipcn import IPCN


@dataclass
class MemorySnapshot:
    k: torch.Tensor
    v: torch.Tensor
    q: torch.Tensor
    age: torch.Tensor
    usage: torch.Tensor
    conf: torch.Tensor
    plast: torch.Tensor
    conflict: torch.Tensor
    tau_write: torch.Tensor
    tau_use: torch.Tensor
    chi_slot: torch.Tensor
    z: torch.Tensor


def snapshot_memory(model: IPCN) -> MemorySnapshot:
    m = model.memory
    return MemorySnapshot(
        k=m.k.clone(), v=m.v.clone(), q=m.q.clone(), age=m.age.clone(),
        usage=m.usage.clone(), conf=m.conf.clone(), plast=m.plast.clone(),
        conflict=m.conflict.clone(), tau_write=m.tau_write.clone(),
        tau_use=m.tau_use.clone(), chi_slot=m.chi_slot.clone(),
        z=model.z.clone(),
    )


def restore_memory(model: IPCN, snap: MemorySnapshot):
    m = model.memory
    with torch.no_grad():
        m.k.copy_(snap.k)
        m.v.copy_(snap.v)
        m.q.copy_(snap.q)
        m.age.copy_(snap.age)
        m.usage.copy_(snap.usage)
        m.conf.copy_(snap.conf)
        m.plast.copy_(snap.plast)
        m.conflict.copy_(snap.conflict)
        m.tau_write.copy_(snap.tau_write)
        m.tau_use.copy_(snap.tau_use)
        m.chi_slot.copy_(snap.chi_slot)
        model.z.copy_(snap.z)


def randomize_memory(model: IPCN, seed: int) -> MemorySnapshot:
    """Random-init episodic memory (for synthetic memory-swap tests)."""
    gen = torch.Generator(device=model.memory.k.device)
    gen.manual_seed(seed)
    cfg = model.cfg
    with torch.no_grad():
        k = torch.randn(cfg.n_slots, cfg.d_memory, generator=gen)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        v = torch.randn(cfg.n_slots, cfg.d_memory, generator=gen) * 0.5
        model.memory.k.copy_(k)
        model.memory.v.copy_(v)
        model.memory.conf.fill_(0.7)
        model.memory.plast.fill_(0.3)
        model.memory.usage.zero_()
        model.memory.conflict.zero_()
        model.memory.age.zero_()
        model.memory.tau_write.zero_()
        model.memory.tau_use.zero_()
        model.memory.chi_slot.zero_()
        model.memory.q.zero_()
        model.z.zero_()
    return snapshot_memory(model)


@torch.no_grad()
def compute_D0(
    model: IPCN,
    input_ids: torch.Tensor,
    snap_a: MemorySnapshot,
    snap_b: MemorySnapshot,
    tau_t: float = 1.0,
    delta_tau: float = 1.0,
) -> float:
    """D_0 = ||H_0^A − H_0^B||_F / (||E(X)||_F + eps) for one input.

    H_0 = the input to the core transformer = [prefix; e_tilde].
    """
    eps = 1e-8

    def grab_H0(snap):
        restore_memory(model, snap)
        out = model.forward_chunk(input_ids, tau_t=tau_t, delta_tau=delta_tau)
        e = model.core.embed(input_ids)
        prefix = out.prefix
        lam = schedule_lambda_pre(int(model.train_step.item()), model.cfg)
        e_tilde, _, _ = model.broadcast(e, prefix, model.z, lam)
        H0 = torch.cat([prefix, e_tilde], dim=0)
        return H0, e

    H0_a, e_a = grab_H0(snap_a)
    H0_b, e_b = grab_H0(snap_b)

    num = (H0_a - H0_b).norm()
    denom = e_a.norm() + eps
    return (num / denom).item()


@torch.no_grad()
def H1_synthetic_check(
    model: IPCN,
    n_trials: int = 20,
    seed_base: int = 9000,
    threshold: float = 0.1,
) -> dict:
    """Quick H1 sanity check with random memory states.

    Untrained: D_0 should still be > 0 because random memories produce
    different prefixes; magnitude depends on PFC initialization. After
    Phase 0 training, we expect D_0 > 0.1 on memory-biased inputs.
    """
    cfg = model.cfg
    d0_values = []
    for i in range(n_trials):
        torch.manual_seed(seed_base + i)
        input_ids = torch.randint(0, cfg.vocab_size, (cfg.chunk_length,))
        snap_a = randomize_memory(model, seed=seed_base + 1000 + i)
        snap_b = randomize_memory(model, seed=seed_base + 2000 + i)
        d0 = compute_D0(model, input_ids, snap_a, snap_b)
        d0_values.append(d0)

    d0 = torch.tensor(d0_values)
    return {
        "n_trials": n_trials,
        "D0_mean": float(d0.mean()),
        "D0_std": float(d0.std()),
        "D0_min": float(d0.min()),
        "D0_max": float(d0.max()),
        "pass_threshold": float(threshold),
        "passes": bool(d0.mean() > threshold),
    }


@torch.no_grad()
def H6_chronometric_check(
    model: IPCN,
    n_trials: int = 20,
    seed_base: int = 9100,
    delta_real_minutes: float = 512.0,
    delta_ablated_minutes: float = 1.0,
) -> dict:
    """H6 SYNTHETIC sanity: random memory, random tokens, two delta_tau.

    For real H6 eval against the spec threshold, use H6_pairs_check() which
    uses the engineered chronometric_pairs dataset.
    """
    cfg = model.cfg
    kl_values = []
    for i in range(n_trials):
        torch.manual_seed(seed_base + i)
        input_ids = torch.randint(0, cfg.vocab_size, (cfg.chunk_length,))
        randomize_memory(model, seed=seed_base + 5000 + i)
        snap = snapshot_memory(model)

        restore_memory(model, snap)
        out_real = model.forward_chunk(input_ids, tau_t=1000.0, delta_tau=delta_real_minutes)
        logp_real = F.log_softmax(out_real.logits, dim=-1)

        restore_memory(model, snap)
        out_abl = model.forward_chunk(input_ids, tau_t=1000.0, delta_tau=delta_ablated_minutes)
        logp_abl = F.log_softmax(out_abl.logits, dim=-1)

        kl = 0.5 * (
            (logp_real.exp() * (logp_real - logp_abl)).sum(dim=-1).mean()
            + (logp_abl.exp() * (logp_abl - logp_real)).sum(dim=-1).mean()
        )
        kl_values.append(kl.item())

    kl = torch.tensor(kl_values)
    return {
        "n_trials": n_trials,
        "KL_mean": float(kl.mean()),
        "KL_std": float(kl.std()),
        "KL_min": float(kl.min()),
        "KL_max": float(kl.max()),
        "delta_real_min": delta_real_minutes,
        "delta_ablated_min": delta_ablated_minutes,
        "note": "Pre-training: any nonzero KL means chi_t affects output; magnitude builds with chrono loss training.",
    }


@torch.no_grad()
def H6_pairs_check(
    model: IPCN,
    pairs_path: str = "data/chronometric_pairs/pairs.jsonl",
    n_pairs: int = 100,
) -> dict:
    """H6 eval against the engineered chronometric_pairs dataset.

    Each pair has:
      - visible_text: tokenized as the prompt
      - delta_tau_real_minutes (e.g. ~512)
      - delta_tau_ablated_minutes (constant, e.g. 1)
      - duration_sensitive_q: should change answer under different Δτ
      - duration_insensitive_q: should be invariant

    Loose metric for now: KL between output distributions under real vs ablated
    Δτ. Pass: KL_sensitive >= 0.1, KL_insensitive <= 0.1. (Full spec uses
    answer accuracy delta; we approximate via KL until trained checkpoints
    let us measure real answers.)
    """
    import json
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    cfg = model.cfg

    pairs = []
    with open(pairs_path) as f:
        for i, line in enumerate(f):
            if i >= n_pairs:
                break
            pairs.append(json.loads(line))

    kl_real_vs_ablated = []
    for pair in pairs:
        # Pack visible_text into a single chunk for both arms
        tokens = enc.encode(pair["visible_text"])
        if len(tokens) > cfg.chunk_length:
            tokens = tokens[-cfg.chunk_length:]
        if len(tokens) < cfg.chunk_length:
            tokens = tokens + [0] * (cfg.chunk_length - len(tokens))
        t = torch.tensor(tokens, dtype=torch.long)

        model.reset_memory()
        out_real = model.forward_chunk(
            t, tau_t=float(pair["delta_tau_real_minutes"]),
            delta_tau=float(pair["delta_tau_real_minutes"]),
        )
        logp_real = F.log_softmax(out_real.logits, dim=-1)

        model.reset_memory()
        out_abl = model.forward_chunk(
            t, tau_t=float(pair["delta_tau_ablated_minutes"]),
            delta_tau=float(pair["delta_tau_ablated_minutes"]),
        )
        logp_abl = F.log_softmax(out_abl.logits, dim=-1)

        kl = 0.5 * (
            (logp_real.exp() * (logp_real - logp_abl)).sum(dim=-1).mean()
            + (logp_abl.exp() * (logp_abl - logp_real)).sum(dim=-1).mean()
        )
        kl_real_vs_ablated.append(float(kl.item()))

    return {
        "n_pairs": len(pairs),
        "KL_real_vs_ablated_mean": sum(kl_real_vs_ablated) / max(1, len(kl_real_vs_ablated)),
        "threshold_min": 0.10,
        "passes_distinct": sum(kl_real_vs_ablated) / max(1, len(kl_real_vs_ablated)) >= 0.10,
        "note": "Approximate: full spec measures answer-accuracy delta; we use KL on output distribution.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    torch.manual_seed(0)
    cfg = IPCNConfig()
    model = IPCN(cfg).to(args.device)
    model.train(False)

    print(f"=== H1 sanity check (D_0 memory-swap, UNTRAINED model) ===")
    r1 = H1_synthetic_check(model, n_trials=args.trials, threshold=args.threshold)
    for k, v in r1.items():
        print(f"  {k}: {v}")

    print(f"\n=== H6 sanity check (chronometric delta-tau ablation, UNTRAINED) ===")
    r6 = H6_chronometric_check(model, n_trials=args.trials)
    for k, v in r6.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
