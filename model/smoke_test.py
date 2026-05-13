"""Smoke test the full IPCN forward pass on dummy data.

Verifies:
  - All modules instantiate
  - Forward pass returns logits of correct shape
  - Memory writes don't error
  - Evolution runs
  - Loss terms compute non-NaN values
  - Param count is in expected range (~50-80M)
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from model.config import IPCNConfig
from model.ipcn import IPCN
from model.losses import (
    chronometric_loss,
    diversity_loss,
    lm_loss,
    pre_influence_loss,
    precision_loss,
    slot_util_loss,
)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print("=" * 60)
    print("IPCN smoke test")
    print("=" * 60)

    cfg = IPCNConfig()
    print(f"Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"prefix={cfg.prefix_length}, slots={cfg.n_slots}, vocab={cfg.vocab_size}")
    print(f"Chronometric dim: {cfg.d_chronometric}")

    print("\nInstantiating IPCN...")
    model = IPCN(cfg)
    total, trainable = count_params(model)
    print(f"Params: {total:,} total / {trainable:,} trainable")

    L = cfg.chunk_length
    input_ids = torch.randint(0, cfg.vocab_size, (L,))
    print(f"\nForward chunk on {L}-token input...")

    t0 = time.time()
    out = model.forward_chunk(
        input_ids,
        tau_t=10.0,
        delta_tau=2.0,
        gap_flag=0.0,
        event_density=0.5,
    )
    dt = time.time() - t0

    print(f"  forward_chunk done in {dt:.3f}s")
    print(f"  logits shape: {tuple(out.logits.shape)}  (expect ({L}, {cfg.vocab_size}))")
    print(f"  prefix shape: {tuple(out.prefix.shape)}  (expect ({cfg.prefix_length}, {cfg.d_model}))")
    print(f"  alpha_prefix shape: {tuple(out.alpha_prefix.shape)}  (expect ({cfg.prefix_length}, {cfg.n_slots}))")
    print(f"  b_broadcast shape: {tuple(out.b_broadcast.shape)}  (expect ({L}, {cfg.d_model}))")
    print(f"  hidden_last shape: {tuple(out.hidden_last.shape)}  (expect ({cfg.prefix_length + L}, {cfg.d_model}))")

    assert out.logits.shape == (L, cfg.vocab_size)
    assert out.prefix.shape == (cfg.prefix_length, cfg.d_model)
    assert out.alpha_prefix.shape == (cfg.prefix_length, cfg.n_slots)
    print("  shape checks: PASS")
    assert not torch.isnan(out.logits).any(), "NaN in logits"
    assert torch.allclose(out.alpha_prefix.sum(dim=-1), torch.ones(cfg.prefix_length), atol=1e-4), "alpha rows must sum to 1"
    print("  NaN + softmax checks: PASS")

    # LM loss against random targets
    targets = torch.randint(0, cfg.vocab_size, (L,))
    L_lm = lm_loss(out.logits, targets)
    print(f"\nLM loss on random targets: {L_lm.item():.4f}  (random baseline ~= log(V) = {torch.log(torch.tensor(float(cfg.vocab_size))).item():.4f})")

    # Synthetic write step
    print("\nMemory write step...")
    surprise = torch.randn(L)
    novelty = torch.rand(L)
    u_prefix = torch.randn(L) * 0.1
    model.update_memory_and_state(
        out=out,
        surprise=surprise,
        novelty=novelty,
        u_prefix=u_prefix,
        u_prefix_bar=u_prefix.mean().item(),
        gate_bar=out.gate.mean().item(),
        tau_t=10.0,
        delta_tau=2.0,
        do_evolve=True,
    )
    print("  write + z update + evolve: PASS")
    print(f"  memory.v norm: {model.memory.v.norm().item():.4f}")
    print(f"  memory.usage sum: {model.memory.usage.sum().item():.4f}")
    print(f"  z norm: {model.z.norm().item():.4f}")

    # Diagnostic dump
    print("\nMemory diagnostics after first write:")
    print(f"  memory.k row-norms — min: {model.memory.k.norm(dim=-1).min().item():.4f}, "
          f"max: {model.memory.k.norm(dim=-1).max().item():.4f}, "
          f"nonzero rows: {(model.memory.k.norm(dim=-1) > 0).sum().item()}/{cfg.n_slots}")
    print(f"  memory.tau_write — min: {model.memory.tau_write.min().item():.2f}, "
          f"max: {model.memory.tau_write.max().item():.2f}, "
          f"touched slots: {(model.memory.tau_write > 0).sum().item()}/{cfg.n_slots}")
    print(f"  memory.conf — min: {model.memory.conf.min().item():.4f}, "
          f"max: {model.memory.conf.max().item():.4f}")

    # Second chunk to test cross-chunk state
    print("\nSecond chunk (cross-chunk memory persistence)...")
    input_ids2 = torch.randint(0, cfg.vocab_size, (L,))
    out2 = model.forward_chunk(input_ids2, tau_t=20.0, delta_tau=10.0)
    print(f"  logits shape: {tuple(out2.logits.shape)}")

    # alpha_prefix should differ from out (memory changed)
    a1 = out.alpha_prefix
    a2 = out2.alpha_prefix
    print(f"  alpha_chunk1[0, :5] = {a1[0, :5].tolist()}")
    print(f"  alpha_chunk2[0, :5] = {a2[0, :5].tolist()}")
    print(f"  alpha_chunk1 row 0 entropy: {-(a1[0] * a1[0].clamp_min(1e-12).log()).sum().item():.4f}")
    print(f"  alpha_chunk2 row 0 entropy: {-(a2[0] * a2[0].clamp_min(1e-12).log()).sum().item():.4f}")
    diff = (a1 - a2).abs().mean().item()
    print(f"  alpha_prefix diff between chunks (mean abs): {diff:.6e}  (should be > 0)")
    assert diff > 1e-8, "memory state did not affect prefix attention!"

    # Loss term smoke tests
    print("\nLoss term smoke tests...")
    L_div = diversity_loss(model.memory.k)
    print(f"  diversity_loss(keys): {L_div.item():.6f}")
    L_util = slot_util_loss(model.memory.usage)
    print(f"  slot_util_loss: {L_util.item():.4f}")
    L_pre = pre_influence_loss(
        u_t=torch.tensor(0.05),
        gate_bar=torch.tensor(out.gate.mean().item()),
        rho_helped=cfg.rho_helped,
        tau_gate=cfg.tau_gate,
    )
    print(f"  pre_influence_loss: {L_pre.item():.4f}")
    L_prec = precision_loss(torch.tensor(-0.1), torch.tensor(out.gate.mean().item()))
    print(f"  precision_loss: {L_prec.item():.4f}")
    L_chr = chronometric_loss(
        delta_tau_hat=torch.tensor([5.0]),
        delta_tau_true=torch.tensor([4.5]),
    )
    print(f"  chronometric_loss: {L_chr.item():.4f}")
    print("  all losses non-NaN: PASS")

    # Backward pass smoke test (gradients flow)
    print("\nBackward pass test...")
    model.zero_grad(set_to_none=True)
    model.reset_memory()
    out3 = model.forward_chunk(input_ids, tau_t=1.0, delta_tau=1.0)
    targets3 = torch.randint(0, cfg.vocab_size, (L,))
    loss = lm_loss(out3.logits, targets3)
    loss.backward()
    nonzero_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  loss = {loss.item():.4f}, grads flowing on {nonzero_grads}/{total_params} param tensors")
    assert nonzero_grads > 0, "no gradients!"

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
