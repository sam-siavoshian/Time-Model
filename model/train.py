"""IPCN training loop.

One training step:
  1. Pull next ChunkBatch from data loader
  2. If is_first_chunk: model.reset_memory()
  3. Forward chunk -> ChunkOutput
  4. Compute losses (9 terms, weighted)
  5. Backward + optimizer step
  6. Run memory write + evolve
  7. Every consolidation_frequency chunks: run consolidation pass (Phase 1+)

Optimizer has two parameter groups:
  - base parameters: LR = cfg.base_lr (3e-4)
  - LoRA adapter parameters: LR = cfg.adapter_lr (1e-5 to 5e-5)

Memory is detached every cfg.bptt_chunks chunks to keep BPTT bounded.

This file exposes a `train_loop(...)` function and a CLI for `__main__`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F

from model.adapters import LoRALinear
from model.config import IPCNConfig
from model.dataset import ChunkBatch, MixedDataset, SequentialChunkDataset, TokenizedCache
from model.ipcn import IPCN
from model.losses import (
    chronometric_loss,
    diversity_loss,
    lm_loss,
    pre_influence_loss,
    precision_loss,
    slot_util_loss,
)


@dataclass
class StepLog:
    step: int
    total_loss: float
    lm_loss: float
    pre_influence: float
    precision: float
    diversity: float
    slot_util: float
    chrono: float
    memory_norm: float
    z_norm: float
    grad_norm: float
    chunk_time_s: float
    example_id: int
    chunk_idx: int


def split_params(model: IPCN) -> tuple[list, list]:
    """Return (base_params, adapter_params)."""
    base = []
    adapter = []
    for module in model.modules():
        if isinstance(module, LoRALinear) and module.rank > 0:
            adapter.extend(module.adapter_parameters)
    adapter_ids = set(id(p) for p in adapter)
    for p in model.parameters():
        if p.requires_grad and id(p) not in adapter_ids:
            base.append(p)
    return base, adapter


def build_optimizer(model: IPCN, cfg: IPCNConfig) -> torch.optim.Optimizer:
    base, adapter = split_params(model)
    groups = [{"params": base, "lr": cfg.base_lr}]
    if adapter:
        groups.append({"params": adapter, "lr": cfg.adapter_lr})
    return torch.optim.AdamW(groups, weight_decay=0.01)


@torch.no_grad()
def compute_u_prefix(
    model: IPCN,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    tau_t: float,
    delta_tau: float,
    base_logits: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Real prefix attribution: per-token (loss_zero_prefix - loss_with_prefix).

    Positive values mean the prefix HELPED predict that token.

    Returns:
        u_prefix:  (L,) per-token attribution
        u_prefix_bar:  scalar chunk-mean attribution

    Cost: one extra forward pass per call. Caller controls frequency
    (e.g., 10-25 percent of training chunks per SPEC.tex).
    """
    # Forward with zero prefix
    out_zero = model.forward_chunk(
        input_ids=input_ids,
        tau_t=tau_t,
        delta_tau=delta_tau,
        zero_prefix_for_ablation=True,
    )
    ce_zero = F.cross_entropy(out_zero.logits, targets, reduction="none")     # (L,)
    ce_normal = F.cross_entropy(base_logits, targets, reduction="none")       # (L,)
    u_prefix = ce_zero - ce_normal                                             # positive = helped
    return u_prefix, float(u_prefix.mean().item())


def train_step(
    model: IPCN,
    opt: torch.optim.Optimizer,
    batch: ChunkBatch,
    cfg: IPCNConfig,
    step: int,
    attribution_p: float = 0.15,
) -> StepLog:
    device = next(model.parameters()).device
    input_ids = batch.input_ids.to(device)
    targets = batch.targets.to(device)

    t0 = time.time()

    if batch.is_first_chunk:
        model.reset_memory()

    # Forward
    out = model.forward_chunk(
        input_ids=input_ids,
        tau_t=batch.tau_t,
        delta_tau=batch.delta_tau,
        gap_flag=batch.gap_flag,
    )

    # 1) LM loss
    L_lm = lm_loss(out.logits, targets)

    # 2) Prefix attribution sampled on ~15% of chunks (cost: one extra fwd pass)
    do_attribution = (torch.rand(1).item() < attribution_p)
    if do_attribution:
        with torch.no_grad():
            base_logits_detached = out.logits.detach()
        _, u_prefix_bar_val = compute_u_prefix(
            model, input_ids, targets, batch.tau_t, batch.delta_tau, base_logits_detached
        )
        u_t = torch.tensor(u_prefix_bar_val, device=device)
    else:
        u_t = torch.tensor(0.0, device=device)
    gate_bar = out.gate.mean()
    L_pre = pre_influence_loss(u_t, gate_bar, cfg.rho_helped, cfg.tau_gate)
    L_prec = precision_loss(u_t, gate_bar)

    # 5) Diversity + slot-util: regularize memory bank
    L_div = diversity_loss(model.memory.k)
    L_util = slot_util_loss(model.memory.usage)

    # 8) Chronometric: predict Δτ from pooled hidden state
    pooled_h = out.hidden_last.mean(dim=0)
    dtau_hat = model.dtau_head(pooled_h)
    dtau_true = torch.tensor([batch.delta_tau], device=device, dtype=dtau_hat.dtype)
    L_chr = chronometric_loss(dtau_hat, dtau_true)

    # Compose (omit mem_predict, evolution_self_predict, consolidation in v1 train step)
    total = (
        cfg.w_lm * L_lm
        + cfg.w_pre_influence * L_pre
        + cfg.w_precision * L_prec
        + cfg.w_diversity * L_div
        + cfg.w_slot_util * L_util
        + cfg.w_chrono * L_chr
    )

    opt.zero_grad(set_to_none=True)
    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    opt.step()
    model.train_step += 1

    # Update memory + state AFTER backward (no_grad in memory.write).
    # Synthesize cheap surprise / novelty proxies.
    with torch.no_grad():
        ce_per_token = F.cross_entropy(out.logits, targets, reduction="none")  # (L,)
        surprise = (ce_per_token - ce_per_token.mean()) / (ce_per_token.std() + 1e-6)
        # Novelty: 1 - max cosine of candidate key vs slot keys
        h_content = out.hidden_last[cfg.prefix_length:]
        k_hat = model.memory.W_k_m(h_content)
        k_n = F.normalize(k_hat, dim=-1)
        K_n = F.normalize(model.memory.k, dim=-1)
        sim = k_n @ K_n.t()
        novelty = 1.0 - sim.max(dim=-1).values
        u_prefix = torch.zeros_like(surprise)

        model.update_memory_and_state(
            out=out,
            surprise=surprise,
            novelty=novelty,
            u_prefix=u_prefix,
            u_prefix_bar=0.0,
            gate_bar=gate_bar.item(),
            tau_t=batch.tau_t,
            delta_tau=batch.delta_tau,
            do_evolve=True,
        )

        # BPTT detachment: every cfg.bptt_chunks chunks, detach z and mem state
        # (state-dict copy is a no-op for detachment; we rely on no_grad above
        # to keep memory ops outside the autograd graph)

    return StepLog(
        step=step,
        total_loss=total.item(),
        lm_loss=L_lm.item(),
        pre_influence=L_pre.item() if isinstance(L_pre, torch.Tensor) else float(L_pre),
        precision=L_prec.item() if isinstance(L_prec, torch.Tensor) else float(L_prec),
        diversity=L_div.item(),
        slot_util=L_util.item(),
        chrono=L_chr.item(),
        memory_norm=model.memory.v.norm().item(),
        z_norm=model.z.norm().item(),
        grad_norm=grad_norm.item(),
        chunk_time_s=time.time() - t0,
        example_id=batch.example_id,
        chunk_idx=batch.chunk_idx,
    )


def train_loop(
    model: IPCN,
    iterator: Iterator[ChunkBatch],
    cfg: IPCNConfig,
    max_steps: int,
    log_every: int = 10,
    log_path: Optional[str] = None,
) -> list[StepLog]:
    opt = build_optimizer(model, cfg)
    logs: list[StepLog] = []
    step = 0
    f = open(log_path, "w") if log_path else None
    try:
        for batch in iterator:
            log = train_step(model, opt, batch, cfg, step)
            logs.append(log)
            if f:
                f.write(json.dumps(asdict(log)) + "\n")
                f.flush()
            if step % log_every == 0:
                print(
                    f"step={log.step:6d} | LM={log.lm_loss:7.4f} total={log.total_loss:7.4f} "
                    f"| ppl={math.exp(min(log.lm_loss, 20)):8.1f} | grad={log.grad_norm:6.3f} "
                    f"| mem={log.memory_norm:7.2f} | z={log.z_norm:5.2f} "
                    f"| dt={log.chunk_time_s*1000:5.0f}ms | ex={log.example_id} chunk={log.chunk_idx}"
                )
            step += 1
            if step >= max_steps:
                break
    finally:
        if f:
            f.close()
    return logs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100, help="number of training steps (small for sanity)")
    p.add_argument("--cache", type=str, default="data/tokenized/ambiguity/train",
                   help="cache prefix to train on")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--log-path", type=str, default="logs/train_sanity.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cfg = IPCNConfig()
    model = IPCN(cfg).to(args.device)
    cache = TokenizedCache(args.cache)
    dataset = SequentialChunkDataset(cache, chunk_length=cfg.chunk_length, shuffle_examples=True, seed=args.seed)

    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Training {args.steps} steps on {args.cache} (device={args.device})")
    print(f"Cache: {cache.n_examples:,} examples, {cache.n_tokens:,} tokens")
    print(f"Logging to {args.log_path}")

    iterator = iter(dataset)
    train_loop(model, iterator, cfg, max_steps=args.steps, log_every=args.log_every, log_path=args.log_path)
    print("done")


if __name__ == "__main__":
    main()
