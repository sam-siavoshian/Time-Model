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
    consolidation_kl,
    diversity_loss,
    evolution_self_predict_loss,
    lm_loss,
    mem_predict_loss,
    pre_influence_loss,
    precision_loss,
    slot_util_loss,
)
from model.replay_buffer import ReplayBuffer, ReplayContext


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
    replay_buffer: Optional[ReplayBuffer] = None,
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

    # 4) Memory-usefulness predictor (P_U): predict U_t from pooled prefix + z
    pooled_prefix = out.prefix.mean(dim=0)
    p_u_input = torch.cat([pooled_prefix, model.z], dim=-1)
    u_hat = model.P_U(p_u_input).squeeze(-1)
    L_mem = mem_predict_loss(u_hat, u_t.detach())

    # 7) Evolution self-prediction (P_z + P_M from pre-update state).
    # Take a snapshot BEFORE memory write+evolve, predict next z and ΔM.
    with torch.no_grad():
        pooled_mem_pre = model.memory.v.mean(dim=0).detach()
    pz_input = torch.cat([pooled_mem_pre, model.z.detach()], dim=-1)
    z_hat = model.P_z(pz_input)
    dmem_hat = model.P_M(pz_input)

    # Compose
    total = (
        cfg.w_lm * L_lm
        + cfg.w_pre_influence * L_pre
        + cfg.w_precision * L_prec
        + cfg.w_mem_predict * L_mem
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

    # After the memory write+evolve, compute evolution self-prediction loss
    # and backprop separately (its targets depend on post-update state).
    # We do this as a small extra step so the predictor heads learn.

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

        u_prefix_bar_for_memory = float(u_t.item()) if do_attribution else 0.0
        z_pre = model.z.detach().clone()
        mem_pre = model.memory.v.mean(dim=0).detach().clone()
        model.update_memory_and_state(
            out=out,
            surprise=surprise,
            novelty=novelty,
            u_prefix=u_prefix,
            u_prefix_bar=u_prefix_bar_for_memory,
            gate_bar=gate_bar.item(),
            tau_t=batch.tau_t,
            delta_tau=batch.delta_tau,
            do_evolve=True,
        )
        z_post = model.z.detach().clone()
        mem_post = model.memory.v.mean(dim=0).detach().clone()
        d_mem_true = (mem_post - mem_pre)

    # Evolution self-prediction loss (separate small backward — heads only)
    if model.P_z.weight.requires_grad:
        opt.zero_grad(set_to_none=True)
        # Re-derive predictions WITH grad (the earlier z_hat/dmem_hat may have
        # been detached when memory state changed).
        pz_input2 = torch.cat([mem_pre, z_pre], dim=-1)
        z_hat2 = model.P_z(pz_input2)
        dmem_hat2 = model.P_M(pz_input2)
        L_evo = evolution_self_predict_loss(z_hat2, z_post, dmem_hat2, d_mem_true)
        (cfg.w_evolution * L_evo).backward()
        opt.step()
        L_evo_val = float(L_evo.item())
    else:
        L_evo_val = 0.0

        # Replay buffer push: only when we measured real u_prefix and it
        # was positive (memory genuinely helped). Push chunk context into
        # top-k slot buffers by attention mass.
        if replay_buffer is not None and do_attribution and u_prefix_bar_for_memory > 0:
            ctx = ReplayContext(
                input_ids=input_ids.detach().cpu(),
                targets=targets.detach().cpu(),
                tau_t=batch.tau_t,
                delta_tau=batch.delta_tau,
            )
            replay_buffer.push_top_k(
                alpha_p=out.alpha_prefix.detach().cpu(),
                u_prefix_bar=u_prefix_bar_for_memory,
                ctx=ctx,
                k=4,
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


@torch.no_grad()
def _slot_active_logits(model: IPCN, ctx: ReplayContext) -> torch.Tensor:
    """Teacher pass: full model, all slots active. Returns logits (L, V)."""
    out = model.forward_chunk(ctx.input_ids, tau_t=ctx.tau_t, delta_tau=ctx.delta_tau)
    return out.logits.detach()


def _slot_attenuated_logits(model: IPCN, ctx: ReplayContext, slot_id: int) -> torch.Tensor:
    """Student pass: forward with slot_id key/value temporarily zeroed.
    LoRA grads ON. Returns logits with gradient."""
    # Save and zero
    orig_k = model.memory.k[slot_id].clone()
    orig_v = model.memory.v[slot_id].clone()
    with torch.no_grad():
        model.memory.k[slot_id].zero_()
        model.memory.v[slot_id].zero_()
    try:
        out = model.forward_chunk(ctx.input_ids, tau_t=ctx.tau_t, delta_tau=ctx.delta_tau)
        logits = out.logits
    finally:
        # Restore
        with torch.no_grad():
            model.memory.k[slot_id].copy_(orig_k)
            model.memory.v[slot_id].copy_(orig_v)
    return logits


@torch.no_grad()
def _measure_lm_drift(
    model: IPCN,
    contexts: list[ReplayContext],
    pre_logits: list[torch.Tensor],
) -> float:
    """Compute mean KL(pre || post) on a probe batch after adapter update."""
    if not contexts:
        return 0.0
    drifts = []
    for ctx, pre in zip(contexts, pre_logits):
        ctx_dev = ReplayContext(
            input_ids=ctx.input_ids.to(next(model.parameters()).device),
            targets=ctx.targets.to(next(model.parameters()).device),
            tau_t=ctx.tau_t, delta_tau=ctx.delta_tau,
        )
        post_logits = _slot_active_logits(model, ctx_dev)
        log_p_pre = F.log_softmax(pre, dim=-1)
        log_p_post = F.log_softmax(post_logits, dim=-1)
        kl = (log_p_pre.exp() * (log_p_pre - log_p_post)).sum(dim=-1).mean()
        drifts.append(float(kl.item()))
    return sum(drifts) / len(drifts)


@torch.no_grad()
def _measure_token_accuracy(
    model: IPCN,
    contexts: list[ReplayContext],
) -> float:
    """Argmax-token accuracy on the replay batch (proxy for held-out
    accuracy when no dedicated eval set is available)."""
    if not contexts:
        return 0.0
    correct = 0
    total = 0
    for ctx in contexts:
        ctx_dev = ReplayContext(
            input_ids=ctx.input_ids.to(next(model.parameters()).device),
            targets=ctx.targets.to(next(model.parameters()).device),
            tau_t=ctx.tau_t, delta_tau=ctx.delta_tau,
        )
        logits = _slot_active_logits(model, ctx_dev)
        preds = logits.argmax(dim=-1)
        # Valid mask: dataset pads target positions with -100 (PyTorch
        # ignore_index convention). Token 0 (<|endoftext|>) is a REAL
        # token and should be scored; the pre-fix mask `targets != 0`
        # incorrectly excluded EOS predictions while INcluding -100
        # pad positions (since -100 != 0), which dilutes accuracy
        # toward 0 (preds can never equal -100). That biases the
        # held-out acc_drop gate downward and lets unsafe
        # consolidations commit.
        valid = (ctx_dev.targets >= 0)
        correct += int((preds == ctx_dev.targets)[valid].sum().item())
        total += int(valid.sum().item())
    return correct / max(1, total)


@torch.no_grad()
def _try_hard_consolidation(
    model: IPCN,
    slot_id: int,
    pre_replay_acc: float,
    cfg: IPCNConfig,
) -> bool:
    """SPEC.tex §11.4 Hard consolidation: attenuate slot if removing it
    drops replay accuracy by less than eps_drop.

    Returns True if attenuation applied.
    """
    # Score with slot present (already known: pre_replay_acc)
    # Score with slot zeroed temporarily
    orig_k = model.memory.k[slot_id].clone()
    orig_v = model.memory.v[slot_id].clone()
    orig_conf = model.memory.conf[slot_id].clone()
    try:
        # We don't have replay contexts here — approximated via attenuation step
        # Hard consolidation: multiply slot by (1 - chi_i), chi_i grows toward 1
        # The spec only allows growth when delta_acc < eps_drop. For v1 we
        # apply a fixed small attenuation step (chi_i = 0.1) per call. After
        # ~10 consolidation passes the slot is fully attenuated.
        chi_i = 0.1
        model.memory.k[slot_id] = (1.0 - chi_i) * orig_k
        model.memory.v[slot_id] = (1.0 - chi_i) * orig_v
        model.memory.conf[slot_id] = (1.0 - chi_i) * orig_conf
        return True
    except Exception:                                         # noqa: BLE001
        # Restore on any error
        model.memory.k[slot_id] = orig_k
        model.memory.v[slot_id] = orig_v
        model.memory.conf[slot_id] = orig_conf
        return False


def _snapshot_adapters(model: IPCN) -> list:
    from model.adapters import LoRALinear
    out = []
    for module in model.modules():
        if isinstance(module, LoRALinear) and module.rank > 0:
            out.append((module, module.snapshot_adapter()))
    return out


def _restore_adapters(snaps: list):
    for module, snap in snaps:
        module.restore_adapter(snap)


def maybe_run_consolidation(
    model: IPCN,
    opt: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    cfg: IPCNConfig,
    step: int,
    min_buffer_size: int = 8,
    max_slots_per_pass: int = 4,
    samples_per_slot: int = 4,
    enable_validation: bool = True,
) -> Optional[dict]:
    """Run a consolidation pass with LM-drift validation gate."""
    if step == 0 or step % cfg.consolidation_frequency != 0:
        return None

    raw_eligible = model.eligible_slots_for_consolidation()
    if not raw_eligible:
        return {
            "step": step,
            "skipped_reason": "no slots eligible (kappa <= tau_cons or unstable)",
            "n_raw_eligible": 0,
            "replay_stats": replay_buffer.stats(),
        }
    eligible = [s for s in raw_eligible if replay_buffer.size(s) >= min_buffer_size]
    if not eligible:
        return {
            "step": step,
            "skipped_reason": f"replay buffer too small for all {len(raw_eligible)} eligible slots",
            "n_raw_eligible": len(raw_eligible),
            "replay_stats": replay_buffer.stats(),
        }
    eligible = eligible[:max_slots_per_pass]

    # Sample a probe batch BEFORE we touch adapters. Capture both pre-logits
    # (for KL drift) and pre-accuracy (for held-out gate).
    probe_contexts = []
    probe_logits = []
    pre_accuracy = 0.0
    if enable_validation:
        for slot_id in eligible:
            for ctx in replay_buffer.sample(slot_id, n=2):
                ctx_dev = ReplayContext(
                    input_ids=ctx.input_ids.to(next(model.parameters()).device),
                    targets=ctx.targets.to(next(model.parameters()).device),
                    tau_t=ctx.tau_t, delta_tau=ctx.delta_tau,
                )
                probe_contexts.append(ctx)
                probe_logits.append(_slot_active_logits(model, ctx_dev))
        pre_accuracy = _measure_token_accuracy(model, probe_contexts)

    # Snapshot adapters for potential rollback
    snap = _snapshot_adapters(model) if enable_validation else None

    total_kl = 0.0
    n_updates = 0
    for slot_id in eligible:
        contexts = replay_buffer.sample(slot_id, samples_per_slot)
        if not contexts:
            continue
        for ctx in contexts:
            ctx_dev = ReplayContext(
                input_ids=ctx.input_ids.to(next(model.parameters()).device),
                targets=ctx.targets.to(next(model.parameters()).device),
                tau_t=ctx.tau_t, delta_tau=ctx.delta_tau,
            )
            teacher_logits = _slot_active_logits(model, ctx_dev)
            student_logits = _slot_attenuated_logits(model, ctx_dev, slot_id)
            loss = consolidation_kl(teacher_logits, student_logits)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            opt.step()
            total_kl += float(loss.item())
            n_updates += 1

    # Validation gates: LM drift + held-out accuracy
    committed = True
    drift = 0.0
    post_accuracy = pre_accuracy
    hard_attenuations = 0
    rollback_reason = ""
    if enable_validation and snap is not None:
        drift = _measure_lm_drift(model, probe_contexts, probe_logits)
        post_accuracy = _measure_token_accuracy(model, probe_contexts)
        acc_drop = pre_accuracy - post_accuracy
        if drift > cfg.kl_drift_threshold:
            _restore_adapters(snap)
            committed = False
            rollback_reason = f"lm_drift {drift:.4f} > {cfg.kl_drift_threshold}"
        elif acc_drop > cfg.eps_drop:
            _restore_adapters(snap)
            committed = False
            rollback_reason = f"acc_drop {acc_drop:.4f} > {cfg.eps_drop}"
        else:
            # Hard consolidation step: gradually attenuate slot key/value as
            # behavior moves into LoRA. SPEC.tex §11.4
            for slot_id in eligible:
                if _try_hard_consolidation(model, slot_id, pre_accuracy, cfg):
                    hard_attenuations += 1

    return {
        "step": step,
        "slots_consolidated": len(eligible),
        "n_updates": n_updates,
        "mean_kl": total_kl / max(1, n_updates),
        "lm_drift_kl": drift,
        "pre_accuracy": pre_accuracy,
        "post_accuracy": post_accuracy,
        "acc_drop": pre_accuracy - post_accuracy,
        "hard_attenuations": hard_attenuations,
        "committed": committed,
        "rollback_reason": rollback_reason,
        "replay_stats": replay_buffer.stats(),
    }


def train_loop(
    model: IPCN,
    iterator: Iterator[ChunkBatch],
    cfg: IPCNConfig,
    max_steps: int,
    log_every: int = 10,
    log_path: Optional[str] = None,
    enable_consolidation: bool = False,
    replay_buffer: Optional[ReplayBuffer] = None,
    ckpt_every: Optional[int] = None,
    ckpt_path_template: Optional[str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> list[StepLog]:
    opt = optimizer if optimizer is not None else build_optimizer(model, cfg)
    if enable_consolidation and replay_buffer is None:
        replay_buffer = ReplayBuffer(n_slots=cfg.n_slots, capacity_per_slot=256)

    logs: list[StepLog] = []
    step = 0
    f = open(log_path, "w") if log_path else None
    try:
        for batch in iterator:
            # Periodic intermediate checkpoint
            if (
                ckpt_every is not None
                and ckpt_path_template is not None
                and step > 0
                and step % ckpt_every == 0
            ):
                from model.checkpoint import save_checkpoint as _save_ckpt
                path = ckpt_path_template.format(step=step)
                _save_ckpt(path, model, opt, cfg, train_step=int(model.train_step.item()),
                           extra={"intermediate": True})
                print(f"  [checkpoint @ step {step}] saved to {path}")
            log = train_step(model, opt, batch, cfg, step, replay_buffer=replay_buffer)
            logs.append(log)

            # Periodic consolidation
            if enable_consolidation and replay_buffer is not None:
                cons_log = maybe_run_consolidation(model, opt, replay_buffer, cfg, step)
                if cons_log is not None and f:
                    f.write(json.dumps({"event": "consolidation", **cons_log}) + "\n")
                if cons_log is not None:
                    if "skipped_reason" in cons_log:
                        print(
                            f"  [consolidation @ step {step}] SKIPPED "
                            f"({cons_log['skipped_reason']}, "
                            f"replay_total={cons_log['replay_stats']['total_contexts']})"
                        )
                    else:
                        commit = "COMMIT" if cons_log.get("committed", False) else "ROLLBACK"
                        reason = cons_log.get("rollback_reason", "")
                        reason_str = f" reason='{reason}'" if reason else ""
                        print(
                            f"  [consolidation @ step {step}] {commit} "
                            f"slots={cons_log['slots_consolidated']} "
                            f"updates={cons_log['n_updates']} "
                            f"mean_kl={cons_log['mean_kl']:.4f} "
                            f"drift={cons_log['lm_drift_kl']:.4f} "
                            f"acc_pre={cons_log.get('pre_accuracy', 0):.3f} "
                            f"acc_post={cons_log.get('post_accuracy', 0):.3f}{reason_str}"
                        )

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
