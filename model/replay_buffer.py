"""Per-slot replay buffer for consolidation.

After each training chunk where slot i's usefulness u_prefix_bar > 0, we
optionally push the chunk's context (input_ids, targets, tau_t, delta_tau)
into slot i's buffer. When that slot becomes eligible for consolidation,
the buffer is sampled to build the teacher/student replay batches.

Storage is a simple per-slot fixed-capacity ring buffer. CPU-side. Tensors
are stored as int64 / float for portability.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ReplayContext:
    input_ids: torch.Tensor                                   # (L,) int64
    targets: torch.Tensor                                     # (L,) int64
    tau_t: float
    delta_tau: float


class ReplayBuffer:
    """Ring buffer per slot."""

    def __init__(self, n_slots: int, capacity_per_slot: int = 256, seed: int = 0):
        self.n_slots = n_slots
        self.capacity = capacity_per_slot
        self.buffers: list[deque[ReplayContext]] = [
            deque(maxlen=capacity_per_slot) for _ in range(n_slots)
        ]
        self.rng = random.Random(seed)

    def push(self, slot_id: int, ctx: ReplayContext):
        self.buffers[slot_id].append(ctx)

    def push_top_k(
        self,
        alpha_p: torch.Tensor,                                # (K_p, N_m)
        u_prefix_bar: float,
        ctx: ReplayContext,
        k: int = 4,
        threshold: float = 0.01,
    ):
        """Push ctx into buffers of the top-k slots that ACTUALLY attended.

        Previously: top-k by mass with only a global `max(mass) >= threshold`
        gate. If only 1 of 256 slots had real attention, we still pushed
        the context into 4 slot buffers -- 3 of them recording a context
        they never attended. When consolidation later sampled those 3
        slots, the teacher/student distillation tried to remove a slot
        the model never used for that input. Either:
          - the student trivially matches the teacher because removing
            an unused slot does nothing -> wasted compute, no learning
            signal,
          - or the gradient updates LoRA in the wrong direction for
            contexts that should not have been associated with that slot.

        Now: each of the top-k slots is gated INDIVIDUALLY against
        `threshold`. Only slots whose own attention mass clears the
        threshold are recorded. The early "max < threshold" check is
        still useful as a fast path (no slot attended -> skip topk).

        Defensive clone: each pushed ReplayContext has its own tensor
        storage so independent slot buffers cannot be cross-corrupted
        by an in-place op on the original chunk tensors.
        """
        if u_prefix_bar <= 0:
            return
        mass = alpha_p.sum(dim=0)                             # (N_m,)
        if mass.max().item() < threshold:
            return
        topk = torch.topk(mass, min(k, self.n_slots))
        for slot_id, mass_val in zip(
            topk.indices.tolist(), topk.values.tolist()
        ):
            if mass_val < threshold:
                continue
            self.buffers[slot_id].append(ReplayContext(
                input_ids=ctx.input_ids.clone(),
                targets=ctx.targets.clone(),
                tau_t=ctx.tau_t,
                delta_tau=ctx.delta_tau,
            ))

    def sample(self, slot_id: int, n: int) -> list[ReplayContext]:
        buf = self.buffers[slot_id]
        if not buf:
            return []
        return self.rng.sample(list(buf), k=min(n, len(buf)))

    def size(self, slot_id: int) -> int:
        return len(self.buffers[slot_id])

    def total_size(self) -> int:
        return sum(len(b) for b in self.buffers)

    def stats(self) -> dict:
        sizes = [len(b) for b in self.buffers]
        nonempty = sum(1 for s in sizes if s > 0)
        return {
            "n_slots_with_data": nonempty,
            "total_contexts": sum(sizes),
            "max_per_slot": max(sizes) if sizes else 0,
            "mean_per_slot": (sum(sizes) / len(sizes)) if sizes else 0.0,
        }
