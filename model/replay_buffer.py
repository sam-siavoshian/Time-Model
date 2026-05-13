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
        """Push ctx into the buffers of the top-k slots by attention mass,
        gated on u_prefix_bar > 0."""
        if u_prefix_bar <= 0:
            return
        # Sum attention mass per slot across all prefix queries
        mass = alpha_p.sum(dim=0)                             # (N_m,)
        if mass.max().item() < threshold:
            return
        topk = torch.topk(mass, min(k, self.n_slots))
        for slot_id in topk.indices.tolist():
            self.buffers[slot_id].append(ctx)

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
