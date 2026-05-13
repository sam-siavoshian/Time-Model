"""Dataset loaders for tokenized binary caches.

Each cache:
  <prefix>.tokens.bin    flat uint16 array of GPT-2 token IDs
  <prefix>.boundaries.npy int64 (N, 2) array — (start, end) per example
  <prefix>.meta.json     summary stats

Two access patterns:
  - SequentialChunkDataset: iterate per-example, then per-chunk-of-L tokens.
    Yields chunks for the IPCN forward loop. Preserves stream order so memory
    persists across chunks within one example.
  - MixedDataset: round-robin or weighted mix across multiple caches (for
    Phase 3 mixed-LM training).

For v1 we use a stateless iterator interface that returns chunks tagged with
example_id + chunk_idx so the training loop knows when to reset memory.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch


@dataclass
class TokenizedCache:
    """Memmap-backed view of one tokenized JSONL cache."""

    prefix: str

    def __post_init__(self):
        tokens_path = Path(self.prefix + ".tokens.bin")
        bounds_path = Path(self.prefix + ".boundaries.npy")
        meta_path = Path(self.prefix + ".meta.json")
        if not tokens_path.exists():
            raise FileNotFoundError(f"missing: {tokens_path}")
        self.tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
        self.boundaries = np.load(bounds_path)                              # (N, 2)
        self.meta = json.loads(meta_path.read_text())

    @property
    def n_examples(self) -> int:
        return len(self.boundaries)

    @property
    def n_tokens(self) -> int:
        return int(self.tokens.size)

    def get_example(self, idx: int) -> np.ndarray:
        s, e = self.boundaries[idx]
        return np.asarray(self.tokens[s:e])

    def __repr__(self) -> str:
        return f"TokenizedCache({self.prefix}, examples={self.n_examples}, tokens={self.n_tokens})"


@dataclass
class ChunkBatch:
    """Single training chunk. v1 batch size = 1 (per-stream memory)."""

    input_ids: torch.Tensor                                                 # (L,)
    targets: torch.Tensor                                                   # (L,) = input_ids shifted by 1
    example_id: int
    chunk_idx: int                                                          # 0-indexed within example
    is_first_chunk: bool                                                    # reset memory at chunk_idx == 0
    is_last_chunk: bool
    total_chunks_in_example: int
    cache_name: str
    # Time metadata (set by caller when known; defaults here for synthetic)
    tau_t: float = 0.0
    delta_tau: float = 1.0
    gap_flag: float = 0.0


class SequentialChunkDataset:
    """Stream chunks from one cache. Resets memory between examples.

    For now: example tokens are split into consecutive non-overlapping chunks
    of length L. Time metadata is synthesized per-chunk linearly; callers can
    override for the Latent World streams that carry real tau values inside
    the rendered text.
    """

    def __init__(
        self,
        cache: TokenizedCache,
        chunk_length: int,
        shuffle_examples: bool = True,
        seed: int = 0,
        time_delta_per_chunk: float = 1.0,
    ):
        self.cache = cache
        self.L = chunk_length
        self.shuffle = shuffle_examples
        self.rng = random.Random(seed)
        self.time_delta = time_delta_per_chunk

    def __iter__(self) -> Iterator[ChunkBatch]:
        order = list(range(self.cache.n_examples))
        if self.shuffle:
            self.rng.shuffle(order)
        for ex_id in order:
            ex_tokens = self.cache.get_example(ex_id)
            if len(ex_tokens) < 2:
                continue
            # Targets = input_ids shifted by 1, so we trim to multiples of L from full
            n_chunks = max(1, (len(ex_tokens) - 1) // self.L)
            for c in range(n_chunks):
                s = c * self.L
                e = min(s + self.L + 1, len(ex_tokens))                     # +1 for target shift
                segment = ex_tokens[s:e]
                if len(segment) < 2:
                    continue
                inputs = segment[:-1].astype(np.int64)
                targets = segment[1:].astype(np.int64)
                # Pad to L if last chunk is short.
                # IMPORTANT: target padding uses -100 (PyTorch ignore_index),
                # NOT 0. Token 0 is <|endoftext|> in GPT-2 vocab; padding
                # with 0 contaminates LM loss by reinforcing EOS prediction
                # at arbitrary positions. -100 is masked by F.cross_entropy.
                if len(inputs) < self.L:
                    pad = self.L - len(inputs)
                    inputs = np.concatenate([inputs, np.zeros(pad, dtype=np.int64)])
                    targets = np.concatenate([targets, np.full(pad, -100, dtype=np.int64)])
                yield ChunkBatch(
                    input_ids=torch.from_numpy(inputs),
                    targets=torch.from_numpy(targets),
                    example_id=ex_id,
                    chunk_idx=c,
                    is_first_chunk=(c == 0),
                    is_last_chunk=(c == n_chunks - 1),
                    total_chunks_in_example=n_chunks,
                    cache_name=self.cache.prefix,
                    tau_t=float(c * self.time_delta),
                    delta_tau=float(self.time_delta),
                    gap_flag=0.0,
                )


class MixedDataset:
    """Round-robin or weighted mix across caches. Each yielded example uses one
    cache at a time; memory resets between examples (so cache mix is at the
    example level, not chunk level)."""

    def __init__(
        self,
        datasets: list[SequentialChunkDataset],
        weights: Optional[list[float]] = None,
        seed: int = 0,
    ):
        if weights is None:
            weights = [1.0] * len(datasets)
        assert len(datasets) == len(weights)
        self.datasets = datasets
        self.weights = [w / sum(weights) for w in weights]
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[ChunkBatch]:
        iters = [iter(d) for d in self.datasets]
        while True:
            choice = self.rng.choices(range(len(iters)), weights=self.weights, k=1)[0]
            try:
                yield next(iters[choice])
            except StopIteration:
                # Re-init that iterator
                iters[choice] = iter(self.datasets[choice])
                yield next(iters[choice])


# ---------- Helper to load default caches ----------

def load_default_train_caches(root: str = "data/tokenized") -> dict[str, TokenizedCache]:
    """Load the caches used in Phase 0 sanity training."""
    paths = {
        "latent_world_train_1k": f"{root}/latent_world/train_1k",
        "latent_world_train_2k": f"{root}/latent_world/train_2k",
        "ambiguity_train":       f"{root}/ambiguity/train",
        "consolidation":         f"{root}/consolidation/ladder_train",
        "real_text":             f"{root}/real_text/gutenberg",
    }
    return {name: TokenizedCache(p) for name, p in paths.items() if Path(p + ".tokens.bin").exists()}
