"""Orchestrator: generate full Temporal Latent World dataset across all bins.

Train bins (1k-8k tokens): 45k streams.
Test bins  (16k-128k tokens): 3.7k streams.
Total: ~48.7k streams. (Held-out validation = test split below.)

Each bin gets its own seed range so seeds never collide across bins.

Usage:
  uv run python -m data_gen.generate_all_latent_world --split train
  uv run python -m data_gen.generate_all_latent_world --split test
  uv run python -m data_gen.generate_all_latent_world --split all
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from data_gen.latent_world_sim import generate_stream

# Bin spec: (target_tokens, n_streams, seed_base, filename)
TRAIN_BINS = [
    (1024,   15000, 1_000_000, "train_1k.jsonl"),
    (2048,   15000, 2_000_000, "train_2k.jsonl"),
    (4096,   10000, 3_000_000, "train_4k.jsonl"),
    (8192,    5000, 4_000_000, "train_8k.jsonl"),
]
TEST_BINS = [
    (16384,   2000, 5_000_000, "test_16k.jsonl"),
    (32768,   1000, 6_000_000, "test_32k.jsonl"),
    (65536,    500, 7_000_000, "test_64k.jsonl"),
    (131072,   200, 8_000_000, "test_128k.jsonl"),
]
VALID_BINS = [
    (1024,    500, 9_000_000, "valid_1k.jsonl"),
    (4096,    500, 9_100_000, "valid_4k.jsonl"),
    (16384,   100, 9_200_000, "valid_16k.jsonl"),
]


def run_bin(target_tokens: int, n_streams: int, seed_base: int, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_path, "w") as f:
        for i in tqdm(range(n_streams), desc=f"{out_path.name}", leave=False):
            stream = generate_stream(seed=seed_base + i, target_tokens=target_tokens)
            f.write(stream.model_dump_json() + "\n")
    dt = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"  {out_path.name}: {n_streams} streams, {size_mb:.1f} MB, {dt:.1f}s ({n_streams/dt:.1f} streams/s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test", "valid", "all"], default="train")
    p.add_argument("--out-dir", type=str, default="data/latent_world")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    splits = []
    if args.split in ("train", "all"):
        splits.extend(TRAIN_BINS)
    if args.split in ("test", "all"):
        splits.extend(TEST_BINS)
    if args.split in ("valid", "all"):
        splits.extend(VALID_BINS)

    total_streams = sum(n for _, n, _, _ in splits)
    print(f"Generating {len(splits)} bins, {total_streams} total streams")
    t_overall = time.time()
    for target_tokens, n, seed_base, fname in splits:
        print(f"\n[{target_tokens // 1024}k tokens bin]")
        run_bin(target_tokens, n, seed_base, out_dir / fname)
    print(f"\nTotal time: {(time.time() - t_overall) / 60:.1f} min")


if __name__ == "__main__":
    main()
